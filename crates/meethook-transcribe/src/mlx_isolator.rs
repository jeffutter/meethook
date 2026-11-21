//! The voice isolator's network on Apple's GPU through MLX (the `mlx` feature, macOS only).
//!
//! [`MlxNetwork`] maps the STFT of an audio window to the complex correction the voice is formed
//! from, the same contract as the ONNX graph ([`crate::VOICE_ISOLATOR_MODEL`]) so that
//! [`crate::VoiceIsolator`] owns everything around it (STFT, windows, crossfades) for both
//! engines. This is a port of the forward pass of `Network.swift` and `Kernels.swift` from
//! <https://github.com/starkdmi/VoiceIsolatorMLX> (Apache-2.0, Copyright 2026 Dmitry Starkov;
//! the model is Inworld AI's voice isolator, Apache-2.0), reading the fp16 weights that project
//! converts ([`crate::VOICE_ISOLATOR_MLX_MODEL`]). Not ported: the realtime variant, 8-bit
//! quantization, and the Swift package's 4096-token chunking of each layer (every window here is
//! at most 801 frames, about 2 GB peak).
//!
//! # The model
//!
//! A band-split dual-path network over the STFT. The 1025 bins are cut into 46 bands, each
//! normalised and projected to a 512-wide token, so a window is one `[bands * frames, 512]`
//! matrix for the whole backbone. Seven blocks (transformer, Mamba, transformer, ...) each run a
//! sequence layer along time (every band is a sequence of frames) and then along bands (every
//! frame is a sequence of bands). The sequence layers address the one matrix through index maps
//! ([`Layout`]) instead of transposing it. The decoder turns each token back into per-bin mask
//! and residual values, and [`MlxNetwork::correction`] forms the bounded complex correction.
//!
//! The five custom Metal kernels (head split with RoPE, head merge with exclusive attention,
//! the depthwise convolution, the selective scan, GELU) are what make this fast: the ONNX
//! export of the same Mamba scans has 15,568 nodes, and CoreML either declines or runs out of
//! memory on it. Arithmetic inside the kernels is float32; activations are fp16, except the
//! Mamba output projections, which exceed fp16's range before their norm and stay float32.
//!
//! # Speed
//!
//! Measured on an M1 Pro: 129 s of audio in 31 s (4.2x realtime, including the STFT and inverse STFT
//! and the first call's kernel compile), about 1.3 s per 8 s window once warm, against about 16 s
//! per window for the ONNX graph on the CPU. The first
//! call in a process takes about 3 s more, while MLX compiles the Metal kernels. The weights are
//! mapped lazily, so loading is near-instant.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::fast::{
    self, MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput,
    ScaledDotProductAttentionMask, TemplateArg,
};
use mlx_rs::ops::indexing::{Ellipsis, IndexOp, take_axis};
use mlx_rs::{Array, Dtype, ops};

use crate::mlx_kernels as src;
use crate::{Error, Result};

type R<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const SAMPLE_RATE: usize = 48_000;
const N_FFT: usize = 2048;
const BINS: usize = N_FFT / 2 + 1;
const WIDTH: i32 = 512;
const HEADS: i32 = 8;
const HEAD_DIM: i32 = 64;
/// Channels of a Mamba layer's inner state: twice the width.
const INNER: i32 = 1024;
const MAMBA_STATE: i32 = 16;
const MAMBA_DT_RANK: i32 = 32;
const MAMBA_TAPS: i32 = 4;
const LAYER_NORM_EPS: f32 = 1e-5;
/// How far the correction may move a bin, relative to its magnitude.
const CORRECTION_BOUND: f32 = 0.5;
/// Times a window is tried before its error is returned.
const ATTEMPTS: u32 = 3;
const F16: Dtype = Dtype::Float16;
const F32: Dtype = Dtype::Float32;
/// Block kinds in order: `true` is transformer, `false` is Mamba.
const BLOCKS: [bool; 7] = [true, false, true, false, true, false, true];

/// One compiled kernel per use. Kernels compile on first evaluation and MLX caches them, so a
/// [`MlxNetwork`] builds each exactly once.
struct Kernels {
    gelu: MetalKernel,
    split_rope: MetalKernel,
    split_plain: MetalKernel,
    merge: MetalKernel,
    merge_exclusive: MetalKernel,
    conv: MetalKernel,
    scan: MetalKernel,
}

impl Kernels {
    fn new() -> R<Self> {
        let half = "half";
        let build = |name: &str,
                     inputs: &[&str],
                     outputs: &[&str],
                     body: &str,
                     header: &str,
                     out: &str|
         -> R<MetalKernel> {
            let body = body.replace("{T}", half).replace("{Y}", out);
            Ok(MetalKernel::new(
                &format!("{name}_{half}_{out}"),
                inputs,
                outputs,
                &body,
                MetalKernelOptions {
                    header: format!("{}{header}", src::TOKEN),
                    ensure_row_contiguous: true,
                    atomic_outputs: false,
                },
            )?)
        };
        let split = |rope: &str| src::SPLIT.replace("{rope}", rope);
        let merge = |exclusive: &str| src::MERGE.replace("{exclusive}", exclusive);
        let scan = src::SCAN
            .replace("{h_init}", "0.0f")
            .replace("{h_store}", "");
        Ok(Self {
            gelu: build("vi_gelu", &["x"], &["y"], src::GELU, src::ERF, half)?,
            split_rope: build(
                "vi_split_heads_rope",
                &["qkv", "cos_t", "sin_t", "params"],
                &["q", "k", "v"],
                &split(src::ROPE),
                "",
                half,
            )?,
            split_plain: build(
                "vi_split_heads",
                &["qkv", "params"],
                &["q", "k", "v"],
                &split(""),
                "",
                half,
            )?,
            merge: build(
                "vi_merge_heads",
                &["o", "params"],
                &["out"],
                &merge(""),
                "",
                half,
            )?,
            merge_exclusive: build(
                "vi_merge_heads_excl",
                &["o", "v", "params"],
                &["out"],
                &merge(src::EXCLUSIVE),
                "",
                half,
            )?,
            conv: build(
                "vi_conv_silu_0",
                &["u", "weight", "bias", "params"],
                &["uc"],
                &src::CONV.replace("{history}", ""),
                "",
                half,
            )?,
            // The scan reads fp16 and writes float32: its output feeds the float32 projection.
            scan: build(
                "vi_scan_00",
                &["uc", "z", "dbc", "A", "D", "Wdt", "bdt", "params"],
                &["y"],
                &scan,
                src::SOFTPLUS,
                "float",
            )?,
        })
    }
}

/// Runs `kernel` over `grid` threads in groups of `group`; one array per `outputs` entry.
fn launch(
    kernel: &MetalKernel,
    inputs: &[&Array],
    outputs: &[(Vec<i32>, Dtype)],
    grid: (i32, i32, i32),
    group: (i32, i32, i32),
    template: &[(&str, i32)],
) -> R<Vec<Array>> {
    let outputs: Vec<MetalKernelOutput> = outputs
        .iter()
        .map(|(shape, dtype)| MetalKernelOutput {
            shape: shape.clone(),
            dtype: *dtype,
        })
        .collect();
    let options = MetalKernelApplyOptions {
        template_args: template
            .iter()
            .map(|(name, value)| (name.to_string(), TemplateArg::Int(*value)))
            .collect(),
        ..Default::default()
    };
    // SAFETY: every grid in this module covers exactly the token and channel ranges its kernel
    // indexes, the same launches as the Swift package these kernels come from; no kernel writes
    // an input, and each writes every output element.
    Ok(unsafe { kernel.apply(inputs, &outputs, grid, group, options) }?)
}

/// How a sequence axis maps onto the canonical `[bands, frames]` token order:
/// `token = (row / inner) * outer + (row % inner) * inner_stride + position * seq_stride`.
struct Layout {
    rows: i32,
    length: i32,
    tokens: i32,
    params: Array,
}

impl Layout {
    fn new(rows: i32, length: i32, inner: i32, outer: i32, inner_stride: i32, seq: i32) -> Self {
        let tokens = rows * length;
        let params = Array::from_slice(&[length, inner, outer, inner_stride, seq, tokens], &[6]);
        Self {
            rows,
            length,
            tokens,
            params,
        }
    }

    /// Every band is a row of consecutive frames.
    fn time(bands: i32, frames: i32) -> Self {
        Self::new(bands, frames, 1, frames, 0, 1)
    }

    /// Every frame is a row across the bands.
    fn band(bands: i32, frames: i32) -> Self {
        Self::new(frames, bands, frames, bands * frames, 1, frames)
    }
}

struct Linear {
    weight: Array,
    weight_t: Array,
    bias: Option<Array>,
}

impl Linear {
    fn new(weights: &HashMap<String, Array>, prefix: &str) -> R<Self> {
        let weight = tensor(weights, &format!("{prefix}.weight"))?;
        let weight_t = weight.swap_axes(-1, -2)?;
        Ok(Self {
            weight,
            weight_t,
            bias: weights.get(&format!("{prefix}.bias")).cloned(),
        })
    }

    fn matmul(&self, x: &Array) -> R<Array> {
        Ok(x.matmul(&self.weight_t)?)
    }

    /// `c + x W^T`.
    fn addmm(&self, c: &Array, x: &Array) -> R<Array> {
        if self.weight.ndim() == 2 && x.ndim() == 2 {
            return Ok(ops::addmm(c, x, &self.weight_t, None, None)?);
        }
        Ok(c + self.matmul(x)?)
    }

    fn apply(&self, x: &Array) -> R<Array> {
        match &self.bias {
            Some(bias) => self.addmm(bias, x),
            None => self.matmul(x),
        }
    }

    fn bias(&self) -> &Array {
        self.bias
            .as_ref()
            .expect("this layer has a bias in every checkpoint")
    }
}

struct LayerNorm {
    weight: Array,
    bias: Array,
}

impl LayerNorm {
    fn new(weights: &HashMap<String, Array>, prefix: &str) -> R<Self> {
        Ok(Self {
            weight: tensor(weights, &format!("{prefix}.weight"))?,
            bias: tensor(weights, &format!("{prefix}.bias"))?,
        })
    }

    fn apply(&self, x: &Array) -> R<Array> {
        Ok(fast::layer_norm(
            x,
            &self.weight,
            &self.bias,
            LAYER_NORM_EPS,
        )?)
    }
}

fn tensor(weights: &HashMap<String, Array>, name: &str) -> R<Array> {
    Ok(weights
        .get(name)
        .ok_or_else(|| format!("the weights lack `{name}`"))?
        .clone())
}

fn gelu(kernels: &Kernels, x: &Array) -> R<Array> {
    let size = x.size() as i32;
    let out = launch(
        &kernels.gelu,
        &[x],
        &[(x.shape().to_vec(), x.dtype())],
        (size, 1, 1),
        (256, 1, 1),
        &[],
    )?;
    Ok(out.into_iter().next().expect("one output"))
}

/// A pre-norm transformer layer along one sequence axis.
struct Attention {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
    /// Exclusive self attention: the part of the output along the value is removed.
    exclusive: bool,
}

impl Attention {
    fn new(weights: &HashMap<String, Array>, prefix: &str, exclusive: bool) -> R<Self> {
        Ok(Self {
            norm1: LayerNorm::new(weights, &format!("{prefix}.norm1"))?,
            norm2: LayerNorm::new(weights, &format!("{prefix}.norm2"))?,
            qkv: Linear::new(weights, &format!("{prefix}.qkv"))?,
            proj: Linear::new(weights, &format!("{prefix}.proj"))?,
            fc1: Linear::new(weights, &format!("{prefix}.fc1"))?,
            fc2: Linear::new(weights, &format!("{prefix}.fc2"))?,
            exclusive,
        })
    }

    /// `rope` is the (cos, sin) tables of the time axis; the band axis has none.
    fn apply(
        &self,
        kernels: &Kernels,
        x: &Array,
        layout: &Layout,
        rope: Option<(&Array, &Array)>,
    ) -> R<Array> {
        let qkv = self.qkv.apply(&self.norm1.apply(x)?)?;
        let shape = vec![layout.rows, HEADS, layout.length, HEAD_DIM];
        let outputs = [(shape.clone(), F16), (shape.clone(), F16), (shape, F16)];
        let grid = (HEAD_DIM / 2, HEADS, layout.tokens);
        let group = (HEAD_DIM / 2, HEADS, 1);
        let template = [("H", HEADS), ("D", HEAD_DIM)];
        let heads = match rope {
            Some((cos, sin)) => launch(
                &kernels.split_rope,
                &[&qkv, cos, sin, &layout.params],
                &outputs,
                grid,
                group,
                &template,
            )?,
            None => launch(
                &kernels.split_plain,
                &[&qkv, &layout.params],
                &outputs,
                grid,
                group,
                &template,
            )?,
        };
        let [q, k, v] = <[Array; 3]>::try_from(heads).map_err(|_| "split_heads returns q, k, v")?;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let attended = fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            scale,
            None::<ScaledDotProductAttentionMask>,
            None::<&Array>,
        )?;

        let (kernel, inputs): (&MetalKernel, Vec<&Array>) = if self.exclusive {
            (
                &kernels.merge_exclusive,
                vec![&attended, &v, &layout.params],
            )
        } else {
            (&kernels.merge, vec![&attended, &layout.params])
        };
        let merged = launch(
            kernel,
            &inputs,
            &[(vec![layout.tokens, HEADS * HEAD_DIM], F16)],
            (32, HEADS, layout.tokens),
            (32, HEADS, 1),
            &template,
        )?
        .remove(0);
        let y = self.proj.addmm(&(x + self.proj.bias()), &merged)?;
        let hidden = gelu(kernels, &self.fc1.apply(&self.norm2.apply(&y)?)?)?;
        self.fc2.addmm(&(&y + self.fc2.bias()), &hidden)
    }
}

/// Selective state-space layers along one axis: a forward and a reverse direction stacked, their
/// output projections folded together at conversion.
struct Mamba {
    in_u: Linear,
    in_z: Linear,
    x_proj: Linear,
    out: Linear,
    skip: Linear,
    conv_weight: Array,
    conv_bias: Array,
    dt_weight: Array,
    dt_bias: Array,
    a: Array,
    d: Array,
    norm: LayerNorm,
}

impl Mamba {
    fn new(weights: &HashMap<String, Array>, prefix: &str) -> R<Self> {
        let get = |name: &str| tensor(weights, &format!("{prefix}.{name}"));
        Ok(Self {
            in_u: Linear::new(weights, &format!("{prefix}.in_u"))?,
            in_z: Linear::new(weights, &format!("{prefix}.in_z"))?,
            x_proj: Linear::new(weights, &format!("{prefix}.x_proj"))?,
            out: Linear::new(weights, &format!("{prefix}.out"))?,
            skip: Linear::new(weights, &format!("{prefix}.skip"))?,
            conv_weight: get("conv.weight")?,
            conv_bias: get("conv.bias")?,
            dt_weight: get("dt.weight")?,
            dt_bias: get("dt.bias")?,
            a: get("A")?,
            d: get("D")?,
            norm: LayerNorm::new(weights, &format!("{prefix}.norm"))?,
        })
    }

    /// `x + norm(projection)` over the tokens `x [tokens, width]`.
    fn apply(&self, kernels: &Kernels, x: &Array, layout: &Layout) -> R<Array> {
        let u = self.in_u.matmul(x)?;
        let runs = (layout.length + 7) / 8;
        let uc = launch(
            &kernels.conv,
            &[&u, &self.conv_weight, &self.conv_bias, &layout.params],
            &[(vec![2, layout.tokens, INNER], F16)],
            (INNER, layout.rows * runs, 2),
            (256, 1, 1),
            &[("KW", MAMBA_TAPS), ("DIRS", 2)],
        )?
        .remove(0);
        let dbc = self.x_proj.matmul(&uc)?;
        let z = self.in_z.matmul(x)?;
        let y = launch(
            &kernels.scan,
            &[
                &uc,
                &z,
                &dbc,
                &self.a,
                &self.d,
                &self.dt_weight,
                &self.dt_bias,
                &layout.params,
            ],
            &[(vec![layout.tokens, 2 * INNER], F32)],
            (INNER, layout.rows, 2),
            (256, 1, 1),
            &[("S", MAMBA_STATE), ("R", MAMBA_DT_RANK), ("DIRS", 2)],
        )?
        .remove(0);
        // The projection before the norm reaches about 1e5: float32 throughout.
        let x32 = x.as_dtype(F32)?;
        let residual = self.skip.addmm(self.out.bias(), &x32)?;
        let projected = self.out.addmm(&residual, &y)?;
        Ok(x + self.norm.apply(&projected)?.as_dtype(x.dtype())?)
    }
}

enum Block {
    Transformer { time: Attention, band: Attention },
    Mamba { time: Mamba, band: Mamba },
}

/// Bin ranges `[start, end)` of the bands, covering every bin once.
fn band_edges() -> Vec<(usize, usize)> {
    let hz_per_bin = SAMPLE_RATE as f64 / N_FFT as f64;
    // (band width in Hz, upper edge in Hz) of each stretch of equal-width bands.
    let plan = [
        (100.0, 1000.0),
        (250.0, 4000.0),
        (500.0, 8000.0),
        (1000.0, 16000.0),
        (2000.0, 24000.0),
    ];
    let (mut edges, mut start) = (Vec::new(), 0usize);
    for (width_hz, upto_hz) in plan {
        // Python's round(): half to even.
        let width = ((width_hz / hz_per_bin).round_ties_even() as usize).max(1);
        let upto = BINS.min((upto_hz / hz_per_bin).round_ties_even() as usize);
        while start < upto {
            let end = (start + width).min(upto);
            edges.push((start, end));
            start = end;
        }
    }
    if start < BINS {
        edges.push((start, BINS));
    }
    edges
}

/// Cosine and sine `[frames][HEAD_DIM / 2]`, flat, of the rotary positions: float32 inverse
/// frequencies and angles as PyTorch forms them, the trigonometry in float64.
fn rope_values(frames: usize) -> (Vec<f32>, Vec<f32>) {
    let half = (HEAD_DIM / 2) as usize;
    let inverse: Vec<f32> = (0..half)
        .map(|i| 10_000f64.powf(f64::from(-(i as f32 / half as f32))) as f32)
        .collect();
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for position in 0..frames {
        for inverse in &inverse {
            let angle = f64::from(position as f32 * inverse);
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}

fn rope_tables(frames: i32) -> (Array, Array) {
    let (cos, sin) = rope_values(frames as usize);
    let shape = [frames, HEAD_DIM / 2];
    (
        Array::from_slice(&cos, &shape),
        Array::from_slice(&sin, &shape),
    )
}

/// The loaded network: weights, the band index tables and the compiled kernels.
pub(crate) struct MlxNetwork {
    kernels: Kernels,
    bands: i32,
    in_pad: i32,
    gather: Array,
    valid: Array,
    counts: Array,
    keep: Array,
    enc_norm_weight: Array,
    enc_norm_bias: Array,
    enc_weight_t: Array,
    enc_bias: Array,
    blocks: Vec<Block>,
    dec_scale: Array,
    dec_shift: Array,
    dec_norm_weight: Array,
    dec_norm_bias: Array,
    dec_fc1: Linear,
    dec_fc1_bias: Array,
    dec_fc2: Linear,
    dec_fc2_bias: Array,
}

impl MlxNetwork {
    /// Loads the fp16 weights of [`crate::VOICE_ISOLATOR_MLX_MODEL`].
    pub(crate) fn load(path: &Path) -> Result<Self> {
        Self::try_load(path).map_err(|e| Error::VoiceIsolator(format!("MLX: {e}")))
    }

    fn try_load(path: &Path) -> R<Self> {
        let weights = Array::load_safetensors(path)?;
        let get = |name: &str| tensor(&weights, name);
        let edges = band_edges();
        let bands = edges.len();
        let widest = edges
            .iter()
            .map(|(s, e)| e - s)
            .max()
            .expect("there are bands");
        let (in_pad, out_pad) = (2 * widest, 4 * widest);

        // The constants above are those of the pinned fp16 file. Fail clearly on another layout
        // (8-bit quantized weights, or an input padded to a multiple of the group size) rather
        // than somewhere inside a matmul.
        let encoder = get("encoder.weight")?;
        if weights.contains_key("blocks.0.time.qkv.scales")
            || encoder.shape() != [bands as i32, WIDTH, in_pad as i32]
        {
            return Err(format!(
                "these weights are not the unquantized fp16 layout this build loads \
                 (encoder.weight is {:?}, expected {:?})",
                encoder.shape(),
                [bands as i32, WIDTH, in_pad as i32]
            )
            .into());
        }

        // Indices into the interleaved (re, im) components with a trailing zero column, the
        // validity mask of each padded band, and the decoder outputs that are real bins.
        let mut gather = vec![(2 * BINS) as i32; bands * in_pad];
        let mut valid = vec![0f32; bands * in_pad];
        let mut keep = Vec::new();
        for (band, (start, end)) in edges.iter().enumerate() {
            for j in 0..2 * (end - start) {
                gather[band * in_pad + j] = (2 * start + j) as i32;
                valid[band * in_pad + j] = 1.0;
            }
            keep.extend((0..4 * (end - start)).map(|j| (band * out_pad + j) as i32));
        }
        let counts: Vec<f32> = edges.iter().map(|(s, e)| (2 * (e - s)) as f32).collect();

        let mut blocks = Vec::new();
        for (i, transformer) in BLOCKS.iter().enumerate() {
            blocks.push(if *transformer {
                Block::Transformer {
                    // Exclusive self attention along time only.
                    time: Attention::new(&weights, &format!("blocks.{i}.time"), true)?,
                    band: Attention::new(&weights, &format!("blocks.{i}.band"), false)?,
                }
            } else {
                Block::Mamba {
                    time: Mamba::new(&weights, &format!("blocks.{i}.time"))?,
                    band: Mamba::new(&weights, &format!("blocks.{i}.freq"))?,
                }
            });
        }
        Ok(Self {
            kernels: Kernels::new()?,
            bands: bands as i32,
            in_pad: in_pad as i32,
            gather: Array::from_slice(&gather, &[gather.len() as i32]),
            valid: Array::from_slice(&valid, &[bands as i32, in_pad as i32]),
            counts: Array::from_slice(&counts, &[bands as i32, 1]),
            keep: Array::from_slice(&keep, &[keep.len() as i32]),
            enc_norm_weight: get("encoder.norm.weight")?,
            enc_norm_bias: get("encoder.norm.bias")?,
            enc_weight_t: get("encoder.weight")?.swap_axes(-1, -2)?,
            enc_bias: get("encoder.bias")?.expand_dims(1)?,
            blocks,
            dec_scale: get("decoder.scale")?,
            dec_shift: get("decoder.shift")?,
            dec_norm_weight: get("decoder.norm.weight")?.expand_dims(1)?,
            dec_norm_bias: get("decoder.norm.bias")?.expand_dims(1)?,
            dec_fc1: Linear::new(&weights, "decoder.fc1")?,
            dec_fc1_bias: get("decoder.fc1.bias")?.expand_dims(1)?,
            dec_fc2: Linear::new(&weights, "decoder.fc2")?,
            dec_fc2_bias: get("decoder.fc2.bias")?.expand_dims(1)?,
        })
    }

    /// The complex correction of a window's spectrum. Both are flat `[frames][bin][re, im]`
    /// `f32`, the layout of the ONNX graph's `spec` input and `correction` output.
    ///
    /// A window that fails is run again, up to `ATTEMPTS` times in all. The GPU can refuse a
    /// command buffer for reasons that pass ("Impacting Interactivity" when the screen or another
    /// process needs it), and a window is about a second of work, so retrying is cheap.
    pub(crate) fn correction(&self, spec: &[f32]) -> Result<Vec<f32>> {
        let mut failure = None;
        for attempt in 0..ATTEMPTS {
            match self.try_correction(spec) {
                Ok(correction) => return Ok(correction),
                Err(e) => {
                    failure = Some(e);
                    if attempt + 1 < ATTEMPTS {
                        std::thread::sleep(std::time::Duration::from_millis(250 << attempt));
                    }
                }
            }
        }
        let e = failure.expect("ATTEMPTS is at least 1");
        Err(Error::VoiceIsolator(format!(
            "MLX, after {ATTEMPTS} attempts: {e}"
        )))
    }

    fn try_correction(&self, spec: &[f32]) -> R<Vec<f32>> {
        let frames = (spec.len() / (BINS * 2)) as i32;
        let spec = Array::from_slice(spec, &[1, frames, BINS as i32, 2]);
        let tokens = self.backbone(&self.encode(&spec, frames)?, frames)?;
        let raw = self.decode(&tokens, frames)?;

        // The bounded complex mask and residual applied to the spectrum.
        let two = Array::from_f32(2.0);
        let (re, im) = (spec.index((Ellipsis, 0)), spec.index((Ellipsis, 1)));
        let magnitude = ops::sqrt(&re * &re + &im * &im)? * Array::from_f32(CORRECTION_BOUND);
        let mask_re = ops::tanh(raw.index((Ellipsis, 0)))? * &two;
        let mask_im = ops::tanh(raw.index((Ellipsis, 1)))? * &two;
        let res_re = ops::tanh(raw.index((Ellipsis, 2)))?;
        let res_im = ops::tanh(raw.index((Ellipsis, 3)))?;
        let norm = ops::maximum(
            ops::sqrt(&res_re * &res_re + &res_im * &res_im)?,
            Array::from_f32(1.0),
        )?;
        let out_re = &re * &mask_re - &im * &mask_im + (&res_re / &norm) * &magnitude;
        let out_im = &re * &mask_im + &im * &mask_re + (&res_im / &norm) * &magnitude;
        let out = ops::stack(&[&out_re, &out_im], -1)?;
        out.eval()?;
        let correction = out.as_slice::<f32>().to_vec();
        // MLX keeps freed buffers for reuse, keyed by size. Spans of every length make every
        // window's buffers a new size, so without this the cache grows without bound (19 GB
        // after 40% of a 30 minute track).
        mlx_rs::memory::clear_cache()?;
        Ok(correction)
    }

    /// Tokens `[bands * frames, WIDTH]` of the spectrum `[1, frames, bins, 2]`.
    fn encode(&self, spec: &Array, frames: i32) -> R<Array> {
        let components = spec.reshape(&[1, frames, 2 * BINS as i32])?;
        let zero = ops::zeros::<f32>(&[1, frames, 1])?;
        let components = ops::concatenate(&[&components, &zero], 2)?;
        let x = take_axis(&components, &self.gather, 2)?.reshape(&[
            1,
            frames,
            self.bands,
            self.in_pad,
        ])?;
        let mean = x.sum_axis(-1, true)? / &self.counts;
        let centered = (&x - &mean) * &self.valid;
        let variance = (&centered * &centered).sum_axis(-1, true)? / &self.counts;
        let x = &centered * ops::rsqrt(&variance + Array::from_f32(LAYER_NORM_EPS))?;
        let x = &x * &self.enc_norm_weight + &self.enc_norm_bias;
        let x = x.transpose_axes(&[0, 2, 1, 3])?.as_dtype(F16)?;
        let tokens = &self.enc_bias + x.matmul(&self.enc_weight_t)?;
        Ok(tokens.reshape(&[-1, WIDTH])?)
    }

    fn backbone(&self, tokens: &Array, frames: i32) -> R<Array> {
        let (cos, sin) = rope_tables(frames);
        let time = Layout::time(self.bands, frames);
        let band = Layout::band(self.bands, frames);
        let kernels = &self.kernels;
        let mut x = tokens.clone();
        // Each sequence layer is evaluated before the next is built. MLX would otherwise commit
        // the whole window as a few long GPU command buffers, and macOS kills a command buffer
        // that runs long while the GPU is also drawing the screen or serving another process
        // ("Impacting Interactivity"). Layer-sized buffers stay well under that; the extra
        // synchronisation is 14 waits in a window of about 1.4 s.
        for block in &self.blocks {
            match block {
                Block::Transformer {
                    time: along_time,
                    band: along_band,
                } => {
                    x = along_time.apply(kernels, &x, &time, Some((&cos, &sin)))?;
                    x.eval()?;
                    x = along_band.apply(kernels, &x, &band, None)?;
                }
                Block::Mamba {
                    time: along_time,
                    band: along_band,
                } => {
                    x = along_time.apply(kernels, &x, &time)?;
                    x.eval()?;
                    x = along_band.apply(kernels, &x, &band)?;
                }
            }
            x.eval()?;
        }
        Ok(x)
    }

    /// Raw decoder output `[1, frames, bins, 4]`: mask (re, im) and residual (re, im) per bin.
    fn decode(&self, tokens: &Array, frames: i32) -> R<Array> {
        let x = tokens.reshape(&[1, self.bands, frames, WIDTH])?;
        let x = &x * &self.dec_scale + &self.dec_shift;
        let x = fast::layer_norm(&x, None::<&Array>, None::<&Array>, LAYER_NORM_EPS)?;
        let x = &x * &self.dec_norm_weight + &self.dec_norm_bias;
        let x = gelu(&self.kernels, &self.dec_fc1.addmm(&self.dec_fc1_bias, &x)?)?;
        let x = self.dec_fc2.addmm(&self.dec_fc2_bias, &x)?.as_dtype(F32)?;
        let x = x.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, frames, -1])?;
        let x = take_axis(&x, &self.keep, 2)?;
        Ok(x.reshape(&[1, frames, BINS as i32, 4])?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bands_cover_every_bin_once_and_match_the_weights() {
        let edges = band_edges();
        assert_eq!(edges.len(), 46, "the checkpoint has 46 per-band layers");
        assert_eq!(edges[0].0, 0);
        assert_eq!(edges.last().unwrap().1, BINS);
        assert!(edges.windows(2).all(|p| p[0].1 == p[1].0));
        // The padded width the checkpoint's encoder was packed to is 2 * 85.
        assert_eq!(edges.iter().map(|(s, e)| e - s).max(), Some(85));
    }

    #[test]
    fn rope_tables_start_at_the_identity() {
        // Position 0 rotates by angle 0 for every frequency. Plain vectors: this test must not
        // need a GPU, which CI runners do not have.
        let (cos, sin) = rope_values(4);
        assert_eq!(cos.len(), 4 * 32);
        assert!(cos[..32].iter().all(|&c| c == 1.0));
        assert!(sin[..32].iter().all(|&s| s == 0.0));
        // The first frequency is 1 rad per frame: position 1 is (cos 1, sin 1).
        assert!((cos[32] - 1f32.cos()).abs() < 1e-6 && (sin[32] - 1f32.sin()).abs() < 1e-6);
    }
}
