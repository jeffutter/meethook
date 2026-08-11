use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{Error, Result, SessionId, SessionPaths, unknown_speaker, write_atomic};

/// Bumped whenever `speaker_clusters.json`'s shape changes incompatibly.
///
/// Separate from the transcript's and the session's versions for the same reason those two
/// are separate from each other: this file is written by diarization and read by `enroll`,
/// on its own schedule.
///
/// Version 2 added [`SpeakerCluster::first_spoke_seconds`], and added it as a required
/// field: a version 1 file now fails to parse rather than being read with a first
/// appearance of zero. Defaulting would be worse than refusing. Every cluster would tie at
/// zero, [`unknown_labels`] would fall through to its cluster-id tie-break, and the
/// "Unknown N" numbering a reader recovered would silently be talk-time order instead of
/// first-appearance order -- which in `enroll` means one person's name written onto another
/// person's turns. Re-transcribing the session rewrites the file correctly.
///
/// Version 3 added [`SpeakerCluster::heard_at_once_with`], required on the same terms. The
/// argument is not quite version 2's and is worth stating because of that: an empty list is
/// not obviously a fabricated value -- it is *today's behaviour*, and today's behaviour is
/// the defect. An empty list is a positive assertion that these voices were never heard
/// talking over each other, made about audio the reader never saw, and its consequence is
/// exactly the silent misattribution the naming rules elsewhere spend visible "Unknown N"
/// labels to avoid. So a version 2 file is refused, and re-transcribing rewrites it.
pub const SPEAKER_CLUSTERS_SCHEMA_VERSION: u32 = 3;

/// The shortest clip a representative segment is allowed to be.
///
/// `enroll` plays one of these and asks "who is this?". Under about a second and a half
/// there is not enough voice there for a person to answer -- a half-second fragment is a
/// syllable, and being asked to name a speaker from a syllable is a bad experience that
/// would be discovered long after this file was written.
///
/// Producers must satisfy this by widening a short segment into the surrounding track
/// rather than by dropping the cluster: a participant with no playable clip is a
/// participant who can never be named.
pub const MIN_REPRESENTATIVE_SECONDS: f64 = 1.5;

/// A stretch of `speaker.wav` to play back when asking who a cluster is.
///
/// `start` and `end` are seconds **into `speaker.wav`**, not seconds from session start.
/// This is the one place in the session contract that uses track time rather than the
/// shared timeline [`crate::Turn`] is on, and it is deliberate: the only thing anyone does
/// with these numbers is seek into that file and play, so making them offsets into it
/// removes a conversion -- and a chance to get it wrong -- from every consumer.
///
/// Guaranteed to span at least [`MIN_REPRESENTATIVE_SECONDS`], which may mean it includes
/// a little audio either side of the speech that earned it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RepresentativeSegment {
    pub start: f64,
    pub end: f64,
}

impl RepresentativeSegment {
    pub fn seconds(&self) -> f64 {
        self.end - self.start
    }
}

/// One voice on the speaker track, with no idea whose it is.
///
/// A cluster is what diarization can honestly produce on its own: everything here is
/// derived from the audio, and nothing from a person. Attaching a name is `enroll`'s job,
/// which is why there is no name field to leave empty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerCluster {
    /// Stable only within this session. Clusters are numbered by how much they spoke, most
    /// first, so `0` is the person who did most of the talking.
    pub id: u32,

    /// The voice fingerprint: the mean of this cluster's per-turn embeddings, L2-normalized
    /// *after* averaging.
    ///
    /// The order matters and is part of the contract, not an implementation detail. The
    /// mean of normalized vectors and the normalized mean are different vectors, and
    /// `speakers.json` stores enrolled speakers in this same representation so that
    /// comparing the two is a dot product. Produce it any other way and enrolment silently
    /// never matches.
    ///
    /// Length is whatever the embedding model emits (256 for the WeSpeaker checkpoint
    /// meethook ships against); this file does not pin it, so a future model change is a
    /// schema bump rather than a lie in a constant here.
    pub embedding: Vec<f32>,

    /// Total speech attributed to this cluster, in seconds. What the numbering is by, and
    /// how a consumer tells a participant from someone who coughed once.
    pub speech_seconds: f64,

    /// When this voice was first heard: seconds **into `speaker.wav`**, the same track time
    /// [`RepresentativeSegment`] uses rather than the session timeline [`crate::Turn`] is on.
    ///
    /// It is here because it is the one thing about a cluster that cannot be recovered from
    /// the rest of this file, and the whole "Unknown N" numbering rests on it: [`id`] ranks
    /// voices by talk time and `representatives` are the longest clips, so neither says who
    /// spoke first. A reader holding only this file reproduces the labels a transcript was
    /// written with by handing these to [`unknown_labels`].
    ///
    /// [`id`]: SpeakerCluster::id
    pub first_spoke_seconds: f64,

    /// Cluster ids this voice is *provably* a different person from, whatever the two
    /// embeddings look like.
    ///
    /// Segmentation supplies this for free: when the model is asked who spoke during one
    /// ten-second window and answers with two different local speakers, those two voices
    /// overlapped in time and cannot be one person. Clustering already refuses to merge
    /// across such a pair. This field is how the same fact reaches everyone downstream --
    /// notably `enroll`, which reads this file and never sees the audio, so a constraint
    /// that is not written here is a constraint it cannot honour.
    ///
    /// Ids are session-scoped, exactly as [`id`] is. The list is ascending, holds no
    /// duplicates, never contains the cluster's own id, and is symmetric: if `a` lists `b`
    /// then `b` lists `a`. Producers write every cluster's list in one pass, so the symmetry
    /// holds by construction rather than by validation -- and a consumer should still read
    /// *both* directions, so that a hand-edited file that broke the symmetry reads
    /// conservatively (one side asserting the exclusion is enough) rather than differently
    /// depending on which cluster it examined first.
    ///
    /// Empty for a voice segmentation never heard alongside another, which is the common
    /// case: an exclusion is positive evidence of overlap, not the absence of evidence of
    /// sameness. Two people who politely took turns all meeting exclude nobody.
    ///
    /// [`id`]: SpeakerCluster::id
    pub heard_at_once_with: Vec<u32>,

    /// Clips to play when asking who this is, longest first.
    ///
    /// Each spans at least [`MIN_REPRESENTATIVE_SECONDS`]. Never empty for a cluster that
    /// made it into this file.
    pub representatives: Vec<RepresentativeSegment>,
}

/// The "Unknown N" label each voice gets, keyed by cluster id.
///
/// `first_spoke` is when each voice was heard: `(cluster id, seconds into the speaker
/// track)`. A cluster may appear more than once -- handing over every diarized turn is as
/// valid as handing over one pair per cluster -- and the earliest time given for it wins.
///
/// Voices are numbered from 1 in order of first appearance, ties broken by ascending
/// cluster id. Ordering by first appearance is what makes "Unknown 1" mean "the first
/// unidentified person to speak"; cluster ids rank voices by how much they talked and mean
/// nothing to somebody reading a transcript from the top. The tie-break matters for two
/// people who started talking over each other, and keeps the labels independent of the
/// order the turns arrived in.
///
/// Numbers are handed out over *every* voice, including ones a caller is about to rename:
/// substituting a name leaves the number it took unused, so a meeting whose second speaker
/// is enrolled reads "Unknown 1 / Alice / Unknown 3". The gap is deliberate. Renumbering the
/// unnamed instead would mean enrolling one person silently relabels everybody else -- and
/// `enroll` rewrites existing transcripts in place, so that would land as a diff across
/// meetings nobody touched.
///
/// This rule lives here, in the on-disk contract, because two parties need the identical
/// numbering: `transcribe` renders these labels into a transcript, and `enroll` has to work
/// out which cluster an "Unknown 2" it is about to replace refers to. Two implementations
/// would drift, and the symptom would be a name written onto the wrong person's turns.
pub fn unknown_labels(first_spoke: impl IntoIterator<Item = (u32, f64)>) -> BTreeMap<u32, String> {
    let mut earliest: BTreeMap<u32, f64> = BTreeMap::new();
    for (cluster, seconds) in first_spoke {
        earliest
            .entry(cluster)
            .and_modify(|first| *first = first.min(seconds))
            .or_insert(seconds);
    }

    // Ascending cluster id out of the map, then a stable sort by time, which is the
    // tie-break stated above.
    let mut order: Vec<(u32, f64)> = earliest.into_iter().collect();
    order.sort_by(|a, b| a.1.total_cmp(&b.1));

    order
        .into_iter()
        .enumerate()
        .map(|(rank, (cluster, _))| (cluster, unknown_speaker(rank + 1)))
        .collect()
}

/// `speaker_clusters.json`: the voices found on one session's speaker track.
///
/// Written once by diarization and read by `enroll`, which is the whole point of it being
/// on disk at all: naming speakers must not require re-running the models over the audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerClusters {
    pub schema_version: u32,
    pub session_id: SessionId,
    pub clusters: Vec<SpeakerCluster>,
}

impl SpeakerClusters {
    pub fn new(session_id: SessionId, clusters: Vec<SpeakerCluster>) -> Self {
        SpeakerClusters {
            schema_version: SPEAKER_CLUSTERS_SCHEMA_VERSION,
            session_id,
            clusters,
        }
    }

    pub fn write(&self, paths: &SessionPaths) -> Result<()> {
        let path = paths.speaker_clusters_json();
        let mut json = serde_json::to_vec_pretty(self).map_err(|e| Error::json(&path, e))?;
        json.push(b'\n');
        write_atomic(&path, &json)
    }

    pub fn read(path: &Path) -> Result<SpeakerClusters> {
        let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::json(path, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two voices whose talk-time order is the opposite of their first-appearance order --
    /// cluster 1 spoke first and cluster 0 did most of the talking -- so nothing below can
    /// pass by confusing the two. They also talked over each other, so the exclusion is
    /// non-empty on both sides and a round trip that dropped it would show.
    fn clusters() -> SpeakerClusters {
        SpeakerClusters::new(
            SessionId::parse("20260809-052600").unwrap(),
            vec![
                SpeakerCluster {
                    id: 0,
                    embedding: vec![0.6, 0.8],
                    speech_seconds: 42.5,
                    first_spoke_seconds: 31.75,
                    heard_at_once_with: vec![1],
                    representatives: vec![RepresentativeSegment {
                        start: 10.0,
                        end: 13.0,
                    }],
                },
                SpeakerCluster {
                    id: 1,
                    embedding: vec![0.8, 0.6],
                    speech_seconds: 8.0,
                    first_spoke_seconds: 2.5,
                    heard_at_once_with: vec![0],
                    representatives: vec![RepresentativeSegment {
                        start: 2.5,
                        end: 5.5,
                    }],
                },
            ],
        )
    }

    #[test]
    fn a_written_file_reads_back_identical() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path());
        let written = clusters();

        written.write(&paths).unwrap();

        let read = SpeakerClusters::read(&paths.speaker_clusters_json()).unwrap();
        assert_eq!(read, written);
        assert_eq!(read.schema_version, SPEAKER_CLUSTERS_SCHEMA_VERSION);
    }

    /// The atomic write leaves no temp file behind, and lands on the one name the rest of
    /// the tool looks for.
    #[test]
    fn writing_leaves_exactly_one_file_in_the_session_directory() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path());

        clusters().write(&paths).unwrap();

        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(
            entries,
            [paths.speaker_clusters_json().file_name().unwrap()]
        );
    }

    /// Rewriting is what a `--force` re-transcribe does, and it must not append or merge.
    #[test]
    fn rewriting_replaces_the_previous_contents() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path());

        clusters().write(&paths).unwrap();
        let mut second = clusters();
        second.clusters.clear();
        second.write(&paths).unwrap();

        let read = SpeakerClusters::read(&paths.speaker_clusters_json()).unwrap();
        assert!(read.clusters.is_empty());
    }

    /// The field `enroll` cannot do its job without, stated on its own rather than left to
    /// the struct-wide equality above: it has to come back off disk as the number that went
    /// on, not as a zero that happens to compare equal to another zero.
    #[test]
    fn first_appearance_survives_a_write_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path());

        clusters().write(&paths).unwrap();

        let read = SpeakerClusters::read(&paths.speaker_clusters_json()).unwrap();
        assert_eq!(
            read.clusters
                .iter()
                .map(|c| (c.id, c.first_spoke_seconds))
                .collect::<Vec<_>>(),
            [(0, 31.75), (1, 2.5)]
        );
    }

    /// The other field nothing else in the file can reconstruct: `enroll` never sees the
    /// audio, so an exclusion that did not survive the write is an exclusion it will never
    /// know about. Stated on its own rather than left to the struct-wide equality above.
    #[test]
    fn the_heard_at_once_relation_survives_a_write_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path());

        clusters().write(&paths).unwrap();

        let read = SpeakerClusters::read(&paths.speaker_clusters_json()).unwrap();
        assert_eq!(
            read.clusters
                .iter()
                .map(|c| (c.id, c.heard_at_once_with.clone()))
                .collect::<Vec<_>>(),
            [(0, vec![1]), (1, vec![0])]
        );
    }

    /// A version 2 file predates the relation, and an empty list is not a neutral stand-in
    /// for it: it asserts that no two of these voices ever overlapped, which is a claim about
    /// audio this file no longer describes, and acting on it is how two people end up filed
    /// under one name. Refuse, and name the file, since that is all `enroll` has to go on when
    /// it tells the user to re-transcribe.
    #[test]
    fn a_version_2_file_is_refused_rather_than_read_with_no_exclusions() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("speaker_clusters.json");
        std::fs::write(
            &path,
            br#"{
              "schema_version": 2,
              "session_id": "20260809-052600",
              "clusters": [
                {
                  "id": 0,
                  "embedding": [0.6, 0.8],
                  "speech_seconds": 42.5,
                  "first_spoke_seconds": 31.75,
                  "representatives": [{ "start": 10.0, "end": 13.0 }]
                }
              ]
            }"#,
        )
        .unwrap();

        let error = SpeakerClusters::read(&path).unwrap_err();

        assert!(
            matches!(&error, Error::Json { path: at, .. } if at == &path),
            "{error:?}"
        );
    }

    /// A `speaker_clusters.json` written by a version 1 build has no first appearance in it,
    /// and there is no honest value to invent: every cluster would tie at zero and the
    /// numbering below would quietly become talk-time order. So it must fail to parse, and
    /// say which file it was, since that is all `enroll` has to tell the user to re-transcribe.
    #[test]
    fn a_version_1_file_is_refused_rather_than_read_with_a_defaulted_first_appearance() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("speaker_clusters.json");
        std::fs::write(
            &path,
            br#"{
              "schema_version": 1,
              "session_id": "20260809-052600",
              "clusters": [
                {
                  "id": 0,
                  "embedding": [0.6, 0.8],
                  "speech_seconds": 42.5,
                  "representatives": [{ "start": 10.0, "end": 13.0 }]
                }
              ]
            }"#,
        )
        .unwrap();

        let error = SpeakerClusters::read(&path).unwrap_err();

        assert!(
            matches!(&error, Error::Json { path: at, .. } if at == &path),
            "{error:?}"
        );
    }

    /// The rule itself: order of first appearance, not the order of the map and not cluster
    /// id. Cluster ids rank voices by talk time, so the first person to speak is routinely
    /// not cluster 0 -- and a transcript whose opening line is "Unknown 3" reads as a bug.
    #[test]
    fn voices_are_numbered_by_first_appearance_rather_than_by_cluster_id() {
        let labels = unknown_labels([(0, 30.0), (1, 10.0), (2, 20.0)]);

        assert_eq!(labels[&1], "Unknown 1");
        assert_eq!(labels[&2], "Unknown 2");
        assert_eq!(labels[&0], "Unknown 3");
    }

    /// Two people who started talking over each other still have to be numbered
    /// reproducibly, whichever order the caller happens to hand them over in.
    #[test]
    fn voices_that_first_speak_at_the_same_instant_are_numbered_by_cluster_id() {
        for order in [[(1, 4.0), (0, 4.0)], [(0, 4.0), (1, 4.0)]] {
            let labels = unknown_labels(order);
            assert_eq!(labels[&0], "Unknown 1");
            assert_eq!(labels[&1], "Unknown 2");
        }
    }

    /// Callers may hand over every turn rather than one pair per voice, which is what
    /// `merge` has in front of it; a voice's *first* appearance is what counts, however many
    /// later ones come with it.
    #[test]
    fn a_voice_seen_more_than_once_is_ranked_by_its_earliest_appearance() {
        let labels = unknown_labels([(0, 9.0), (1, 5.0), (0, 1.0), (1, 12.0)]);

        assert_eq!(labels[&0], "Unknown 1");
        assert_eq!(labels[&1], "Unknown 2");
    }

    /// Diarization finding nobody -- an unusually quiet or noisy track -- is an ordinary
    /// meeting, not an error.
    #[test]
    fn no_voices_yields_no_labels() {
        assert!(unknown_labels([]).is_empty());
    }
}
