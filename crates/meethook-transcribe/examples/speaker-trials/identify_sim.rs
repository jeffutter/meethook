//! What meethook itself would have decided.

use meethook_session::{EnrolledSpeaker, EnrolledSpeakers, RepresentativeSegment, SpeakerCluster};
use meethook_transcribe::{IDENTIFY_DISTANCE, identify_clusters};

use super::voices::Voice;

/// Runs the real identification decision over the same items, and separates its three
/// outcomes.
///
/// The rates above are the standard speaker-verification quantities and they are not what this
/// code does. [`identify_clusters`] is argmax over every enrolled reference and *then* the cut,
/// so a reference that clears the threshold while a nearer one wins is not a match, and a bare
/// trial-list false-accept rate would be a number about a decision rule meethook does not use.
///
/// Each speaker's **first session in manifest order** is their enrolled reference -- mirroring
/// `enroll`, which stores one session's cluster and replaces rather than averages -- and every
/// other session of theirs is a probe. The chosen reference is printed per speaker, because
/// "first" is only reproducible if it names an order somebody can re-read.
///
/// `threshold` is accepted but not applied: it exists so the caller can *say* whether the
/// simulation below is running at the same cut as the trial-list rates. Identification uses
/// [`IDENTIFY_DISTANCE`] internally and nothing here can override it, which is the point.
pub fn report_identification(voices: &[Voice], threshold: f32) {
    println!("\nwhat meethook would have decided (identify_clusters, argmax then threshold)");
    if threshold != IDENTIFY_DISTANCE {
        println!(
            "  note: --threshold {threshold:.3} moved the rates above; this block is the real \
             decision, which is fixed at IDENTIFY_DISTANCE {IDENTIFY_DISTANCE:.3}"
        );
    }

    // First-seen wins, and manifest order is preserved all the way from `read_manifest`.
    let mut references: Vec<&Voice> = Vec::new();
    let mut probes: Vec<&Voice> = Vec::new();
    for voice in voices {
        if references
            .iter()
            .any(|reference| reference.speaker == voice.speaker)
        {
            probes.push(voice);
        } else {
            references.push(voice);
        }
    }

    println!(
        "  enrolled {} speaker(s) from their first session, probing with {} other session(s)",
        references.len(),
        probes.len()
    );
    for reference in &references {
        println!("    {} <- {}", reference.speaker, reference.session);
    }
    if probes.is_empty() {
        println!(
            "  no probes: every speaker has exactly one session, so there is nothing to \
             identify. A closed-set simulation needs at least one speaker recorded twice."
        );
        return;
    }

    let enrolled = database(&references, None);
    let (mut correct, mut misattributed, mut rejected) = (0usize, 0usize, 0usize);
    for probe in &probes {
        match identify(probe, &enrolled) {
            Some(name) if name == probe.speaker => correct += 1,
            Some(name) => {
                misattributed += 1;
                println!(
                    "    MISATTRIBUTED: {} / {} named as {name}",
                    probe.speaker, probe.session
                );
            }
            None => rejected += 1,
        }
    }

    let of_probes = |count: usize| format!("{:.1}%", 100.0 * count as f32 / probes.len() as f32);
    println!("  closed set, {} probe(s):", probes.len());
    println!("    correct:        {correct} ({})", of_probes(correct));
    println!(
        "    misattributed:  {misattributed} ({}) -- one person's words under another \
         person's name",
        of_probes(misattributed)
    );
    println!(
        "    rejected:       {rejected} ({}) -- an enrolled speaker left as Unknown N",
        of_probes(rejected)
    );

    // The same sweep with the probe's own speaker taken out of the database: how often a voice
    // meethook has never enrolled is given somebody else's name anyway. One filtered database
    // and no new code, and it is the number that maps most directly onto user-visible harm.
    if references.len() < 2 {
        println!(
            "  open set: needs at least two enrolled speakers to be meaningful, this run has {}",
            references.len()
        );
        return;
    }
    let mut false_alarms = 0usize;
    for probe in &probes {
        let strangers = database(&references, Some(probe.speaker.as_str()));
        if let Some(name) = identify(probe, &strangers) {
            false_alarms += 1;
            println!(
                "    OPEN-SET FALSE ALARM: {} / {} named as {name} with {} not enrolled",
                probe.speaker, probe.session, probe.speaker
            );
        }
    }
    println!(
        "  open set, same {} probe(s) with their own speaker removed from the database:",
        probes.len()
    );
    println!(
        "    false alarms:   {false_alarms} ({}) -- an unenrolled voice given a name",
        of_probes(false_alarms)
    );
}

/// The enrolled database these references would have produced, optionally without one person.
fn database(references: &[&Voice], without: Option<&str>) -> EnrolledSpeakers {
    EnrolledSpeakers::new(
        references
            .iter()
            .filter(|reference| Some(reference.speaker.as_str()) != without)
            .map(|reference| EnrolledSpeaker {
                name: reference.speaker.clone(),
                embedding: reference.embedding.clone(),
                clip_seconds: None,
            })
            .collect(),
    )
}

/// The name the real decision would have put on this voice, if any.
fn identify(probe: &Voice, enrolled: &EnrolledSpeakers) -> Option<String> {
    let cluster = SpeakerCluster {
        id: 0,
        embedding: probe.embedding.clone(),
        speech_seconds: probe.speech_seconds,
        first_spoke_seconds: 0.0,
        // One cluster at a time, so there is nobody for it to be excluded from: this
        // instrument measures the reference distances, not the contested-name rule.
        heard_at_once_with: Vec::new(),
        representatives: vec![RepresentativeSegment {
            start: 0.0,
            end: probe.speech_seconds.min(2.0),
        }],
    };
    identify_clusters(std::slice::from_ref(&cluster), enrolled)
        .remove(&0)
        .map(|identification| identification.name)
}
