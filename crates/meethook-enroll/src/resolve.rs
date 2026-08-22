//! Deciding what a typed name means: an enrolled person, a shortlist, or somebody new.
//!
//! Everywhere else in this tool a name matches exactly and case-sensitively --
//! [`EnrolledSpeakers::store_reference`](meethook_session::EnrolledSpeakers::store_reference)
//! says so in its own doc -- so `marco`, `Marco ` and `Marclo` are three
//! people. The failure is silent: a fresh row appears in `speakers.json` and the person the user
//! meant gains nothing. [`resolve`] is the one place that is allowed to decide two spellings mean
//! one person.
//!
//! # Three outcomes, and the resolver decides rather than chooses
//!
//! Typed text is [`Resolution::Blank`], [`Resolution::Enrolled`], [`Resolution::Candidates`] or
//! [`Resolution::New`], and which of the four it is decides what lands on disk -- which is why
//! this lives here rather than in the terminal, where it could not be reached from `cargo test`.
//! Presenting candidates and taking the confirmation is the interface layer's job. This module
//! never picks between two enrolled people, and an ambiguous prefix never resolves itself
//! silently.
//!
//! Against this install the collisions are real rather than hypothetical: `Ivan` and `Owen`,
//! `Marco` and `Marcel`, `Nate` and `Nina`, `Jane` and `Jon`. Typing `N` or
//! `Mar` is genuinely ambiguous, and any rule that guesses is picking which of two real people
//! gets a recording of the other one.
//!
//! # The fold
//!
//! Two spellings are compared after folding: **trim, lowercase, and collapse internal whitespace
//! runs to one space**. So "exact" throughout this module means exact *after folding*, and
//! `  marco ` is the same answer as `Marco` -- a double-tapped space is the
//! same slip as a trailing one. Lowercasing is `str::to_lowercase` rather than the ASCII-only
//! form, because a name in `speakers.json` may not be ASCII.
//!
//! # Only an exact fold resolves; a lone inexact match is still a candidate
//!
//! `Owe` against an install holding `Owen` returns one candidate, not
//! `Enrolled("Owen")`. "The only thing it could be" is a claim this module is not entitled
//! to make: `Ivan` and `Owen` are two real people here, so the same reasoning that forbids
//! choosing between two candidates forbids promoting one. It is also what makes the resolver safe
//! behind a headless driver, which can treat [`Resolution::Candidates`] as "must ask" and cannot
//! be handed a silent rewrite.
//!
//! For the same reason an *ambiguous* fold does not resolve either. `speakers.json` stores names
//! case-sensitively, so `alice` and `Alice` can both be enrolled today; typing either folds onto
//! two names, and that is two candidates rather than a coin toss.
//!
//! # Non-goals
//!
//! **Voice similarity is deliberately not mixed into the ranking.** How much a voice *sounds*
//! like each candidate is what [`crate::Voice::resembles`] already carries, and combining a
//! textual order with an acoustic one is an interface policy -- a caller holding both lists can
//! re-sort. Nothing here takes an embedding.
//!
//! This is also cheap on purpose: no allocation per comparison beyond the folded name, and an
//! early exit as soon as a length difference exceeds the edit budget. It is meant to be called on
//! every keystroke, which is the opposite of the enrolment preview a caller may show beside it --
//! that one costs a database clone and two full labellings, and must be computed for one
//! highlighted candidate only. The two are neighbours with opposite cost profiles; do not treat
//! them alike.

/// What typed text turned out to mean.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    /// Not a name at all: empty, or whitespace only.
    ///
    /// The same treatment `run_speakers` already gives an answer that trims to nothing --
    /// somebody pressing Enter with a stray keystroke in the buffer, not a request for a person
    /// called "".
    Blank,
    /// Exactly one enrolled name, and the text can mean nothing else.
    ///
    /// Carries the name **as `speakers.json` spells it**, not as it was typed, so that storing a
    /// reference against it lands on the person already there rather than beside them.
    Enrolled(String),
    /// The enrolled names the text plausibly means, ranked, none of them chosen.
    ///
    /// `typed` is the text with surrounding whitespace removed and the user's own case and
    /// internal spelling intact -- what the new person would be called if the caller offers
    /// "none of these" and the user takes it. `matches` is never empty.
    Candidates { typed: String, matches: Vec<Match> },
    /// Nobody enrolled is plausible: this is a new person, under this spelling.
    ///
    /// Trimmed, keeping the user's case and internal spelling, which is the normalisation the
    /// `--name` path already applies -- so both entry paths agree about what a new person is
    /// called.
    New(String),
}

/// One enrolled name the typed text might have meant, and how strong the claim is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Match {
    /// The name as `speakers.json` spells it.
    pub name: String,
    /// Why this name is a candidate.
    pub likeness: Likeness,
}

/// Why a name is a candidate -- and, by declaration order, how strong the claim is.
///
/// `Ord` is the ranking: [`Likeness::Same`] outranks [`Likeness::Prefix`] outranks
/// [`Likeness::WordPrefix`] outranks [`Likeness::NearMiss`]. A name is reported at the strongest
/// tier it qualifies under, once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Likeness {
    /// The folded name equals the folded text.
    ///
    /// One of these is [`Resolution::Enrolled`] rather than a candidate, so this variant only
    /// ever surfaces when two enrolled names fold together -- `alice` and `Alice`. On a sane
    /// database it never appears.
    Same,
    /// The folded name starts with the folded text: `N` -> `Nate`, `Nina`.
    Prefix,
    /// Some word of the folded name starts with the folded text: `Brack` -> `Owen Brack`.
    ///
    /// This tier carries most of the practical value, because the names in `speakers.json` are
    /// `First Last` and the half a user remembers is often the second one.
    WordPrefix,
    /// The folded text is within an edit budget of the whole folded name, or of one of its
    /// words: `Marclo` -> `Marco`, `Vancle` -> `Elliot Vance`.
    NearMiss,
}

/// What `typed` means against `enrolled`, which is every enrolled name.
///
/// The universe must be
/// [`EnrolledSpeakers::enrolled_names`](meethook_session::EnrolledSpeakers::enrolled_names) and
/// **not** the names a voice resembles. Ranking a voice against the database drops a person whose
/// every stored recording is a stale embedding dimension, and that person is still real: a typo
/// must not duplicate them. The `resembles` list is the more convenient call from an interviewer
/// that already holds one, which is exactly why this says so.
///
/// `enrolled` is a set of names rather than an `EnrolledSpeakers` so the decision is testable
/// against literals; duplicates are collapsed inside, so a caller who passes the raw rows gets
/// one entry per person rather than one per recording.
///
/// See the module doc for the fold, for why a lone inexact match is still a candidate, and for
/// the ranking.
#[must_use]
pub fn resolve(typed: &str, enrolled: &[&str]) -> Resolution {
    let folded = fold(typed);
    if folded.is_empty() {
        return Resolution::Blank;
    }
    let typed_chars: Vec<char> = folded.chars().collect();
    let budget = near_miss_budget(typed_chars.len());

    let mut seen: Vec<&str> = Vec::new();
    let mut matches: Vec<Match> = Vec::new();
    for name in enrolled {
        if seen.contains(name) {
            continue;
        }
        seen.push(name);
        if let Some(likeness) = likeness(&folded, &typed_chars, budget, name) {
            matches.push(Match {
                name: (*name).to_string(),
                likeness,
            });
        }
    }

    // Exactly one name folds onto the text: the user spelt somebody's name, and offering it back
    // as a question would be the resolver refusing a correct answer. Two is the `alice`/`Alice`
    // ambiguity the module doc describes, and falls through to `Candidates`.
    let mut folds_together = matches.iter().filter(|m| m.likeness == Likeness::Same);
    if let (Some(only), None) = (folds_together.next(), folds_together.next()) {
        return Resolution::Enrolled(only.name.clone());
    }

    if matches.is_empty() {
        return Resolution::New(typed.trim().to_string());
    }
    matches.sort_by(|a, b| rank(a).cmp(&rank(b)));
    Resolution::Candidates {
        typed: typed.trim().to_string(),
        matches,
    }
}

/// The candidate order, as a key: strongest [`Likeness`] first, then the shortest name, then the
/// name itself.
///
/// Total -- names are deduplicated before this is applied, so no two keys are equal -- and so
/// independent of the order the enrolled names arrived in. Length is counted in `char`s, and the
/// tie-break is the one worth justifying: among prefix matches the shorter the name, the larger
/// the fraction of it the user actually typed, so it is the entry the resolver is guessing least
/// about. It also orders both real collisions the way a person would, `N` -> `Nate`, `Nina` and
/// `Mar` -> `Marco`, `Marcel`.
fn rank(candidate: &Match) -> (Likeness, usize, &str) {
    (
        candidate.likeness,
        candidate.name.chars().count(),
        &candidate.name,
    )
}

/// The strongest tier `name` qualifies under against already-folded typed text, or `None` if the
/// text cannot plausibly mean it.
///
/// `typed_chars` is `folded_typed`'s `char`s, hoisted because the caller compares one typed
/// string against every name. `budget` is [`near_miss_budget`] of its length.
fn likeness(
    folded_typed: &str,
    typed_chars: &[char],
    budget: Option<usize>,
    name: &str,
) -> Option<Likeness> {
    let folded_name = fold(name);
    if folded_name == folded_typed {
        return Some(Likeness::Same);
    }
    if folded_name.starts_with(folded_typed) {
        return Some(Likeness::Prefix);
    }
    // The fold collapsed every whitespace run to one space, so a single-space split is the word
    // split.
    if folded_name
        .split(' ')
        .any(|word| word.starts_with(folded_typed))
    {
        return Some(Likeness::WordPrefix);
    }
    let budget = budget?;
    let whole: Vec<char> = folded_name.chars().collect();
    if edit_distance(typed_chars, &whole, budget).is_some() {
        return Some(Likeness::NearMiss);
    }
    // The per-word arm is what catches a misspelt surname typed on its own -- `Vancle` ->
    // `Elliot Vance` -- which neither the whole-string arm nor any prefix tier reaches.
    for word in folded_name.split(' ') {
        let word: Vec<char> = word.chars().collect();
        if edit_distance(typed_chars, &word, budget).is_some() {
            return Some(Likeness::NearMiss);
        }
    }
    None
}

/// How many edits [`Likeness::NearMiss`] tolerates for folded text of `length` `char`s, or `None`
/// where the tier does not apply at all.
///
/// | folded length | budget |
/// |---|---|
/// | 0-3 | tier does not apply |
/// | 4-7 | 1 |
/// | 8+ | 2 |
///
/// The gate at the bottom is the load-bearing half. At three characters or fewer, a budget of
/// even one edit makes most of the database a candidate for most of the database -- `Nate` and `Nina`
/// are two edits apart, `Jane` and `Jon` are two -- while the prefix tiers already cover everything
/// a user typing two letters could have meant. Above it the budget buys the cases this module
/// exists for: `Marclo` is one edit from `marco`, and `Janet` one from `Jane`.
const fn near_miss_budget(length: usize) -> Option<usize> {
    match length {
        0..=3 => None,
        4..=7 => Some(1),
        _ => Some(2),
    }
}

/// `text` trimmed, lowercased, and with internal whitespace runs collapsed to one space.
///
/// The one normalisation in this module: everything it calls "exact" is exact after this. See the
/// module doc for why collapsing internal runs is part of it.
fn fold(text: &str) -> String {
    let mut folded = String::with_capacity(text.len());
    for word in text.split_whitespace() {
        if !folded.is_empty() {
            folded.push(' ');
        }
        folded.extend(word.chars().flat_map(char::to_lowercase));
    }
    folded
}

/// The optimal string alignment distance between `a` and `b`, or `None` once it is certain to
/// exceed `budget`.
///
/// Damerau-Levenshtein restricted to adjacent transpositions: swapping two neighbouring
/// characters is one keystroke and so costs one edit, which is most of what a mistyped name is.
///
/// Over `char`s rather than bytes, so a non-ASCII name cannot split a code point. Two rolling
/// rows plus the one before them -- the transposition case needs it -- rather than a full matrix,
/// and it returns early both on a length difference the budget cannot close and on a whole row
/// already past the budget, which is what keeps a caller free to run this on every keystroke.
fn edit_distance(a: &[char], b: &[char], budget: usize) -> Option<usize> {
    if a.len().abs_diff(b.len()) > budget {
        return None;
    }
    let mut before_previous: Vec<usize> = Vec::new();
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut row: Vec<usize> = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        row[0] = i;
        let mut best = row[0];
        for j in 1..=b.len() {
            let substitution = usize::from(a[i - 1] != b[j - 1]);
            let mut cost = (previous[j] + 1)
                .min(row[j - 1] + 1)
                .min(previous[j - 1] + substitution);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                cost = cost.min(before_previous[j - 2] + 1);
            }
            row[j] = cost;
            best = best.min(cost);
        }
        if best > budget {
            return None;
        }
        std::mem::swap(&mut before_previous, &mut previous);
        std::mem::swap(&mut previous, &mut row);
        // `row` now holds whichever buffer fell out of the window, which the next iteration
        // overwrites cell by cell -- except on the first pass, where it fell out empty.
        row.resize(b.len() + 1, 0);
    }
    let distance = previous[b.len()];
    (distance <= budget).then_some(distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This install's real shape, so that every test below is a statement about a hazard that
    /// exists rather than one invented for the test: `Marco` and `Marcel` share a prefix,
    /// `Nate` and `Nina` another, `Jane` and `Jon` a third.
    const ENROLLED: &[&str] = &[
        "Ivan", "Owen", "Marco", "Marcel", "Nate", "Nina", "Jane", "Jon",
    ];

    fn candidates(typed: &str, enrolled: &[&str]) -> Vec<(String, Likeness)> {
        match resolve(typed, enrolled) {
            Resolution::Candidates { matches, .. } => {
                matches.into_iter().map(|m| (m.name, m.likeness)).collect()
            }
            other => panic!("expected candidates for {typed:?}, got {other:?}"),
        }
    }

    fn names(typed: &str, enrolled: &[&str]) -> Vec<String> {
        candidates(typed, enrolled)
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    }

    /// The single most important case here: a correct answer must not be turned into a question
    /// just because the database holds other people. `Ivan` is a person, and `Owen` is
    /// somebody else.
    #[test]
    fn an_exact_fold_resolves_and_is_not_offered_the_longer_name() {
        assert_eq!(
            resolve("Ivan", ENROLLED),
            Resolution::Enrolled("Ivan".to_string())
        );
    }

    #[test]
    fn case_and_surrounding_whitespace_resolve_to_the_enrolled_spelling() {
        for typed in ["Ivan", "ivan", "IVAN", "  Ivan  ", "\tIvan\n"] {
            assert_eq!(
                resolve(typed, ENROLLED),
                Resolution::Enrolled("Ivan".to_string()),
                "typed {typed:?}"
            );
        }
    }

    /// The fold collapses internal runs too, so a double-tapped space is the same slip as a
    /// trailing one, and neither enrols a second Marco.
    #[test]
    fn a_lowercased_or_doubly_spaced_full_name_resolves_to_the_enrolled_one() {
        for typed in ["marco", "Marco", "  MARCO "] {
            assert_eq!(
                resolve(typed, ENROLLED),
                Resolution::Enrolled("Marco".to_string()),
                "typed {typed:?}"
            );
        }
    }

    /// Two real people share the prefix, so the answer is both of them and the variant itself is
    /// the criterion: the resolver did not choose.
    #[test]
    fn an_ambiguous_prefix_returns_both_people_without_choosing() {
        assert_eq!(names("N", ENROLLED), ["Nate", "Nina"]);
        assert_eq!(names("Mar", ENROLLED), ["Marco", "Marcel"]);
    }

    #[test]
    fn a_prefix_carries_the_typed_text_for_the_none_of_these_case() {
        assert_eq!(
            resolve("  N ", ENROLLED),
            Resolution::Candidates {
                typed: "N".to_string(),
                matches: vec![
                    Match {
                        name: "Nate".to_string(),
                        likeness: Likeness::Prefix,
                    },
                    Match {
                        name: "Nina".to_string(),
                        likeness: Likeness::Prefix,
                    },
                ],
            }
        );
    }

    /// The lone-match rule: one plausible person is still a question, because "the only thing it
    /// could be" is a claim the resolver is not entitled to make.
    #[test]
    fn a_lone_inexact_match_is_a_candidate_rather_than_a_resolution() {
        assert_eq!(
            candidates("Marclo", ENROLLED),
            [("Marco".to_string(), Likeness::NearMiss)]
        );
        assert_eq!(
            candidates("Owe", ENROLLED),
            [("Owen".to_string(), Likeness::Prefix)]
        );
    }

    /// A misspelt surname on its own, which no prefix tier and no whole-string comparison
    /// reaches.
    #[test]
    fn a_misspelt_word_near_misses_the_name_holding_it() {
        assert_eq!(
            candidates("Vancle", &["Elliot Vance"]),
            [("Elliot Vance".to_string(), Likeness::NearMiss)]
        );
    }

    #[test]
    fn a_surname_typed_alone_finds_the_person_by_word_prefix() {
        assert_eq!(
            candidates("Brack", &["Owen Brack", "Braxton"]),
            [("Owen Brack".to_string(), Likeness::WordPrefix)]
        );
        assert_eq!(
            candidates("Fenn", &["Quill Fenn"]),
            [("Quill Fenn".to_string(), Likeness::WordPrefix)]
        );
    }

    #[test]
    fn a_trailing_keystroke_near_misses_the_shorter_name() {
        assert_eq!(
            candidates("Janet", ENROLLED),
            [("Jane".to_string(), Likeness::NearMiss)]
        );
    }

    /// Pins the length tie-break rather than leaving it incidental: both are one-character
    /// prefixes, and the shorter name is the one the resolver is guessing least about.
    #[test]
    fn equally_strong_candidates_are_ordered_shortest_name_first() {
        assert_eq!(names("J", ENROLLED), ["Jon", "Jane"]);
    }

    /// The length gate on the near-miss tier. `Ja` is two edits from `Jon`, and offering it would
    /// make most of this database a candidate for most of it; the prefix tiers already cover what
    /// two letters can mean.
    #[test]
    fn the_near_miss_tier_does_not_fire_under_four_characters() {
        assert_eq!(names("Ja", ENROLLED), ["Jane"]);
        assert_eq!(resolve("Nik", ENROLLED), Resolution::New("Nik".to_string()));
    }

    #[test]
    fn text_matching_nobody_is_a_new_person_under_the_typed_spelling() {
        assert_eq!(resolve("Zoe", ENROLLED), Resolution::New("Zoe".to_string()));
        assert_eq!(
            resolve("  zoe  ", ENROLLED),
            Resolution::New("zoe".to_string())
        );
    }

    #[test]
    fn empty_or_whitespace_only_text_is_not_a_name() {
        for typed in ["", "   ", "\t\n", " \u{a0}"] {
            assert_eq!(
                resolve(typed, ENROLLED),
                Resolution::Blank,
                "typed {typed:?}"
            );
            assert_eq!(resolve(typed, &[]), Resolution::Blank, "typed {typed:?}");
        }
    }

    /// An empty candidate list must be unconstructible in practice, and an empty database is
    /// where it would come from.
    #[test]
    fn nobody_enrolled_makes_every_name_new() {
        assert_eq!(resolve("Zoe", &[]), Resolution::New("Zoe".to_string()));
    }

    /// `speakers.json` matches names case-sensitively, so both of these can be enrolled today.
    /// Typing either folds onto two people, and that is a question rather than a coin toss.
    #[test]
    fn text_folding_onto_two_enrolled_names_is_not_resolved_to_either() {
        assert_eq!(
            candidates("ALICE", &["alice", "Alice"]),
            [
                ("Alice".to_string(), Likeness::Same),
                ("alice".to_string(), Likeness::Same),
            ]
        );
    }

    /// The order is a property of the names, not of the slice they arrived in -- otherwise the
    /// highlighted candidate could move between runs with nothing having changed.
    #[test]
    fn the_candidate_order_is_independent_of_the_enrolled_order() {
        for typed in ["N", "Mar", "J"] {
            let expected = names(typed, ENROLLED);
            let mut permuted: Vec<&str> = ENROLLED.to_vec();
            for _ in 0..ENROLLED.len() {
                permuted.rotate_left(1);
                assert_eq!(
                    names(typed, &permuted),
                    expected,
                    "{typed:?} in {permuted:?}"
                );
            }
            permuted.reverse();
            assert_eq!(names(typed, &permuted), expected, "{typed:?} reversed");
        }
    }

    /// A caller who hands over the raw rows rather than the deduplicated names is counting
    /// recordings; one person must still be one entry, and must still resolve rather than look
    /// ambiguous.
    #[test]
    fn a_repeated_enrolled_name_is_one_person() {
        assert_eq!(
            resolve("Ivan", &["Ivan", "Ivan"]),
            Resolution::Enrolled("Ivan".to_string())
        );
        assert_eq!(names("I", &["Ivan", "Ivan"]), ["Ivan"]);
    }

    /// The char-versus-byte regression: folding and the edit distance both walk this name, and
    /// slicing it by byte would panic mid code point.
    #[test]
    fn a_non_ascii_name_folds_and_near_misses_without_panicking() {
        let enrolled = &["Zoë Müller", "Ünal"];
        assert_eq!(
            resolve("zoë  müller", enrolled),
            Resolution::Enrolled("Zoë Müller".to_string())
        );
        assert_eq!(
            candidates("Zoe Müller", enrolled),
            [("Zoë Müller".to_string(), Likeness::NearMiss)]
        );
        assert_eq!(
            resolve("ünal", enrolled),
            Resolution::Enrolled("Ünal".to_string())
        );
    }

    #[test]
    fn a_transposition_costs_one_edit() {
        assert_eq!(
            edit_distance(&chars("elephant"), &chars("elephatn"), 1),
            Some(1)
        );
        assert_eq!(
            edit_distance(&chars("elephant"), &chars("elephant"), 0),
            Some(0)
        );
        assert_eq!(edit_distance(&chars("elephant"), &chars("elep"), 2), None);
        assert_eq!(edit_distance(&chars(""), &chars("iv"), 2), Some(2));
    }

    fn chars(text: &str) -> Vec<char> {
        text.chars().collect()
    }
}
