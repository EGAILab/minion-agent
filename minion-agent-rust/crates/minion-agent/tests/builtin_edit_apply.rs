use minion_agent::tools::builtin::{Edit, apply_edits};

#[test]
fn replacements_are_matched_against_the_original_not_incrementally() {
    assert_eq!(
        apply_edits(
            "abc def",
            &[Edit::new("def", "D"), Edit::new("abc", "A")],
            "f"
        )
        .unwrap(),
        "A D"
    );
    assert!(
        apply_edits(
            "abc",
            &[Edit::new("abc", "def"), Edit::new("def", "X")],
            "f"
        )
        .unwrap_err()
        .message()
        .as_str()
        .unwrap()
        .contains("Could not find edits[1]")
    );
}

#[test]
fn exact_match_is_still_counted_in_fuzzy_space() {
    assert_eq!(
        apply_edits("a’ a'", &[Edit::new("a'", "b")], "f")
            .unwrap_err()
            .message(),
        "Found 2 occurrences of the text in f. The text must be unique. Please provide more context to make it unique."
    );
}

#[test]
fn fuzzy_changes_preserve_untouched_original_lines() {
    assert_eq!(
        apply_edits("keep’  \na’  \nlast’  \n", &[Edit::new("a'", "A")], "f").unwrap(),
        "keep’  \nA\nlast’  \n"
    );
    assert_eq!(
        apply_edits("a’\n   ", &[Edit::new("a'", "A")], "f")
            .unwrap_err()
            .message(),
        "Cannot preserve unchanged lines because the base content has a different line count."
    );
}

#[test]
fn overlapping_edits_and_empty_normalized_match_keep_pi_diagnostics() {
    assert_eq!(
        apply_edits("abc", &[Edit::new("ab", "A"), Edit::new("bc", "B")], "f")
            .unwrap_err()
            .message(),
        "edits[0] and edits[1] overlap in f. Merge them into one edit or target disjoint regions."
    );
    assert_eq!(
        apply_edits("", &[Edit::new(" ", "A")], "f")
            .unwrap_err()
            .message(),
        "Replacement range is outside the base content."
    );
}
