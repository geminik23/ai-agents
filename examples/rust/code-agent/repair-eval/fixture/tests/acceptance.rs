use range_repair::merge_ranges;

#[test]
fn empty_and_single() {
    assert!(merge_ranges(vec![]).is_empty());
    assert_eq!(merge_ranges(vec![(3, 7)]), [(3, 7)]);
}

#[test]
fn contained_then_overlapping() {
    assert_eq!(merge_ranges(vec![(8, 12), (2, 4), (1, 10)]), [(1, 12)]);
}

#[test]
fn adjacent_and_separate() {
    assert_eq!(
        merge_ranges(vec![(8, 9), (0, 2), (-4, 0)]),
        [(-4, 2), (8, 9)]
    );
}

#[test]
fn permutation_preserves_union() {
    let cases = [(-8, 5), (0, 1), (4, 7), (12, 16)];
    for offset in 0..cases.len() {
        let mut ranges = cases;
        ranges.rotate_left(offset);
        assert_eq!(merge_ranges(ranges.to_vec()), [(-8, 7), (12, 16)]);
    }
}
