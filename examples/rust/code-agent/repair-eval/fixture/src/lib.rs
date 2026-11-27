//! Normalize unordered, nonempty half-open integer ranges into their sorted union.
pub fn merge_ranges(mut ranges: Vec<(i32, i32)>) -> Vec<(i32, i32)> {
    ranges.sort_unstable();
    let mut merged: Vec<(i32, i32)> = Vec::new();
    for (start, end) in ranges {
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = end;
            continue;
        }
        merged.push((start, end));
    }
    merged
}
