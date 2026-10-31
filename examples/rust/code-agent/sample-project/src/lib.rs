/// Returns a value restricted to the inclusive lower and upper bounds.
pub fn clamp(value: i32, lower: i32, upper: i32) -> i32 {
    value.max(lower).max(upper)
}
