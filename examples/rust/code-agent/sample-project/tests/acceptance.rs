use code_agent_fixture::clamp;

#[test]
fn preserves_value_inside_bounds() {
    assert_eq!(clamp(5, 0, 10), 5);
}

#[test]
fn clamps_below_lower_bound() {
    assert_eq!(clamp(-3, 0, 10), 0);
}

#[test]
fn clamps_above_upper_bound() {
    assert_eq!(clamp(17, 0, 10), 10);
}

#[test]
fn uses_supplied_positive_bounds() {
    assert_eq!(clamp(17, 12, 30), 17);
    assert_eq!(clamp(0, 12, 30), 12);
    assert_eq!(clamp(40, 12, 30), 30);
}

#[test]
fn uses_supplied_negative_bounds() {
    assert_eq!(clamp(-7, -12, -3), -7);
    assert_eq!(clamp(-20, -12, -3), -12);
    assert_eq!(clamp(1, -12, -3), -3);
}
