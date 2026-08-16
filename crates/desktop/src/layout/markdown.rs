pub(crate) fn list_marker_width(ordered: bool, start: u32, item_count: usize) -> f32 {
    if !ordered {
        return 18.0;
    }
    let last = start.saturating_add(item_count.saturating_sub(1).min(u32::MAX as usize) as u32);
    let digits = last.max(1).ilog10() + 1;
    (digits as f32 * 9.0 + 15.0).max(28.0)
}

#[cfg(test)]
mod tests {
    use super::list_marker_width;

    #[test]
    fn ordered_marker_column_grows_for_three_digit_items() {
        assert_eq!(list_marker_width(true, 1, 9), 28.0);
        assert_eq!(list_marker_width(true, 1, 400), 42.0);
        assert_eq!(list_marker_width(false, 1, 400), 18.0);
    }
}
