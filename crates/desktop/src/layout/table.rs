#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ColumnSpec {
    pub(crate) min: f32,
    pub(crate) preferred: f32,
    pub(crate) max: Option<f32>,
    pub(crate) weight: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TableLayout {
    pub(crate) tracks: Vec<f32>,
    pub(crate) content_width: f32,
    pub(crate) horizontal_scroll: bool,
}

pub(crate) fn allocate_columns(available_width: f32, specs: &[ColumnSpec]) -> TableLayout {
    if specs.is_empty() {
        return TableLayout {
            tracks: Vec::new(),
            content_width: 0.0,
            horizontal_scroll: false,
        };
    }
    let available = finite_non_negative(available_width);
    let normalized = specs
        .iter()
        .map(|spec| {
            let min = finite_non_negative(spec.min);
            let max = spec.max.map(finite_non_negative).map(|max| max.max(min));
            let preferred = finite_non_negative(spec.preferred)
                .max(min)
                .min(max.unwrap_or(f32::MAX));
            (min, preferred, max, finite_non_negative(spec.weight))
        })
        .collect::<Vec<_>>();
    let minimum_width: f32 = normalized.iter().map(|(min, _, _, _)| *min).sum();
    if minimum_width > available {
        return TableLayout {
            tracks: normalized.iter().map(|(min, _, _, _)| *min).collect(),
            content_width: minimum_width,
            horizontal_scroll: true,
        };
    }

    let mut tracks = normalized
        .iter()
        .map(|(min, preferred, _, _)| preferred.max(*min))
        .collect::<Vec<_>>();
    let preferred_width: f32 = tracks.iter().sum();
    if preferred_width > available {
        let shrink_needed = preferred_width - available;
        let shrink_capacity: f32 = tracks
            .iter()
            .zip(&normalized)
            .map(|(track, (min, _, _, _))| track - min)
            .sum();
        if shrink_capacity > 0.0 {
            for (track, (min, _, _, _)) in tracks.iter_mut().zip(&normalized) {
                let capacity = *track - min;
                *track -= shrink_needed * capacity / shrink_capacity;
            }
        }
    } else {
        distribute_growth(&mut tracks, &normalized, available - preferred_width);
    }

    let content_width = tracks.iter().sum();
    TableLayout {
        tracks,
        content_width,
        horizontal_scroll: false,
    }
}

fn distribute_growth(
    tracks: &mut [f32],
    specs: &[(f32, f32, Option<f32>, f32)],
    mut remaining: f32,
) {
    while remaining > 0.01 {
        let growable = tracks
            .iter()
            .zip(specs)
            .enumerate()
            .filter_map(|(index, (track, (_, _, max, weight)))| {
                let capacity = max.map(|max| (max - track).max(0.0)).unwrap_or(remaining);
                (capacity > 0.01 && *weight > 0.0).then_some((index, capacity, *weight))
            })
            .collect::<Vec<_>>();
        let total_weight: f32 = growable.iter().map(|(_, _, weight)| weight).sum();
        if growable.is_empty() || total_weight <= 0.0 {
            break;
        }
        let before = remaining;
        for (index, capacity, weight) in growable {
            let growth = (before * weight / total_weight)
                .min(capacity)
                .min(remaining);
            tracks[index] += growth;
            remaining -= growth;
        }
        if before - remaining <= 0.01 {
            break;
        }
    }
}

fn finite_non_negative(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const COLUMNS: [ColumnSpec; 3] = [
        ColumnSpec {
            min: 80.0,
            preferred: 120.0,
            max: Some(180.0),
            weight: 1.0,
        },
        ColumnSpec {
            min: 120.0,
            preferred: 200.0,
            max: None,
            weight: 2.0,
        },
        ColumnSpec {
            min: 60.0,
            preferred: 80.0,
            max: Some(100.0),
            weight: 1.0,
        },
    ];

    #[test]
    fn one_plan_is_shared_by_every_table_row() {
        let first = allocate_columns(560.0, &COLUMNS);
        let second = allocate_columns(560.0, &COLUMNS);
        assert_eq!(first, second);
        assert_eq!(first.content_width, 560.0);
        assert!(!first.horizontal_scroll);
    }

    #[test]
    fn table_scrolls_as_one_unit_below_its_minimum_width() {
        let layout = allocate_columns(200.0, &COLUMNS);
        assert_eq!(layout.tracks, vec![80.0, 120.0, 60.0]);
        assert_eq!(layout.content_width, 260.0);
        assert!(layout.horizontal_scroll);
    }

    #[test]
    fn every_track_stays_within_its_constraints() {
        for width in 0..1_000 {
            let layout = allocate_columns(width as f32, &COLUMNS);
            for (track, spec) in layout.tracks.iter().zip(COLUMNS) {
                assert!(*track >= spec.min);
                if let Some(max) = spec.max {
                    assert!(*track <= max);
                }
                assert!(track.is_finite());
            }
        }
    }

    proptest! {
        #[test]
        fn allocation_never_breaks_column_constraints(width in 0.0f32..5_000.0) {
            let layout = allocate_columns(width, &COLUMNS);
            prop_assert_eq!(layout.tracks.len(), COLUMNS.len());
            for (track, spec) in layout.tracks.iter().zip(COLUMNS) {
                prop_assert!(track.is_finite());
                prop_assert!(*track + 0.01 >= spec.min);
                if let Some(max) = spec.max {
                    prop_assert!(*track <= max + 0.01);
                }
            }
            if layout.horizontal_scroll {
                prop_assert!(layout.content_width > width);
            } else {
                prop_assert!(layout.content_width <= width + 0.1 || width < 260.0);
            }
        }
    }
}
