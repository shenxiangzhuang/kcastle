use crate::domain::{LayoutGeneration, MessageId};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ScrollAnchor {
    Tail,
    Message { id: MessageId, local_offset: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ScrollRestore {
    Tail,
    Message { id: MessageId, local_offset: f32 },
    IgnoreStale,
}

pub(crate) fn resolve_scroll_restore(
    requested_generation: LayoutGeneration,
    current_generation: LayoutGeneration,
    anchor: ScrollAnchor,
) -> ScrollRestore {
    if requested_generation != current_generation {
        return ScrollRestore::IgnoreStale;
    }
    match anchor {
        ScrollAnchor::Tail => ScrollRestore::Tail,
        ScrollAnchor::Message { id, local_offset } => ScrollRestore::Message {
            id,
            local_offset: if local_offset.is_finite() {
                local_offset.max(0.0)
            } else {
                0.0
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_reflow_cannot_move_the_current_view() {
        assert_eq!(
            resolve_scroll_restore(LayoutGeneration(2), LayoutGeneration(3), ScrollAnchor::Tail,),
            ScrollRestore::IgnoreStale,
        );
    }

    #[test]
    fn message_anchor_survives_reflow_without_raw_document_offset() {
        let anchor = ScrollAnchor::Message {
            id: MessageId(42),
            local_offset: 12.0,
        };
        assert_eq!(
            resolve_scroll_restore(LayoutGeneration(4), LayoutGeneration(4), anchor),
            ScrollRestore::Message {
                id: MessageId(42),
                local_offset: 12.0,
            }
        );
    }
}
