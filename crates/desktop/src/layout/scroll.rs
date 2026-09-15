use crate::domain::MessageId;

/// A source locator survives eviction and plain-text/rich-text replacement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ScrollAnchor {
    Tail,
    Block {
        id: MessageId,
        field: u8,
        source_offset: usize,
        local_offset: f32,
    },
}
