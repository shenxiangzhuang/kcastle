macro_rules! numeric_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub(crate) struct $name(pub(crate) u64);

        impl $name {
            pub(crate) const fn next(self) -> Self {
                Self(self.0.saturating_add(1))
            }
        }
    };
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct MessageId(pub(crate) u64);

impl std::fmt::Display for MessageId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

pub(crate) fn next_message_id() -> MessageId {
    MessageId(NEXT_MESSAGE_ID.fetch_add(1, Ordering::Relaxed))
}

numeric_id!(RunId);
numeric_id!(LayoutGeneration);
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_MESSAGE_ID: AtomicU64 = AtomicU64::new(1);
