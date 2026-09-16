//! One byte-bounded cache for all sessions in this window. It owns no GPUI entities.
use std::{
    collections::HashMap,
    ops::Range,
    sync::Arc,
    time::{Duration, Instant},
};

use super::{PreparedMarkdown, SourceChunk};
use crate::domain::MessageId;

pub(super) const IDLE_TTL: Duration = Duration::from_secs(5 * 60);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct CacheKey {
    pub namespace: String,
    pub lineage: u64,
    pub message: MessageId,
    pub revision: u64,
    pub field: u8,
    pub source: Range<usize>,
    pub dark: bool,
    pub index: bool,
}

pub(super) enum Cached {
    Index(Vec<SourceChunk>),
    Markdown(Arc<PreparedMarkdown>),
}

impl Cached {
    fn bytes(&self) -> usize {
        match self {
            Self::Index(chunks) => {
                chunks.capacity() * size_of::<SourceChunk>()
                    + chunks
                        .iter()
                        .filter_map(|c| c.fence.as_ref())
                        .map(String::capacity)
                        .sum::<usize>()
            }
            Self::Markdown(prepared) => prepared.bytes(),
        }
    }

    fn in_use(&self) -> bool {
        matches!(self, Self::Markdown(prepared) if Arc::strong_count(prepared) > 1)
    }
}

struct Entry {
    value: Cached,
    bytes: usize,
    used: Instant,
}

pub(super) struct PreparationCache {
    entries: HashMap<CacheKey, Entry>,
    pub priorities: HashMap<CacheKey, u8>,
    pub bytes: usize,
    pub budget: usize,
}

impl PreparationCache {
    pub fn new(budget: usize) -> Self {
        Self {
            entries: HashMap::new(),
            priorities: HashMap::new(),
            bytes: 0,
            budget,
        }
    }

    pub fn get(&mut self, key: &CacheKey, now: Instant) -> Option<&Cached> {
        let entry = self.entries.get_mut(key)?;
        entry.used = now;
        Some(&entry.value)
    }

    pub fn peek(&self, key: &CacheKey) -> Option<&Cached> {
        self.entries.get(key).map(|entry| &entry.value)
    }

    pub fn expire(&mut self, now: Instant) {
        self.entries.retain(|key, entry| {
            // Reading a stationary viewport is still use. Keep its semantic index
            // too, so a later quick session round-trip can restore rich rows.
            if self.priorities.get(key) == Some(&2) {
                entry.used = now;
            }
            let keep = entry.value.in_use() || now.saturating_duration_since(entry.used) < IDLE_TTL;
            if !keep {
                self.bytes -= entry.bytes;
            }
            keep
        });
    }

    pub fn insert(&mut self, key: CacheKey, value: Cached, now: Instant) -> bool {
        if self.entries.contains_key(&key) {
            return true;
        }
        let bytes = value.bytes() + size_of::<(CacheKey, Entry)>() + key.namespace.capacity();
        if bytes > self.budget {
            return false;
        }
        self.expire(now);
        // ponytail: linear LRU eviction over a byte-bounded cache. Use an ordered map
        // only if profiling shows eviction scans matter.
        while self.bytes + bytes > self.budget {
            let priority = self.priorities.get(&key).copied().unwrap_or(0);
            let oldest = self
                .entries
                .iter()
                .filter(|(key, entry)| {
                    let other = self.priorities.get(*key).copied().unwrap_or(0);
                    !entry.value.in_use() && (other == 0 || other < priority)
                })
                .min_by_key(|(key, entry)| {
                    (self.priorities.get(*key).copied().unwrap_or(0), entry.used)
                })
                .map(|(key, _)| key.clone());
            let Some(oldest) = oldest else {
                return false;
            };
            if let Some(entry) = self.entries.remove(&oldest) {
                self.bytes -= entry.bytes;
            }
        }
        self.entries.insert(
            key,
            Entry {
                value,
                bytes,
                used: now,
            },
        );
        self.bytes += bytes;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn key(id: u64) -> CacheKey {
        CacheKey {
            namespace: "session".into(),
            lineage: 0,
            message: MessageId(id),
            revision: 0,
            field: 1,
            source: 0..4,
            dark: false,
            index: false,
        }
    }

    fn value() -> Cached {
        Cached::Markdown(Arc::new(
            crate::dsh_markdown::prepare_markdown(
                "text",
                crate::ui_theme::markdown_highlight_theme(false),
                &AtomicBool::new(false),
            )
            .unwrap(),
        ))
    }

    #[test]
    fn admission_is_bounded_prioritizes_visible_and_preserves_live_results() {
        let now = Instant::now();
        let mut cache = PreparationCache::new(usize::MAX);
        assert!(cache.insert(key(1), value(), now));
        let one = cache.bytes;
        cache.budget = 2 * one;
        assert!(cache.insert(key(2), value(), now + Duration::from_secs(1)));
        // The older visible result beats the newer offscreen one.
        cache.priorities.insert(key(1), 2);
        cache.priorities.insert(key(2), 1);
        cache.priorities.insert(key(3), 2);
        assert!(cache.insert(key(3), value(), now + Duration::from_secs(2)));
        assert!(cache.peek(&key(1)).is_some());
        assert!(cache.peek(&key(2)).is_none());
        assert_eq!(cache.bytes, 2 * one);
        assert!(!cache.insert(key(4), value(), now));
        assert_eq!(cache.bytes, 2 * one);
        // Several code slices hold one allocation, charged only once.
        let Some(Cached::Markdown(prepared)) = cache.get(&key(1), now) else {
            panic!()
        };
        let slice_a = prepared.clone();
        let slice_b = prepared.clone();
        cache.priorities.clear();
        cache.expire(now + IDLE_TTL);
        assert!(cache.peek(&key(1)).is_some());
        drop((slice_a, slice_b));
        cache.expire(now + IDLE_TTL + Duration::from_secs(3));
        assert_eq!(cache.bytes, 0);
        cache.budget = one - 1;
        assert!(!cache.insert(key(1), value(), now));
        assert_eq!(cache.bytes, 0);
    }

    #[test]
    fn inactive_results_use_lru_across_sessions_and_freshness_keys_do_not_alias() {
        let now = Instant::now();
        let mut cache = PreparationCache::new(usize::MAX);
        assert!(cache.insert(key(1), value(), now));
        cache.budget = cache.bytes * 2;
        let mut other_session = key(1);
        other_session.namespace = "another".into();
        assert!(cache.insert(other_session.clone(), value(), now + Duration::from_secs(1)));
        cache.get(&key(1), now + Duration::from_secs(2));
        assert!(cache.insert(key(3), value(), now + Duration::from_secs(3)));
        assert!(cache.peek(&key(1)).is_some());
        assert!(cache.peek(&other_session).is_none());
        for changed in 0..3 {
            let mut different = key(1);
            match changed {
                0 => different.revision += 1,
                1 => different.lineage += 1,
                _ => different.dark = true,
            }
            assert!(cache.peek(&different).is_none());
        }
    }
}
