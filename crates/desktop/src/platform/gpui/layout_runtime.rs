use std::collections::HashMap;

use crate::domain::{LayoutGeneration, MessageId};
use crate::layout::ScrollAnchor;
use crate::platform::gpui::MeasuredBounds;

#[derive(Clone, Copy, Debug)]
struct GenerationBounds {
    generation: LayoutGeneration,
    bounds: MeasuredBounds,
}

#[derive(Default)]
pub(crate) struct GpuiLayoutRuntime {
    transcript: Option<GenerationBounds>,
    messages: HashMap<MessageId, GenerationBounds>,
    pub(crate) pending_chat_anchor: Option<(LayoutGeneration, ScrollAnchor)>,
    pub(crate) restore_scheduled: bool,
    tail_realign_pending: bool,
    tail_realign_scheduled: bool,
}

impl GpuiLayoutRuntime {
    pub(crate) fn has_current_transcript(&self, generation: LayoutGeneration) -> bool {
        self.transcript
            .is_some_and(|measurement| measurement.generation == generation)
    }
}

impl GpuiLayoutRuntime {
    pub(crate) fn request_tail_realign(&mut self) {
        self.tail_realign_pending = true;
    }

    pub(crate) fn schedule_tail_realign(&mut self) -> bool {
        if !self.tail_realign_pending || self.tail_realign_scheduled {
            return false;
        }
        self.tail_realign_scheduled = true;
        true
    }

    pub(crate) fn take_tail_realign(&mut self) -> bool {
        self.tail_realign_scheduled = false;
        std::mem::take(&mut self.tail_realign_pending)
    }

    pub(crate) fn observe_transcript(
        &mut self,
        generation: LayoutGeneration,
        bounds: MeasuredBounds,
    ) {
        self.transcript = Some(GenerationBounds { generation, bounds });
    }

    pub(crate) fn observe_message(
        &mut self,
        generation: LayoutGeneration,
        id: MessageId,
        bounds: MeasuredBounds,
    ) {
        self.messages
            .insert(id, GenerationBounds { generation, bounds });
    }

    pub(crate) fn capture_chat_anchor(
        &self,
        generation: LayoutGeneration,
        following_tail: bool,
    ) -> ScrollAnchor {
        if following_tail {
            return ScrollAnchor::Tail;
        }
        let Some(viewport) = self
            .transcript
            .filter(|measurement| measurement.generation == generation)
            .map(|measurement| measurement.bounds)
        else {
            return ScrollAnchor::Tail;
        };
        self.messages
            .iter()
            .filter(|(_, measurement)| measurement.generation == generation)
            .filter(|(_, measurement)| {
                measurement.bounds.y + measurement.bounds.height >= viewport.y
                    && measurement.bounds.y <= viewport.y + viewport.height
            })
            .min_by(|(_, left), (_, right)| left.bounds.y.total_cmp(&right.bounds.y))
            .map(|(id, measurement)| ScrollAnchor::Message {
                id: *id,
                local_offset: (viewport.y - measurement.bounds.y).max(0.0),
            })
            .unwrap_or(ScrollAnchor::Tail)
    }

    pub(crate) fn restored_offset_y(
        &self,
        generation: LayoutGeneration,
        anchor: ScrollAnchor,
        current_offset_y: f32,
    ) -> Option<f32> {
        let ScrollAnchor::Message { id, local_offset } = anchor else {
            return None;
        };
        let viewport = self
            .transcript
            .filter(|measurement| measurement.generation == generation)?
            .bounds;
        let message = self
            .messages
            .get(&id)
            .filter(|measurement| measurement.generation == generation)?
            .bounds;
        let desired_message_y = viewport.y - local_offset;
        Some(current_offset_y + desired_message_y - message.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(y: f32, height: f32) -> MeasuredBounds {
        MeasuredBounds {
            x: 0.0,
            y,
            width: 600.0,
            height,
        }
    }

    #[test]
    fn captures_the_first_visible_message_as_a_semantic_anchor() {
        let generation = LayoutGeneration(2);
        let mut runtime = GpuiLayoutRuntime::default();
        runtime.observe_transcript(generation, bounds(100.0, 400.0));
        runtime.observe_message(generation, MessageId(1), bounds(20.0, 120.0));
        runtime.observe_message(generation, MessageId(2), bounds(160.0, 120.0));
        assert_eq!(
            runtime.capture_chat_anchor(generation, false),
            ScrollAnchor::Message {
                id: MessageId(1),
                local_offset: 80.0,
            }
        );
    }

    #[test]
    fn stale_measurements_cannot_restore_a_new_layout() {
        let mut runtime = GpuiLayoutRuntime::default();
        runtime.observe_transcript(LayoutGeneration(1), bounds(100.0, 400.0));
        runtime.observe_message(LayoutGeneration(1), MessageId(1), bounds(20.0, 120.0));
        assert_eq!(
            runtime.restored_offset_y(
                LayoutGeneration(2),
                ScrollAnchor::Message {
                    id: MessageId(1),
                    local_offset: 80.0,
                },
                -300.0,
            ),
            None
        );
    }

    #[test]
    fn requested_tail_realign_runs_once_after_the_next_layout() {
        let mut runtime = GpuiLayoutRuntime::default();

        runtime.request_tail_realign();
        assert!(runtime.schedule_tail_realign());
        assert!(!runtime.schedule_tail_realign());
        assert!(runtime.take_tail_realign());
        assert!(!runtime.take_tail_realign());
    }
}
