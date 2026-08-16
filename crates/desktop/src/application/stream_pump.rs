use kcastle_agent::AgentEvent;

pub(crate) const MAX_EVENTS_PER_FRAME: usize = 128;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StreamTelemetry {
    pub(crate) batches: u64,
    pub(crate) raw_deltas: u64,
    pub(crate) delivered_events: u64,
    pub(crate) largest_batch: usize,
}

impl StreamTelemetry {
    pub(crate) fn record(&mut self, batch: &StreamBatch) {
        self.batches = self.batches.saturating_add(1);
        self.raw_deltas = self
            .raw_deltas
            .saturating_add(batch.raw_delta_count() as u64);
        self.delivered_events = self.delivered_events.saturating_add(batch.len() as u64);
        self.largest_batch = self.largest_batch.max(batch.len());
    }
}

pub(crate) fn is_frame_stream_event(event: &AgentEvent) -> bool {
    matches!(
        event,
        AgentEvent::ReasoningDelta(_) | AgentEvent::TextDelta(_)
    )
}

pub(crate) struct StreamBatch {
    events: Vec<AgentEvent>,
    delta_count: usize,
}

impl StreamBatch {
    pub(crate) fn new(first: AgentEvent) -> Self {
        let delta_count = usize::from(is_frame_stream_event(&first));
        Self {
            events: vec![first],
            delta_count,
        }
    }

    pub(crate) fn push(&mut self, event: AgentEvent) {
        self.delta_count += usize::from(is_frame_stream_event(&event));
        match (self.events.last_mut(), event) {
            (Some(AgentEvent::ReasoningDelta(previous)), AgentEvent::ReasoningDelta(next)) => {
                previous.push_str(&next);
            }
            (Some(AgentEvent::TextDelta(previous)), AgentEvent::TextDelta(next)) => {
                previous.push_str(&next);
            }
            (_, event) => self.events.push(event),
        }
    }

    pub(crate) fn raw_delta_count(&self) -> usize {
        self.delta_count
    }

    pub(crate) fn len(&self) -> usize {
        self.events.len()
    }

    pub(crate) fn into_events(self) -> Vec<AgentEvent> {
        self.events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_text_deltas_are_coalesced_without_losing_unread_count() {
        let mut batch = StreamBatch::new(AgentEvent::TextDelta("one".into()));
        batch.push(AgentEvent::TextDelta(" two".into()));
        batch.push(AgentEvent::TextDelta(" three".into()));
        assert_eq!(batch.raw_delta_count(), 3);
        assert_eq!(batch.len(), 1);
        match batch.into_events().as_slice() {
            [AgentEvent::TextDelta(text)] => assert_eq!(text, "one two three"),
            _ => panic!("text deltas were not coalesced"),
        }
    }

    #[test]
    fn reasoning_and_answer_boundaries_are_preserved() {
        let mut batch = StreamBatch::new(AgentEvent::ReasoningDelta("think".into()));
        batch.push(AgentEvent::TextDelta("answer".into()));
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn telemetry_counts_raw_input_and_coalesced_delivery_separately() {
        let mut batch = StreamBatch::new(AgentEvent::TextDelta("a".into()));
        batch.push(AgentEvent::TextDelta("b".into()));
        batch.push(AgentEvent::TextDelta("c".into()));
        let mut telemetry = StreamTelemetry::default();
        telemetry.record(&batch);
        assert_eq!(telemetry.batches, 1);
        assert_eq!(telemetry.raw_deltas, 3);
        assert_eq!(telemetry.delivered_events, 1);
        assert_eq!(telemetry.largest_batch, 1);
    }
}
