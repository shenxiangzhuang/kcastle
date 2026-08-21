use kcastle_agent::{AgentEvent, SessionEvent};

pub(crate) const MAX_EVENTS_PER_FRAME: usize = 128;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StreamTelemetry {
    pub(crate) batches: u64,
    pub(crate) raw_deltas: u64,
    pub(crate) delivered_events: u64,
    pub(crate) largest_batch: usize,
}

#[cfg(test)]
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
    match event {
        AgentEvent::ReasoningDelta(_) | AgentEvent::TextDelta(_) => true,
        AgentEvent::SessionEvent(recorded) => {
            matches!(recorded.event, SessionEvent::AssistantChunk { .. })
        }
        _ => false,
    }
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

    #[cfg(test)]
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
    use kcastle_agent::{AssistantChunk, EventTime, RecordedEvent};
    use proptest::prelude::*;

    fn event(tag: u8, value: String) -> AgentEvent {
        match tag {
            0 => AgentEvent::ReasoningDelta(value),
            1 => AgentEvent::TextDelta(value),
            _ => AgentEvent::ModelStarted(value.parse().unwrap()),
        }
    }

    fn encoded(event: AgentEvent) -> (u8, String) {
        match event {
            AgentEvent::ReasoningDelta(value) => (0, value),
            AgentEvent::TextDelta(value) => (1, value),
            AgentEvent::ModelStarted(value) => (2, value.to_string()),
            _ => unreachable!("property strategy only generates stream and boundary events"),
        }
    }

    fn assistant_chunk(seq: u64, delta: &str) -> AgentEvent {
        AgentEvent::SessionEvent(RecordedEvent {
            seq,
            time: EventTime {
                wall_time_ms: seq as i64,
                clock_id: "stream-frame-test".into(),
                monotonic_ns: seq,
            },
            source_event_seqs: Vec::new(),
            surface_op: None,
            event: SessionEvent::AssistantChunk {
                turn: 1,
                step: 1,
                chunk: AssistantChunk::OutputTextDelta {
                    delta: delta.into(),
                },
            },
        })
    }

    #[test]
    fn durable_assistant_chunks_share_the_stream_frame_gate() {
        assert!(is_frame_stream_event(&assistant_chunk(1, "hello")));
        assert!(!is_frame_stream_event(&AgentEvent::ModelStarted(1)));
    }

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

    proptest! {
        #[test]
        fn batching_preserves_arbitrary_delta_chunks_and_boundaries(
            inputs in prop::collection::vec((0u8..3, "[^\\PC]{0,24}", 0usize..10_000), 1..100)
        ) {
            let normalized = inputs
                .into_iter()
                .map(|(tag, text, boundary)| {
                    let value = if tag < 2 { text } else { boundary.to_string() };
                    (tag, value)
                })
                .collect::<Vec<_>>();
            let expected_raw_delta_count = normalized
                .iter()
                .filter(|(tag, _)| *tag < 2)
                .count();
            let mut expected = Vec::<(u8, String)>::new();
            for (tag, value) in &normalized {
                match expected.last_mut() {
                    Some((previous_tag, previous)) if *previous_tag == *tag && *tag < 2 => {
                        previous.push_str(value);
                    }
                    _ => expected.push((*tag, value.clone())),
                }
            }

            let mut inputs = normalized.into_iter();
            let (first_tag, first_value) = inputs.next().unwrap();
            let mut batch = StreamBatch::new(event(first_tag, first_value));
            for (tag, value) in inputs {
                batch.push(event(tag, value));
            }

            let actual_raw_delta_count = batch.raw_delta_count();
            let actual = batch
                .into_events()
                .into_iter()
                .map(encoded)
                .collect::<Vec<_>>();
            prop_assert_eq!(actual, expected);
            prop_assert_eq!(actual_raw_delta_count, expected_raw_delta_count);
        }
    }
}
