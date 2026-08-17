use std::cell::RefCell;
use std::rc::Rc;

use gpui::{ScrollHandle, Window, point, px};

const VISUAL_UPDATE_INTERVAL_FRAMES: u8 = 3;

#[cfg(test)]
pub(crate) fn arm_next_frame(window: &mut Window, ready: tokio::sync::oneshot::Sender<()>) {
    window.on_next_frame(move |_, _| {
        let _ = ready.send(());
    });
    window.refresh();
}

#[derive(Clone, Debug, Default)]
pub(crate) struct FrameThrottledScroll {
    scroll: ScrollHandle,
    state: Rc<RefCell<FrameThrottleState>>,
}

impl FrameThrottledScroll {
    pub(crate) fn handle(&self) -> ScrollHandle {
        self.scroll.clone()
    }

    pub(crate) fn follow_end(&self, revision: u64, window: &mut Window) {
        let Some(generation) = self.state.borrow_mut().request(revision) else {
            return;
        };
        schedule_follow_end(self.scroll.clone(), self.state.clone(), generation, window);
    }

    pub(crate) fn cancel_and_reset(&self) {
        self.state.borrow_mut().cancel();
        if self.scroll.offset().x != px(0.0) {
            let offset = self.scroll.offset();
            self.scroll.set_offset(point(px(0.0), offset.y));
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameAdvance {
    Cancelled,
    Pending,
    Ready,
}

#[derive(Debug, Default)]
struct FrameThrottleState {
    remaining_frames: u8,
    generation: u64,
    last_revision: Option<u64>,
}

impl FrameThrottleState {
    fn request(&mut self, revision: u64) -> Option<u64> {
        if self.last_revision == Some(revision) {
            return None;
        }
        self.last_revision = Some(revision);
        if self.remaining_frames > 0 {
            return None;
        }
        self.generation = self.generation.wrapping_add(1);
        self.remaining_frames = VISUAL_UPDATE_INTERVAL_FRAMES;
        Some(self.generation)
    }

    fn advance(&mut self, generation: u64) -> FrameAdvance {
        if generation != self.generation || self.remaining_frames == 0 {
            return FrameAdvance::Cancelled;
        }
        self.remaining_frames -= 1;
        if self.remaining_frames == 0 {
            FrameAdvance::Ready
        } else {
            FrameAdvance::Pending
        }
    }

    fn cancel(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.remaining_frames = 0;
        self.last_revision = None;
    }
}

fn schedule_follow_end(
    scroll: ScrollHandle,
    state: Rc<RefCell<FrameThrottleState>>,
    generation: u64,
    window: &mut Window,
) {
    window.on_next_frame(move |window, _| {
        let advance = state.borrow_mut().advance(generation);
        match advance {
            FrameAdvance::Cancelled => {}
            FrameAdvance::Pending => {
                schedule_follow_end(scroll, state, generation, window);
            }
            FrameAdvance::Ready => {
                scroll.scroll_to_item(0);
                window.refresh();
            }
        }
    });
    window.refresh();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visual_scroll_requests_coalesce_until_the_third_frame() {
        let mut state = FrameThrottleState::default();
        let generation = state.request(1).expect("first request should schedule");

        assert_eq!(state.request(2), None);
        assert_eq!(state.advance(generation), FrameAdvance::Pending);
        assert_eq!(state.advance(generation), FrameAdvance::Pending);
        assert_eq!(state.advance(generation), FrameAdvance::Ready);
        assert_eq!(state.request(2), None);
        assert!(state.request(3).is_some());
    }

    #[test]
    fn cancelling_a_visual_scroll_invalidates_it_and_restores_the_leading_edge() {
        let scroll = FrameThrottledScroll::default();
        scroll.scroll.set_offset(point(px(-80.0), px(0.0)));
        let generation = scroll
            .state
            .borrow_mut()
            .request(1)
            .expect("request should schedule");

        scroll.cancel_and_reset();

        assert_eq!(
            scroll.state.borrow_mut().advance(generation),
            FrameAdvance::Cancelled
        );
        assert_eq!(scroll.scroll.offset().x, px(0.0));
    }
}
