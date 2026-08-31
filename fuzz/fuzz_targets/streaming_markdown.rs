#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/desktop/src/streaming_markdown.rs"]
#[allow(dead_code)]
mod streaming_markdown;

fuzz_target!(|data: &[u8]| {
    assert!(streaming_markdown::fuzz_updates(data));
});
