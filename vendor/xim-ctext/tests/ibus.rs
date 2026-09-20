use xim_ctext::compound_text_to_utf8;

#[test]
fn ubuntu_ibus_compound_text_commits() {
    // Xutf8TextListToTextProperty(XCompoundTextStyle), as used by ibus-x11,
    // on Ubuntu 24.04 with en_US.UTF-8. A commit can switch charsets mid-string.
    for (encoded, expected) in [
        ("1b24284144631b2428423925", "你好"),
        ("1b24284243664a38", "中文"),
        (
            "1b24284144631b24284239251b284220776f726c6420313233",
            "你好 world 123",
        ),
        ("68656c6c6f201b24284243664a38212a", "hello 中文！"),
        ("1b24284243664a381b2547f09f98801b2540", "中文😀"),
        ("1b24284132624a544a641b242842467e", "测试输入"),
        ("1b242842484b717343664a38", "繁體中文"),
        (
            "1b2428423a23453740271b2842323032361b242842472f1b2842391b242842376e1b284232301b242842467c",
            "今天是2026年9月20日",
        ),
    ] {
        let bytes: Vec<_> = (0..encoded.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&encoded[i..i + 2], 16).unwrap())
            .collect();
        assert_eq!(compound_text_to_utf8(&bytes).unwrap(), expected);
    }
}

#[test]
fn utf8_compound_text_round_trips() {
    for text in ["", "ASCII", "中文😀", "日本語", "한국어"] {
        assert_eq!(
            compound_text_to_utf8(&xim_ctext::utf8_to_compound_text(text)).unwrap(),
            text,
        );
    }
}
