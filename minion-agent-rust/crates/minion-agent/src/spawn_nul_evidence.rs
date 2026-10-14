//! Test-only observed panic identity/origin. Not production error semantics.
use std::sync::Once;

pub(crate) fn install() {
    let Ok(token) = std::env::var("MINION_NUL_EVIDENCE_TOKEN") else {
        return;
    };
    static INSTALLED: Once = Once::new();
    INSTALLED.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic| {
            let (payload_type, message) =
                if let Some(value) = panic.payload().downcast_ref::<String>() {
                    ("String", value.as_str())
                } else if let Some(value) = panic.payload().downcast_ref::<&'static str>() {
                    ("str", *value)
                } else {
                    ("other", "")
                };
            let location = panic.location();
            let record = serde_json::json!({
                "token": token,
                "thread": std::thread::current().name(),
                "payload_type": payload_type,
                "message": message,
                "file": location.map(|l| l.file()),
                "line": location.map(|l| l.line()),
                "column": location.map(|l| l.column()),
            });
            eprintln!("MINION_NUL_PANIC {record}");
            previous(panic);
        }));
    });
}
