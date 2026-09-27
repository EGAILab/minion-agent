//! JavaScript-number operations observable at the built-in tool argument boundary.

pub(super) fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

pub(super) fn js_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

pub(super) fn number_to_string(number: f64) -> String {
    if number.is_nan() {
        return "NaN".into();
    }
    if number == f64::INFINITY {
        return "Infinity".into();
    }
    if number == f64::NEG_INFINITY {
        return "-Infinity".into();
    }
    if number == 0.0 {
        return "0".into();
    }
    ryu_js::Buffer::new().format(number).to_owned()
}

fn slice_index(index: f64, len: usize) -> usize {
    if index.is_nan() {
        return 0;
    }
    if index == f64::INFINITY {
        return len;
    }
    if index == f64::NEG_INFINITY {
        return 0;
    }
    let integer = index.trunc();
    if integer < 0.0 {
        js_max(len as f64 + integer, 0.0) as usize
    } else {
        js_min(integer, len as f64) as usize
    }
}

pub(super) fn js_slice<'a>(lines: &'a [&'a str], start: f64, end: Option<f64>) -> Vec<&'a str> {
    let from = slice_index(start, lines.len());
    let to = end.map_or(lines.len(), |n| slice_index(n, lines.len()));
    lines[from..to.max(from)].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_and_fractional_slice_indices_match_js() {
        let lines = ["1", "2", "3", "4"];
        assert_eq!(js_slice(&lines, 0.0, Some(-1.0)), ["1", "2", "3"]);
        assert_eq!(js_slice(&lines, 1.5, Some(2.5)), ["2"]);
        assert_eq!(number_to_string(-0.0), "0");
    }
}
