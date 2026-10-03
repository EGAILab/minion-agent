use std::time::{Duration, Instant};

use minion_agent::{argument_object::ArgumentObject, javascript::JsString};

#[test]
fn fifty_thousand_mixed_properties_build_quickly_even_with_descending_indices() {
    const HALF: usize = 25_000;
    // A deliberately generous debug-build bound, not a microbenchmark target.
    // Check during construction too, so the old whole-map-sort mutant fails
    // promptly instead of running its full quadratic workload.
    let bound = Duration::from_secs(10);
    let start = Instant::now();
    let mut object = ArgumentObject::<JsString, usize>::new();
    for i in 0..HALF {
        assert_eq!(object.insert(format!("key-{i}").into(), i), None);
        assert_eq!(object.insert((HALF - i - 1).to_string().into(), i), None);
        if i % 512 == 0 {
            assert!(start.elapsed() < bound, "construction exceeded {bound:?}");
        }
    }
    let elapsed = start.elapsed();
    eprintln!("50,000 mixed properties built in {elapsed:?}");
    assert!(elapsed < bound, "construction took {elapsed:?}");
    assert_eq!(object.len(), 2 * HALF);
    let expected = (0..HALF)
        .map(|i| i.to_string())
        .chain((0..HALF).map(|i| format!("key-{i}")))
        .collect::<Vec<_>>();
    assert_eq!(
        object
            .keys()
            .map(|key| key.to_string().unwrap())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(object.insert("key-0".into(), usize::MAX), Some(0));
    assert_eq!(object.insert("17".into(), usize::MAX), Some(HALF - 18));
    assert_eq!(object.remove(&"key-100".into()), Some(100));
    object.insert("key-100".into(), 100);
    assert_eq!(object.remove(&"17".into()), Some(usize::MAX));
    object.insert("17".into(), 17);
    assert_eq!(object.keys().next_back(), Some(&JsString::from("key-100")));
    assert_eq!(object.keys().nth(17), Some(&JsString::from("17")));
    let forward = object
        .iter()
        .map(|(key, value)| (key.clone(), *value))
        .collect::<Vec<_>>();
    assert_eq!(object.iter().len(), 2 * HALF);
    assert_eq!(
        object
            .iter()
            .rev()
            .map(|(key, value)| (key.clone(), *value))
            .collect::<Vec<_>>(),
        forward.iter().rev().cloned().collect::<Vec<_>>()
    );
    let mut mixed = object.iter();
    for _ in 0..HALF {
        mixed.next().unwrap();
        mixed.next_back().unwrap();
    }
    assert_eq!(mixed.len(), 0);
    assert_eq!(mixed.next(), None);
    assert_eq!(mixed.next_back(), None);
    assert_eq!(object.into_iter().collect::<Vec<_>>(), forward);
}
