use minion_agent::agent_loop::{TurnStopping, resolve_stopping};

#[test]
fn stopping_uses_the_first_concrete_listener_opinion() {
    assert_eq!(
        resolve_stopping([
            TurnStopping::NoOpinion,
            TurnStopping::Continue,
            TurnStopping::Stop,
        ]),
        TurnStopping::Continue
    );
    assert_eq!(
        resolve_stopping([
            TurnStopping::NoOpinion,
            TurnStopping::Stop,
            TurnStopping::Continue,
        ]),
        TurnStopping::Stop
    );
}

#[test]
fn stopping_defaults_to_continue_without_a_concrete_opinion() {
    assert_eq!(resolve_stopping([]), TurnStopping::Continue);
    assert_eq!(
        resolve_stopping([TurnStopping::NoOpinion, TurnStopping::NoOpinion]),
        TurnStopping::Continue
    );
}
