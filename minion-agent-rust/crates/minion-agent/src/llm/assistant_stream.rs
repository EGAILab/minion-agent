use std::{
    pin::Pin,
    task::{Context, Poll},
};

use futures::{Stream, stream::FusedStream};

use super::{
    AdapterStreamError, AdapterStreamErrorKind, AssistantMessage, DoneReason, ErrorReason,
    RawAssistantStream, StopReason, StreamChunk,
};

pub struct AssistantStream {
    raw: Option<RawAssistantStream>,
    partial: AssistantMessage,
    fused: bool,
}

impl AssistantStream {
    pub(crate) fn new(raw: RawAssistantStream, partial: AssistantMessage) -> Self {
        Self {
            raw: Some(raw),
            partial,
            fused: false,
        }
    }

    fn settle_error(&mut self, error: AdapterStreamError) -> StreamChunk {
        let aborted = error.kind == AdapterStreamErrorKind::Cancelled;
        let mut message = self.partial.clone();
        message.stop_reason = if aborted {
            StopReason::Aborted
        } else {
            StopReason::Error
        };
        message.error_message = Some(error.message);
        self.fused = true;
        self.raw = None;
        StreamChunk::Error {
            reason: if aborted {
                ErrorReason::Aborted
            } else {
                ErrorReason::Error
            },
            error: message,
        }
    }
}

impl Stream for AssistantStream {
    type Item = StreamChunk;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();
        if this.fused {
            return Poll::Ready(None);
        }

        let polled = this
            .raw
            .as_mut()
            .expect("unfused stream retains raw source")
            .as_mut()
            .poll_next(cx);
        match polled {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(Ok(mut chunk))) => {
                match &mut chunk {
                    StreamChunk::Done { reason, message } => {
                        message.stop_reason = match reason {
                            DoneReason::Stop => StopReason::Stop,
                            DoneReason::Length => StopReason::Length,
                            DoneReason::ToolUse => StopReason::ToolUse,
                            DoneReason::Deferred => StopReason::Deferred,
                        };
                    }
                    StreamChunk::Error { reason, error } => {
                        error.stop_reason = match reason {
                            ErrorReason::Error => StopReason::Error,
                            ErrorReason::Aborted => StopReason::Aborted,
                        };
                    }
                    _ => {}
                }
                this.partial = chunk.partial().clone();
                if chunk.is_terminal() {
                    this.fused = true;
                    this.raw = None;
                }
                Poll::Ready(Some(chunk))
            }
            Poll::Ready(Some(Err(error))) => Poll::Ready(Some(this.settle_error(error))),
            Poll::Ready(None) => Poll::Ready(Some(this.settle_error(AdapterStreamError::new(
                AdapterStreamErrorKind::Protocol,
                "adapter stream ended before a terminal response",
            )))),
        }
    }
}

impl FusedStream for AssistantStream {
    fn is_terminated(&self) -> bool {
        self.fused
    }
}
