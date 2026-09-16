use std::{collections::BTreeMap, future::Future, sync::Arc, time::Duration};

use async_trait::async_trait;
use thiserror::Error;
use tokio::sync::{Mutex, Notify};

use super::Abortable;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpRequest {
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum HttpTransportError {
    #[error("{0}")]
    Request(String),
    #[error("{0}")]
    Body(String),
    #[error("operation cancelled")]
    Cancelled,
}

#[async_trait]
pub trait HttpResponseBody: Send {
    async fn text(
        self: Box<Self>,
        signal: Arc<dyn Abortable>,
    ) -> Result<String, HttpTransportError>;

    async fn discard(self: Box<Self>) -> Result<(), HttpTransportError> {
        Ok(())
    }
}

struct StaticBody(String);

#[async_trait]
impl HttpResponseBody for StaticBody {
    async fn text(
        self: Box<Self>,
        _signal: Arc<dyn Abortable>,
    ) -> Result<String, HttpTransportError> {
        Ok(self.0)
    }
}

enum HttpResponseState {
    Unread(Box<dyn HttpResponseBody>),
    Reading,
    Cached(Result<String, HttpTransportError>),
    Discarded,
}

#[derive(Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub reason_phrase: String,
    state: Arc<Mutex<HttpResponseState>>,
    changed: Arc<Notify>,
}

impl HttpResponse {
    pub fn new(status: u16, reason_phrase: impl Into<String>, body: impl Into<String>) -> Self {
        Self::deferred(status, reason_phrase, Box::new(StaticBody(body.into())))
    }

    pub fn deferred(
        status: u16,
        reason_phrase: impl Into<String>,
        body: Box<dyn HttpResponseBody>,
    ) -> Self {
        Self {
            status,
            reason_phrase: reason_phrase.into(),
            state: Arc::new(Mutex::new(HttpResponseState::Unread(body))),
            changed: Arc::new(Notify::new()),
        }
    }

    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    pub async fn text(&self, signal: Arc<dyn Abortable>) -> Result<String, HttpTransportError> {
        loop {
            let notified = self.changed.notified();
            let body = {
                let mut state = self.state.lock().await;
                match &*state {
                    HttpResponseState::Cached(result) => return result.clone(),
                    HttpResponseState::Discarded => return Ok(String::new()),
                    HttpResponseState::Reading => None,
                    HttpResponseState::Unread(_) => {
                        let HttpResponseState::Unread(body) =
                            std::mem::replace(&mut *state, HttpResponseState::Reading)
                        else {
                            unreachable!()
                        };
                        Some(body)
                    }
                }
            };
            if let Some(body) = body {
                let state = self.state.clone();
                let changed = self.changed.clone();
                let signal = signal.clone();
                tokio::spawn(async move {
                    let result = body.text(signal).await.map(strip_one_leading_bom);
                    *state.lock().await = HttpResponseState::Cached(result);
                    changed.notify_waiters();
                });
            }
            notified.await;
        }
    }

    /// Abandon an unread body. Cleanup failures are deliberately unobservable.
    pub async fn discard(&self) {
        loop {
            let notified = self.changed.notified();
            let body = {
                let mut state = self.state.lock().await;
                match &*state {
                    HttpResponseState::Reading => None,
                    HttpResponseState::Unread(_) => {
                        let HttpResponseState::Unread(body) =
                            std::mem::replace(&mut *state, HttpResponseState::Discarded)
                        else {
                            unreachable!()
                        };
                        Some(body)
                    }
                    HttpResponseState::Cached(_) | HttpResponseState::Discarded => return,
                }
            };
            let Some(body) = body else {
                notified.await;
                continue;
            };
            let _ = body.discard().await;
            self.changed.notify_waiters();
            return;
        }
    }
}

fn strip_one_leading_bom(mut value: String) -> String {
    if value.starts_with('\u{feff}') {
        value.remove(0);
    }
    value
}

#[async_trait]
pub trait HttpTransport: Send + Sync {
    async fn post(
        &self,
        request: HttpRequest,
        signal: Arc<dyn Abortable>,
    ) -> Result<HttpResponse, HttpTransportError>;
}

#[derive(Clone)]
pub struct ReqwestTransport {
    client: reqwest::Client,
}

impl ReqwestTransport {
    pub fn new() -> Result<Self, HttpTransportError> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::limited(20))
            .build()
            .map_err(|error| HttpTransportError::Request(error.to_string()))?;
        Ok(Self { client })
    }
}

struct ReqwestBody(reqwest::Response);

#[async_trait]
impl HttpResponseBody for ReqwestBody {
    async fn text(
        self: Box<Self>,
        signal: Arc<dyn Abortable>,
    ) -> Result<String, HttpTransportError> {
        let bytes = run_cancellable(self.0.bytes(), signal)
            .await?
            .map_err(|error| HttpTransportError::Body(error.to_string()))?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

#[async_trait]
impl HttpTransport for ReqwestTransport {
    async fn post(
        &self,
        request: HttpRequest,
        signal: Arc<dyn Abortable>,
    ) -> Result<HttpResponse, HttpTransportError> {
        let mut builder = self.client.post(&request.url).body(request.body);
        for (name, value) in request.headers {
            builder = builder.header(name, value);
        }
        let response = run_cancellable(builder.send(), signal)
            .await?
            .map_err(|error| HttpTransportError::Request(error.to_string()))?;
        let status = response.status();
        Ok(HttpResponse::deferred(
            status.as_u16(),
            status.canonical_reason().unwrap_or_default(),
            Box::new(ReqwestBody(response)),
        ))
    }
}

pub async fn run_cancellable<F, T>(
    future: F,
    signal: Arc<dyn Abortable>,
) -> Result<T, HttpTransportError>
where
    F: Future<Output = T>,
{
    if signal.aborted() {
        return Err(HttpTransportError::Cancelled);
    }
    tokio::pin!(future);
    loop {
        tokio::select! {
            result = &mut future => return Ok(result),
            () = tokio::time::sleep(Duration::from_millis(10)) => {
                if signal.aborted() {
                    return Err(HttpTransportError::Cancelled);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;
    use crate::auth::AuthAbortController;

    struct TrackingBody {
        read: Arc<AtomicBool>,
        discarded: Arc<AtomicBool>,
        value: Result<String, HttpTransportError>,
    }

    #[async_trait]
    impl HttpResponseBody for TrackingBody {
        async fn text(
            self: Box<Self>,
            _signal: Arc<dyn Abortable>,
        ) -> Result<String, HttpTransportError> {
            self.read.store(true, Ordering::Release);
            self.value
        }
        async fn discard(self: Box<Self>) -> Result<(), HttpTransportError> {
            self.discarded.store(true, Ordering::Release);
            Ok(())
        }
    }

    #[tokio::test]
    async fn response_body_is_lazy_bom_stripped_cached_and_discardable() {
        let read = Arc::new(AtomicBool::new(false));
        let discarded = Arc::new(AtomicBool::new(false));
        let response = HttpResponse::deferred(
            404,
            "Not Found",
            Box::new(TrackingBody {
                read: read.clone(),
                discarded: discarded.clone(),
                value: Ok("\u{feff}body\u{feff}".into()),
            }),
        );
        assert!(!read.load(Ordering::Acquire));
        let signal = AuthAbortController::default().signal();
        assert_eq!(response.text(signal.clone()).await.unwrap(), "body\u{feff}");
        assert_eq!(response.text(signal).await.unwrap(), "body\u{feff}");
        response.discard().await;
        assert!(!discarded.load(Ordering::Acquire));

        let response = HttpResponse::deferred(
            404,
            "Not Found",
            Box::new(TrackingBody {
                read,
                discarded: discarded.clone(),
                value: Ok("never".into()),
            }),
        );
        response.discard().await;
        assert!(discarded.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn cancellation_drops_an_in_flight_operation_promptly() {
        let controller = AuthAbortController::default();
        let signal = controller.signal();
        let abort = controller.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(1)).await;
            abort.abort();
        });
        assert_eq!(
            run_cancellable(std::future::pending::<()>(), signal).await,
            Err(HttpTransportError::Cancelled)
        );
    }
}
