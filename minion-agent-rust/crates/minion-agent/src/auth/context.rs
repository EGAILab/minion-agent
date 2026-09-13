use async_trait::async_trait;

#[async_trait]
pub trait AuthContext: Send + Sync {
    async fn env(&self, name: &str) -> Option<String>;
    async fn file_exists(&self, path: &str) -> bool;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultAuthContext;

#[async_trait]
impl AuthContext for DefaultAuthContext {
    async fn env(&self, name: &str) -> Option<String> {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    }

    async fn file_exists(&self, path: &str) -> bool {
        let resolved = if let Some(suffix) = path.strip_prefix('~') {
            let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));
            let Some(home) = home else { return false };
            let Some(home) = home.to_str() else {
                return false;
            };
            format!("{home}{suffix}")
        } else {
            path.to_owned()
        };
        std::path::Path::new(&resolved).exists()
    }
}
