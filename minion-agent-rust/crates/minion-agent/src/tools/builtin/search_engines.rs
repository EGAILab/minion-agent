//! TOOL-038 / DIV-003: explicit provisioning, never implicit acquisition.
use crate::{execution::ExecutionWorldIdentity, tools::ToolCapabilityError};
use async_trait::async_trait;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchEngine {
    Fd,
    Ripgrep,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchEnginePin {
    pub artifact: &'static str,
    pub artifact_sha256: &'static str,
    pub member: &'static str,
    pub binary_sha256: &'static str,
    pub url: &'static str,
}

impl SearchEngine {
    pub fn name(self) -> &'static str {
        match self {
            Self::Fd => "fd",
            Self::Ripgrep => "ripgrep (rg)",
        }
    }
    pub fn version(self) -> &'static str {
        match self {
            Self::Fd => "10.4.2",
            Self::Ripgrep => "15.2.0",
        }
    }
    pub fn executable(self) -> &'static str {
        match (self, cfg!(windows)) {
            (Self::Fd, true) => "fd.exe",
            (Self::Fd, false) => "fd",
            (Self::Ripgrep, true) => "rg.exe",
            (Self::Ripgrep, false) => "rg",
        }
    }
    fn engine_name(self) -> &'static str {
        match self {
            Self::Fd => "fd",
            Self::Ripgrep => "ripgrep",
        }
    }
    fn unavailable(self) -> ToolCapabilityError {
        ToolCapabilityError::new(format!(
            "{} is not provisioned: the certified {} {} engine is missing or failed verification. Run provision_search_engines() to provision it.",
            self.name(),
            self.engine_name(),
            self.version()
        ))
    }
    pub fn pin(self) -> Option<SearchEnginePin> {
        if std::env::consts::ARCH != "x86_64" {
            return None;
        }
        Some(match (self, std::env::consts::OS) {
            (Self::Fd, "windows") => SearchEnginePin {
                artifact: "fd-v10.4.2-x86_64-pc-windows-msvc.zip",
                artifact_sha256: "b2816e506390a89941c63c9187d58a3cc10e9a55f2ef0685f9ea0eccaf7c98c8",
                member: "fd-v10.4.2-x86_64-pc-windows-msvc/fd.exe",
                binary_sha256: "4c9d082ee20f0d9e44881ac4e92adf765efc314d82103c53d7f576bd78dc5761",
                url: "https://github.com/sharkdp/fd/releases/download/v10.4.2/fd-v10.4.2-x86_64-pc-windows-msvc.zip",
            },
            (Self::Fd, "linux") => SearchEnginePin {
                artifact: "fd-v10.4.2-x86_64-unknown-linux-gnu.tar.gz",
                artifact_sha256: "def59805cd14b5651b68990855f426ad087f3b96881296d963910431ba3143c8",
                member: "fd-v10.4.2-x86_64-unknown-linux-gnu/fd",
                binary_sha256: "0dff4a420feb3e57fd1d4402d3e29f46115aa38d962467d2f3b72e7439d3ada8",
                url: "https://github.com/sharkdp/fd/releases/download/v10.4.2/fd-v10.4.2-x86_64-unknown-linux-gnu.tar.gz",
            },
            (Self::Ripgrep, "windows") => SearchEnginePin {
                artifact: "ripgrep-15.2.0-x86_64-pc-windows-msvc.zip",
                artifact_sha256: "71b2fef860abe467217a538ff31de02f5258807c0129f771846f87bd029aafc5",
                member: "ripgrep-15.2.0-x86_64-pc-windows-msvc/rg.exe",
                binary_sha256: "14231169855ec5205cf5a1b6f1db358ff4aed4247c86b69ce8aae647c77f6680",
                url: "https://github.com/BurntSushi/ripgrep/releases/download/15.2.0/ripgrep-15.2.0-x86_64-pc-windows-msvc.zip",
            },
            (Self::Ripgrep, "linux") => SearchEnginePin {
                artifact: "ripgrep-15.2.0-x86_64-unknown-linux-musl.tar.gz",
                artifact_sha256: "33e15bcf1624b25cdd2a55813a47a2f95dbe126268203e76aa6a585d1e7b149c",
                member: "ripgrep-15.2.0-x86_64-unknown-linux-musl/rg",
                binary_sha256: "e62198eb19b136b88c330af83647b5a962cb99b6b1f066758568f12de1974849",
                url: "https://github.com/BurntSushi/ripgrep/releases/download/15.2.0/ripgrep-15.2.0-x86_64-unknown-linux-musl.tar.gz",
            },
            _ => return None,
        })
    }
}

/// Resolve is called before EACH spawn. Test/extension overrides are explicitly uncertified.
#[async_trait]
pub trait SearchEngines: Send + Sync {
    async fn resolve(
        &self,
        engine: SearchEngine,
        world: &ExecutionWorldIdentity,
    ) -> Result<PathBuf, ToolCapabilityError>;
}

#[derive(Clone, Debug)]
pub struct SearchEngineStore {
    root: PathBuf,
    #[cfg(test)]
    verification_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}
impl Default for SearchEngineStore {
    fn default() -> Self {
        let home = std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        Self::new(home.join(".minion").join("search-engines"))
    }
}
impl SearchEngineStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            #[cfg(test)]
            verification_calls: Default::default(),
        }
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    async fn verified(&self, engine: SearchEngine, pin: SearchEnginePin) -> bool {
        #[cfg(test)]
        self.verification_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        tokio::fs::read(self.root.join(engine.executable()))
            .await
            .is_ok_and(|bytes| hash(&bytes) == pin.binary_sha256)
    }
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[async_trait]
impl SearchEngines for SearchEngineStore {
    async fn resolve(
        &self,
        engine: SearchEngine,
        world: &ExecutionWorldIdentity,
    ) -> Result<PathBuf, ToolCapabilityError> {
        if world != &ExecutionWorldIdentity::local() {
            return Err(ToolCapabilityError::new(format!(
                "{} is not available on this platform: no certified {} engine for {} (non-local execution world).",
                engine.name(),
                engine.engine_name(),
                world.as_str()
            )));
        }
        let pin = engine.pin().ok_or_else(|| {
            ToolCapabilityError::new(format!(
                "{} is not available on this platform: no certified {} engine for {}-{}.",
                engine.name(),
                engine.engine_name(),
                if cfg!(windows) {
                    "win32"
                } else {
                    std::env::consts::OS
                },
                if std::env::consts::ARCH == "x86_64" {
                    "x64"
                } else {
                    std::env::consts::ARCH
                }
            ))
        })?;
        if !self.verified(engine, pin).await {
            return Err(engine.unavailable());
        }
        Ok(self.root.join(engine.executable()))
    }
}

/// Explicit uncertified executable selection; never a managed-store fallback.
#[derive(Clone, Debug)]
pub struct UncertifiedSearchEngines {
    pub fd: PathBuf,
    pub rg: PathBuf,
}
#[async_trait]
impl SearchEngines for UncertifiedSearchEngines {
    async fn resolve(
        &self,
        engine: SearchEngine,
        _: &ExecutionWorldIdentity,
    ) -> Result<PathBuf, ToolCapabilityError> {
        Ok(match engine {
            SearchEngine::Fd => self.fd.clone(),
            SearchEngine::Ripgrep => self.rg.clone(),
        })
    }
}

/// `None` obtains official pinned artifacts; `Some` reads their official names locally.
/// The host archive utility is a provisioning-only adapter: it extracts the single pinned
/// member from an already hash-verified artifact. Tools never invoke it or the network.
pub async fn provision_search_engines(
    store: &SearchEngineStore,
    source: Option<&Path>,
) -> Result<(), ToolCapabilityError> {
    for engine in [SearchEngine::Fd, SearchEngine::Ripgrep] {
        let pin = engine.pin().ok_or_else(|| engine.unavailable())?;
        if store.verified(engine, pin).await {
            continue;
        }
        let artifact = match source {
            Some(dir) => tokio::fs::read(dir.join(pin.artifact))
                .await
                .map_err(|e| ToolCapabilityError::new(e.to_string()))?,
            None => reqwest::get(pin.url)
                .await
                .map_err(|e| ToolCapabilityError::new(e.to_string()))?
                .error_for_status()
                .map_err(|e| ToolCapabilityError::new(e.to_string()))?
                .bytes()
                .await
                .map_err(|e| ToolCapabilityError::new(e.to_string()))?
                .to_vec(),
        };
        if hash(&artifact) != pin.artifact_sha256 {
            return Err(ToolCapabilityError::new(
                "search engine artifact failed verification",
            ));
        }
        tokio::fs::create_dir_all(&store.root)
            .await
            .map_err(|e| ToolCapabilityError::new(e.to_string()))?;
        let archive = store
            .root
            .join(format!(".{}.archive", uuid::Uuid::new_v4()));
        tokio::fs::write(&archive, &artifact)
            .await
            .map_err(|e| ToolCapabilityError::new(e.to_string()))?;
        let output = tokio::process::Command::new("tar")
            .args(["-xOf"])
            .arg(&archive)
            .arg(pin.member)
            .output()
            .await;
        let _ = tokio::fs::remove_file(&archive).await;
        let output = output.map_err(|e| ToolCapabilityError::new(e.to_string()))?;
        if !output.status.success() || hash(&output.stdout) != pin.binary_sha256 {
            return Err(ToolCapabilityError::new(
                "search engine binary failed verification",
            ));
        }
        let temp = stage_binary(store, &output.stdout).await?;
        let installed = tokio::fs::rename(&temp, store.root.join(engine.executable())).await;
        if let Err(error) = installed {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(ToolCapabilityError::new(error.to_string()));
        }
    }
    Ok(())
}

async fn stage_binary(
    store: &SearchEngineStore,
    bytes: &[u8],
) -> Result<PathBuf, ToolCapabilityError> {
    let temp = store
        .root
        .join(format!(".{}.install", uuid::Uuid::new_v4()));
    tokio::fs::write(&temp, bytes)
        .await
        .map_err(|e| ToolCapabilityError::new(e.to_string()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        tokio::fs::set_permissions(&temp, std::fs::Permissions::from_mode(0o755))
            .await
            .map_err(|e| ToolCapabilityError::new(e.to_string()))?;
    }
    Ok(temp)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn non_local_world_has_exact_text_without_store_consultation() {
        let dir = tempfile::tempdir().unwrap();
        let store = SearchEngineStore::new(dir.path());
        let world = ExecutionWorldIdentity::fresh();
        for engine in [SearchEngine::Fd, SearchEngine::Ripgrep] {
            let error = store.resolve(engine, &world).await.unwrap_err();
            assert_eq!(
                String::from_utf16_lossy(error.message().code_units()),
                format!(
                    "{} is not available on this platform: no certified {} engine for {} (non-local execution world).",
                    engine.name(),
                    engine.engine_name(),
                    world.as_str()
                )
            );
            assert_eq!(
                store
                    .verification_calls
                    .load(std::sync::atomic::Ordering::SeqCst),
                0
            );
        }
        // Calibrate the observation seam: a local lookup really consults verification.
        assert!(
            store
                .resolve(SearchEngine::Fd, &ExecutionWorldIdentity::local())
                .await
                .is_err()
        );
        assert_eq!(
            store
                .verification_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }
    #[tokio::test]
    async fn staging_cannot_publish_a_partial_binary_at_the_fixed_name() {
        let dir = tempfile::tempdir().unwrap();
        let store = SearchEngineStore::new(dir.path());
        let temp = stage_binary(&store, b"verified bytes").await.unwrap();
        assert!(!store.root.join(SearchEngine::Fd.executable()).exists());
        assert!(!store.root.join(SearchEngine::Ripgrep.executable()).exists());
        assert_eq!(std::fs::read(&temp).unwrap(), b"verified bytes");
        assert_ne!(temp, store.root.join(SearchEngine::Fd.executable()));
    }
}
