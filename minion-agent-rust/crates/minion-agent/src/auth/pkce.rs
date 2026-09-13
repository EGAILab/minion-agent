use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const VERIFIER_ENTROPY_BYTES: usize = 32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PkcePair {
    pub verifier: String,
    pub challenge: String,
}

#[derive(Debug, Error)]
#[error("cryptographic randomness unavailable: {0}")]
pub struct PkceError(String);

pub fn derive_pkce_challenge(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

pub fn generate_pkce() -> Result<PkcePair, PkceError> {
    let mut entropy = [0_u8; VERIFIER_ENTROPY_BYTES];
    getrandom::fill(&mut entropy).map_err(|error| PkceError(error.to_string()))?;
    let verifier = URL_SAFE_NO_PAD.encode(entropy);
    let challenge = derive_pkce_challenge(&verifier);
    Ok(PkcePair {
        verifier,
        challenge,
    })
}
