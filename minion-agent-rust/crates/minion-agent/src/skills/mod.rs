//! Harness skill discovery (WP-14.1). File-text and filesystem strings have
//! distinct certified domains; addressed paths are never canonicalized here.

mod discovery;
mod frontmatter;
mod ignore;
mod ignore_units;

pub use discovery::{
    SkillDiscoveryError, SourcedSkillError, load_skills, load_sourced_skills,
    load_sourced_skills_with,
};

use crate::execution::FsPath;
use serde::{Deserialize, Serialize};

/// Writable loaded skill. Filesystem-origin names retain their UTF-16 units.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Skill {
    pub name: FsPath,
    pub description: String,
    pub content: String,
    pub file_path: FsPath,
    pub disable_model_invocation: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillDiagnosticCode {
    FileInfoFailed,
    ListFailed,
    ReadFailed,
    ParseFailed,
    InvalidMetadata,
    InvalidPath,
    InvalidIgnorePattern,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillDiagnosticType {
    Warning,
}

/// All skill diagnostics are warnings; severity is not caller-selected.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkillDiagnostic {
    #[serde(rename = "type")]
    pub diagnostic_type: SkillDiagnosticType,
    pub code: SkillDiagnosticCode,
    pub message: FsPath,
    pub path: FsPath,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoadedSkills {
    pub skills: Vec<Skill>,
    pub diagnostics: Vec<SkillDiagnostic>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SkillSource<S> {
    pub path: FsPath,
    pub source: S,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SourcedSkill<S, T = Skill> {
    pub skill: T,
    pub source: S,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SourcedSkillDiagnostic<S> {
    pub diagnostic: SkillDiagnostic,
    pub source: S,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoadedSourcedSkills<S, T = Skill> {
    pub skills: Vec<SourcedSkill<S, T>>,
    pub diagnostics: Vec<SourcedSkillDiagnostic<S>>,
}
