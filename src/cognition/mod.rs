//! Cognition module — higher-level cognitive functions.
//!
//! This is where Genesis's ability to understand and manipulate code
//! lives. Unlike the subcognitive (which is about feeling and memory),
//! the cognition module is about **understanding** — parsing,
//! analyzing, and reasoning about code structure.
//!
//! # Design philosophy
//!
//! The cognition module uses **neurosymbolic architecture**: deterministic
//! structural analysis (AST parsing, dependency graphs) combined with
//! associative memory (SimHash over LTM). This is fundamentally
//! different from LLM-based coding tools:
//!
//! - **Structural analysis** gives exact, deterministic answers:
//!   "what does this function call?", "who calls this?", "what's the
//!   complexity?"
//! - **Associative memory** gives fuzzy, similarity-based answers:
//!   "have I seen a bug like this before?", "is there a similar pattern
//!   in my memory?"
//!
//! - **Self-modeling** gives a structured first-person summary of the
//!   system's own state — a prerequisite for any functional account of
//!   selfhood.
//!
//! - **Intention selection** scores active goals against the self-model
//!   and decides which cognitive zone to enter next.
//!
//! Together, they give Genesis the ability to understand code the way
//! a human developer does — with both precision and intuition.

pub mod code_analysis;
pub mod intention;
pub mod self_model;

pub use code_analysis::{
    DependencyGraph, FileAnalysis, GraphEdge, GraphNode, SemanticUnit, UnitKind, analyze_file,
    analyze_file_path, find_refactoring_targets,
};
pub use intention::{GoalKind, Intention, IntentionManager};
pub use self_model::SelfModel;
