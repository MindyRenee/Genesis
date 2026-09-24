//! Code analysis module — how Genesis understands code.
//!
//! This is the first piece of Genesis's ability to read, understand,
//! refactor, debug, and create code. Unlike LLM-based coding tools
//! that "understand" code by predicting tokens, Genesis understands
//! code **structurally** — by parsing it into an AST, extracting
//! semantic information, and building a dependency graph.
//!
//! # Why structural analysis, not statistical
//!
//! An LLM can tell you "this function probably does X" based on
//! patterns it's seen. But it can't tell you *for certain*:
//!
//! - What functions this function calls
//! - What functions call this function (callers)
//! - What types flow through this function
//! - Whether this function has side effects
//! - Whether this function can panic
//! - What the cyclomatic complexity is
//! - Whether there are unused imports
//! - Whether there are dead code paths
//!
//! All of these are **deterministic** properties that can be
//! extracted from the AST with zero ambiguity. Genesis uses
//! structural analysis for these, and reserves statistical methods
//! (the future language model) for the fuzzy parts: "what does the
//! user want?" and "how should I phrase this response?"
//!
//! # What this module does
//!
//! 1. **Parse** Rust source code into an AST (using `syn`)
//! 2. **Extract** semantic units: functions, structs, enums, traits,
//!    impls, modules
//! 3. **Build** a dependency graph: what calls what, what uses what
//! 4. **Compute** metrics: complexity, size, call depth
//! 5. **Identify** patterns: common structures, repeated code,
//!    potential refactoring targets
//! 6. **Serialize** all of this into a compact format that can be
//!    stored in LTM as episodes — so Genesis remembers code it's
//!    seen before and can find similar patterns

use std::collections::{HashMap, HashSet};

use syn::{
    ItemEnum, ItemFn, ItemImpl, ItemStruct, ItemTrait, ReturnType, Safety, Visibility, visit::Visit,
};

/// Extend a Vec<String> with deduplication — O(|existing| + |new|) per call
/// instead of O(|existing| * |new|). Uses a HashSet for O(1) lookup and
/// avoids cloning duplicate items via a `contains` check before `insert`.
fn extend_dedup(existing: &mut Vec<String>, new: Vec<String>) {
    let mut seen: HashSet<String> = existing.iter().cloned().collect();
    for r in new {
        if seen.contains(r.as_str()) {
            continue;
        }
        seen.insert(r.clone());
        existing.push(r);
    }
}

// ─────────────────────────────────────────────────────────────────
//  Semantic units — the atoms of code understanding
// ─────────────────────────────────────────────────────────────────

/// The kind of semantic unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnitKind {
    /// A free function or method.
    Function,
    /// A struct definition.
    Struct,
    /// An enum definition.
    Enum,
    /// A trait definition.
    Trait,
    /// An `impl` block (inherent or trait impl).
    Impl,
    /// A (sub)module declaration.
    Module,
    /// A type alias.
    TypeAlias,
    /// A compile-time constant.
    Constant,
    /// A static item.
    Static,
}

impl UnitKind {
    /// Returns the lowercase snake-case string used when serializing
    /// this kind for LTM storage and IPC.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Impl => "impl",
            Self::Module => "module",
            Self::TypeAlias => "type_alias",
            Self::Constant => "constant",
            Self::Static => "static",
        }
    }
}

/// A semantic unit — a single meaningful piece of code.
///
/// This is Genesis's atomic unit of code understanding. When it
/// reads a file, it doesn't see text — it sees a collection of
/// semantic units with relationships between them.
#[derive(Clone, Debug)]
pub struct SemanticUnit {
    /// What kind of unit this is (function, struct, enum, …).
    pub kind: UnitKind,
    /// The simple (unqualified) name of the unit.
    pub name: String,
    /// The fully-qualified name including the enclosing module path.
    pub qualified_name: String,
    /// First source line of the unit's span (1-based).
    pub line_start: usize,
    /// Last source line of the unit's span (1-based).
    pub line_end: usize,
    /// Functions called within this unit (by name).
    pub calls: Vec<String>,
    /// Types referenced within this unit (by name).
    pub references: Vec<String>,
    /// Parameters (for functions).
    pub params: Vec<String>,
    /// Return type (for functions).
    pub return_type: Option<String>,
    /// Whether this unit is public (visible outside its module).
    pub is_public: bool,
    /// Whether this function is async.
    pub is_async: bool,
    /// Whether this function is unsafe.
    pub is_unsafe: bool,
    /// Whether this function has a body (not just a declaration).
    pub has_body: bool,
    /// Number of statements in the body (complexity proxy).
    pub statement_count: usize,
    /// Number of branches (if/else/match arms) — cyclomatic complexity proxy.
    pub branch_count: usize,
    /// Number of loop constructs (for/while/loop).
    pub loop_count: usize,
    /// Doc comments associated with this unit.
    pub doc_comments: Vec<String>,
}

impl SemanticUnit {
    /// Generate a natural-language description of this unit for LTM
    /// storage. This is what gets SimHashed and stored as an episode.
    pub fn describe(&self) -> String {
        let mut parts: Vec<String> = Vec::with_capacity(10);

        // Visibility and modifiers
        if self.is_public {
            parts.push("pub".to_string());
        }
        if self.is_async {
            parts.push("async".to_string());
        }
        if self.is_unsafe {
            parts.push("unsafe".to_string());
        }

        parts.push(self.kind.as_str().to_string());
        parts.push(self.name.clone());

        if self.kind == UnitKind::Function {
            if !self.params.is_empty() {
                parts.push(format!("params: {}", self.params.join(", ")));
            }
            if let Some(rt) = &self.return_type {
                parts.push(format!("returns: {rt}"));
            }
            parts.push(format!("calls: {}", self.calls.join(", ")));
            parts.push(format!("branches: {}", self.branch_count));
            parts.push(format!("loops: {}", self.loop_count));
            parts.push(format!("statements: {}", self.statement_count));
        }

        if !self.references.is_empty() {
            parts.push(format!("refs: {}", self.references.join(", ")));
        }

        if !self.doc_comments.is_empty() {
            parts.push(format!("docs: {}", self.doc_comments.join(" ")));
        }

        parts.join(" | ")
    }

    /// A compact fingerprint for SimHash. This is what gets stored
    /// in LTM for associative retrieval.
    pub fn fingerprint(&self) -> String {
        let mut parts: Vec<String> =
            Vec::with_capacity(2 + self.calls.len() + self.references.len());
        parts.push(self.kind.as_str().to_string());
        parts.push(self.name.clone());
        for call in &self.calls {
            parts.push(format!("calls:{call}"));
        }
        for ref_ in &self.references {
            parts.push(format!("refs:{ref_}"));
        }
        parts.join(" ")
    }
}

// ─────────────────────────────────────────────────────────────────
//  Dependency graph — the relationships between units
// ─────────────────────────────────────────────────────────────────

/// A node in the dependency graph.
#[derive(Clone, Debug)]
pub struct GraphNode {
    /// Fully-qualified name of the unit this node represents.
    pub qualified_name: String,
    /// Kind of the underlying semantic unit.
    pub kind: UnitKind,
    /// How many other units this unit calls (out-degree).
    pub call_count: usize,
    /// How many other units call this unit (in-degree).
    pub caller_count: usize,
}

/// An edge in the dependency graph (A calls B).
#[derive(Clone, Debug)]
pub struct GraphEdge {
    /// Fully-qualified name of the calling unit (the source of the edge).
    pub from: String,
    /// Fully-qualified name of the called unit (the target of the edge).
    pub to: String,
}

/// The dependency graph for a file or module.
#[derive(Clone, Debug, Default)]
pub struct DependencyGraph {
    /// All nodes (units) in the graph.
    pub nodes: Vec<GraphNode>,
    /// All directed call edges (`from` calls `to`).
    pub edges: Vec<GraphEdge>,
}

impl DependencyGraph {
    /// Find all callers of a given function (reverse lookup).
    /// `name` is a qualified name (`Type::method` or `mod::func`) —
    /// top-level functions qualify as their bare name.
    pub fn callers_of(&self, name: &str) -> Vec<&str> {
        self.edges
            .iter()
            .filter(|e| e.to == name)
            .map(|e| e.from.as_str())
            .collect()
    }

    /// Find all callees of a given function (forward lookup).
    /// `name` is a qualified name, as in [`Self::callers_of`].
    pub fn callees_of(&self, name: &str) -> Vec<&str> {
        self.edges
            .iter()
            .filter(|e| e.from == name)
            .map(|e| e.to.as_str())
            .collect()
    }

    /// Find the most-connected functions (hub functions).
    /// These are the ones where refactoring has the highest impact.
    pub fn hubs(&self, limit: usize) -> Vec<&GraphNode> {
        let mut indexed: Vec<(usize, usize)> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (i, n.call_count + n.caller_count))
            .collect();
        indexed.sort_by_key(|&(_, count)| core::cmp::Reverse(count));
        indexed
            .iter()
            .take(limit)
            .map(|(i, _)| &self.nodes[*i])
            .collect()
    }

    /// Find leaf functions (no calls, few callers) — candidates
    /// for inlining or removal.
    pub fn leaves(&self) -> Vec<&GraphNode> {
        self.nodes
            .iter()
            .filter(|n| n.call_count == 0 && n.caller_count <= 1)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────
//  File analysis result
// ─────────────────────────────────────────────────────────────────

/// The complete analysis of a single source file.
#[derive(Clone, Debug, Default)]
pub struct FileAnalysis {
    /// Path of the analyzed source file.
    pub file_path: String,
    /// All semantic units extracted from the file.
    pub units: Vec<SemanticUnit>,
    /// Call dependency graph built from the units.
    pub graph: DependencyGraph,
    /// Total number of source lines in the file.
    pub total_lines: usize,
    /// Count of function units in the file.
    pub total_functions: usize,
    /// Count of struct definitions in the file.
    pub total_structs: usize,
    /// Count of enum definitions in the file.
    pub total_enums: usize,
    /// Count of trait definitions in the file.
    pub total_traits: usize,
    /// Count of `impl` blocks in the file.
    pub total_impls: usize,
    /// Highest cyclomatic-complexity proxy across all units.
    pub max_complexity: usize,
    /// Mean cyclomatic-complexity proxy across all units.
    pub avg_complexity: f64,
}

impl FileAnalysis {
    /// Generate a summary string for LTM storage.
    pub fn summary(&self) -> String {
        format!(
            "file: {} | lines: {} | functions: {} | structs: {} | enums: {} | traits: {} | impls: {} | max_complexity: {} | avg_complexity: {:.1}",
            self.file_path,
            self.total_lines,
            self.total_functions,
            self.total_structs,
            self.total_enums,
            self.total_traits,
            self.total_impls,
            self.max_complexity,
            self.avg_complexity,
        )
    }

    /// Generate episode texts for each semantic unit, ready for LTM.
    pub fn episodes(&self) -> Vec<String> {
        let mut episodes = Vec::new();

        // One episode for the file summary
        episodes.push(self.summary());

        // One episode per semantic unit
        for unit in &self.units {
            episodes.push(format!("file: {} | {}", self.file_path, unit.describe()));
        }

        episodes
    }
}

// ─────────────────────────────────────────────────────────────────
//  AST visitor — extracts semantic units from the AST
// ─────────────────────────────────────────────────────────────────

/// A visitor that walks the AST and collects semantic units.
struct UnitVisitor {
    units: Vec<SemanticUnit>,
    /// Lexical scope stack: module names, and the self-type or trait
    /// name while inside an impl/trait block. Methods inside
    /// `impl Foo` qualify as `mod::Foo::method`, so same-named
    /// methods on different types don't collide in the dependency
    /// graph.
    current_module: Vec<String>,
    /// Visibility of the trait currently being visited — trait items
    /// carry no visibility of their own, so they inherit the trait's.
    trait_is_pub: bool,
}

impl UnitVisitor {
    fn new() -> Self {
        Self {
            units: Vec::new(),
            current_module: Vec::new(),
            trait_is_pub: false,
        }
    }

    fn qualified_name(&self, name: &str) -> String {
        if self.current_module.is_empty() {
            name.to_string()
        } else {
            format!("{}::{name}", self.current_module.join("::"))
        }
    }

    /// The enclosing scope prefix of a qualified name — the portion
    /// before the last `::` segment. `mod::Foo::method` → `mod::Foo`;
    /// `function` → `""`.
    fn scope_of(qualified: &str) -> &str {
        match qualified.rfind("::") {
            Some(i) => &qualified[..i],
            None => "",
        }
    }

    /// Extract a function unit shared by free functions, impl methods,
    /// and trait items. `block` is `None` for declaration-only items
    /// (required trait methods, extern fns). `is_public` is resolved by
    /// the caller because trait items have no visibility field.
    fn visit_fn_like(
        &mut self,
        ident: &syn::Ident,
        is_public: bool,
        sig: &syn::Signature,
        attrs: &[syn::Attribute],
        block: Option<&syn::Block>,
        line_end: usize,
    ) {
        let name = ident.to_string();
        let qualified = self.qualified_name(&name);

        let params: Vec<String> = sig
            .inputs
            .iter()
            .filter_map(|arg| {
                if let syn::FnArg::Typed(p) = arg {
                    Some(type_to_string(&p.ty))
                } else {
                    None
                }
            })
            .collect();
        let return_type = return_type_string(&sig.output);
        let (statement_count, branch_count, loop_count) = block
            .map(|b| self.count_statements(&b.stmts))
            .unwrap_or_default();
        let calls = block
            .map(|b| self.extract_calls(&b.stmts))
            .unwrap_or_default();

        let mut references = Vec::new();
        for arg in &sig.inputs {
            if let syn::FnArg::Typed(p) = arg {
                extend_dedup(&mut references, self.extract_references(&p.ty));
            }
        }
        if let ReturnType::Type(_, ty) = &sig.output {
            extend_dedup(&mut references, self.extract_references(ty));
        }

        let doc_comments = self.extract_doc_comments(attrs);
        let line_start = ident.span().start().line;

        self.units.push(SemanticUnit {
            kind: UnitKind::Function,
            name,
            qualified_name: qualified,
            line_start,
            line_end,
            calls,
            references,
            params,
            return_type,
            is_public,
            is_async: sig.asyncness.is_some(),
            is_unsafe: is_unsafe(&sig.safety),
            has_body: block.is_some(),
            statement_count,
            branch_count,
            loop_count,
            doc_comments,
        });
    }

    fn extract_doc_comments(&self, attrs: &[syn::Attribute]) -> Vec<String> {
        let mut docs = Vec::new();
        for attr in attrs {
            if attr.path().is_ident("doc")
                && let syn::Meta::NameValue(nv) = &attr.meta
                && let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) = &nv.value
            {
                docs.push(s.value().trim().to_string());
            }
        }
        docs
    }

    fn count_statements(&self, stmts: &[syn::Stmt]) -> (usize, usize, usize) {
        let mut statement_count = 0;
        let mut branch_count = 0;
        let mut loop_count = 0;

        for stmt in stmts {
            self.count_stmt(
                stmt,
                &mut statement_count,
                &mut branch_count,
                &mut loop_count,
            );
        }

        (statement_count, branch_count, loop_count)
    }

    fn count_stmt(&self, stmt: &syn::Stmt, sc: &mut usize, bc: &mut usize, lc: &mut usize) {
        *sc += 1;
        match stmt {
            syn::Stmt::Local(l) => {
                if let Some(init) = &l.init {
                    self.count_expr(&init.expr, sc, bc, lc);
                }
            }
            syn::Stmt::Expr(e, _) => {
                self.count_expr(e, sc, bc, lc);
            }
            syn::Stmt::Item(_) => {}
            syn::Stmt::Macro(_) => {}
        }
    }

    fn count_expr(&self, e: &syn::Expr, sc: &mut usize, bc: &mut usize, lc: &mut usize) {
        match e {
            syn::Expr::If(i) => {
                *bc += 1;
                // Recurse into then branch
                for stmt in &i.then_branch.stmts {
                    self.count_stmt(stmt, sc, bc, lc);
                }
                // Recurse into else branch (may contain another if)
                if let Some((_, else_expr)) = &i.else_branch {
                    self.count_expr(else_expr, sc, bc, lc);
                }
            }
            syn::Expr::Match(m) => {
                *bc += m.arms.len();
                // Recurse into arm bodies
                for arm in &m.arms {
                    self.count_expr(&arm.body, sc, bc, lc);
                }
            }
            syn::Expr::ForLoop(f) => {
                *lc += 1;
                for stmt in &f.body.stmts {
                    self.count_stmt(stmt, sc, bc, lc);
                }
            }
            syn::Expr::While(w) => {
                *lc += 1;
                self.count_expr(&w.cond, sc, bc, lc);
                for stmt in &w.body.stmts {
                    self.count_stmt(stmt, sc, bc, lc);
                }
            }
            syn::Expr::Loop(l) => {
                *lc += 1;
                for stmt in &l.body.stmts {
                    self.count_stmt(stmt, sc, bc, lc);
                }
            }
            syn::Expr::Block(b) => {
                for stmt in &b.block.stmts {
                    self.count_stmt(stmt, sc, bc, lc);
                }
            }
            syn::Expr::Call(c) => {
                for arg in &c.args {
                    self.count_expr(arg, sc, bc, lc);
                }
            }
            syn::Expr::Binary(b) => {
                self.count_expr(&b.left, sc, bc, lc);
                self.count_expr(&b.right, sc, bc, lc);
            }
            syn::Expr::Paren(p) => {
                self.count_expr(&p.expr, sc, bc, lc);
            }
            _ => {}
        }
    }

    fn extract_calls(&self, stmts: &[syn::Stmt]) -> Vec<String> {
        let mut calls = Vec::new();
        for stmt in stmts {
            self.collect_calls_stmt(stmt, &mut calls);
        }
        calls
    }

    fn collect_calls_stmt(&self, stmt: &syn::Stmt, calls: &mut Vec<String>) {
        match stmt {
            syn::Stmt::Local(l) => {
                if let Some(init) = &l.init {
                    self.collect_calls_expr(&init.expr, calls);
                }
            }
            syn::Stmt::Expr(e, _) => {
                self.collect_calls_expr(e, calls);
            }
            _ => {}
        }
    }

    fn collect_calls_expr(&self, e: &syn::Expr, calls: &mut Vec<String>) {
        match e {
            syn::Expr::Call(c) => {
                if let syn::Expr::Path(p) = &*c.func {
                    let name = p
                        .path
                        .segments
                        .last()
                        .map(|s| s.ident.to_string())
                        .unwrap_or_default();
                    if !name.is_empty() && !calls.contains(&name) {
                        calls.push(name);
                    }
                }
                for arg in &c.args {
                    self.collect_calls_expr(arg, calls);
                }
            }
            syn::Expr::MethodCall(m) => {
                let name = m.method.to_string();
                if !calls.contains(&name) {
                    calls.push(name);
                }
                // Recurse into the receiver (for chained calls like a.b().c())
                self.collect_calls_expr(&m.receiver, calls);
                for arg in &m.args {
                    self.collect_calls_expr(arg, calls);
                }
            }
            syn::Expr::If(i) => {
                self.collect_calls_expr(&i.cond, calls);
                for stmt in &i.then_branch.stmts {
                    self.collect_calls_stmt(stmt, calls);
                }
                if let Some((_, e)) = &i.else_branch {
                    self.collect_calls_expr(e, calls);
                }
            }
            syn::Expr::Match(m) => {
                self.collect_calls_expr(&m.expr, calls);
                for arm in &m.arms {
                    self.collect_calls_expr(&arm.body, calls);
                }
            }
            syn::Expr::ForLoop(f) => {
                for stmt in &f.body.stmts {
                    self.collect_calls_stmt(stmt, calls);
                }
            }
            syn::Expr::While(w) => {
                self.collect_calls_expr(&w.cond, calls);
                for stmt in &w.body.stmts {
                    self.collect_calls_stmt(stmt, calls);
                }
            }
            syn::Expr::Loop(l) => {
                for stmt in &l.body.stmts {
                    self.collect_calls_stmt(stmt, calls);
                }
            }
            syn::Expr::Block(b) => {
                for stmt in &b.block.stmts {
                    self.collect_calls_stmt(stmt, calls);
                }
            }
            syn::Expr::Binary(b) => {
                self.collect_calls_expr(&b.left, calls);
                self.collect_calls_expr(&b.right, calls);
            }
            syn::Expr::Paren(p) => {
                self.collect_calls_expr(&p.expr, calls);
            }
            syn::Expr::Tuple(t) => {
                for e in &t.elems {
                    self.collect_calls_expr(e, calls);
                }
            }
            syn::Expr::Assign(a) => {
                self.collect_calls_expr(&a.left, calls);
                self.collect_calls_expr(&a.right, calls);
            }
            _ => {}
        }
    }

    fn extract_references(&self, ty: &syn::Type) -> Vec<String> {
        let mut refs = Vec::new();
        self.collect_type_refs(ty, &mut refs);
        refs
    }

    fn collect_type_refs(&self, ty: &syn::Type, refs: &mut Vec<String>) {
        match ty {
            syn::Type::Path(p) => {
                let name = p
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                if !name.is_empty() && !refs.contains(&name) {
                    refs.push(name);
                }
                for seg in &p.path.segments {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        for arg in &args.args {
                            if let syn::GenericArgument::Type(t) = arg {
                                self.collect_type_refs(t, refs);
                            }
                        }
                    }
                }
            }
            syn::Type::Reference(r) => {
                self.collect_type_refs(&r.elem, refs);
            }
            syn::Type::Array(a) => {
                self.collect_type_refs(&a.elem, refs);
            }
            syn::Type::Slice(s) => {
                self.collect_type_refs(&s.elem, refs);
            }
            syn::Type::Tuple(t) => {
                for e in &t.elems {
                    self.collect_type_refs(e, refs);
                }
            }
            _ => {}
        }
    }
}

/// Helper: check if a visibility is public.
fn is_pub(vis: &Visibility) -> bool {
    matches!(vis, Visibility::Public(_))
}

/// Helper: check if a safety marker is unsafe.
fn is_unsafe(safety: &Safety) -> bool {
    matches!(safety, Safety::Unsafe(_))
}

/// Helper: format a return type as a string.
fn return_type_string(rt: &ReturnType) -> Option<String> {
    match rt {
        ReturnType::Default => None,
        ReturnType::Type(_, ty) => Some(type_to_string(ty)),
    }
}

/// Helper: format a type as a string (using ToTokens via quote).
fn type_to_string(ty: &syn::Type) -> String {
    let tokens = quote::quote!(#ty);
    tokens.to_string()
}

impl<'ast> Visit<'ast> for UnitVisitor {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        let name = node.sig.ident.to_string();
        let qualified = self.qualified_name(&name);

        let is_public = is_pub(&node.vis);
        let is_async = node.sig.asyncness.is_some();
        let is_unsafe_fn = is_unsafe(&node.sig.safety);
        // An ItemFn always has a body block — has_body distinguishes
        // "has a body" from "declaration only" (trait/extern items).
        let has_body = true;

        let params: Vec<String> = node
            .sig
            .inputs
            .iter()
            .filter_map(|arg| {
                if let syn::FnArg::Typed(p) = arg {
                    Some(type_to_string(&p.ty))
                } else {
                    None // self
                }
            })
            .collect();

        let return_type = return_type_string(&node.sig.output);

        let (statement_count, branch_count, loop_count) = self.count_statements(&node.block.stmts);
        let calls = self.extract_calls(&node.block.stmts);

        let mut references = Vec::new();
        for arg in &node.sig.inputs {
            if let syn::FnArg::Typed(p) = arg {
                extend_dedup(&mut references, self.extract_references(&p.ty));
            }
        }
        if let ReturnType::Type(_, ty) = &node.sig.output {
            extend_dedup(&mut references, self.extract_references(ty));
        }

        let doc_comments = self.extract_doc_comments(&node.attrs);

        // Line numbers — use the ident span as a proxy
        let line_start = node.sig.ident.span().start().line;
        let line_end = node.block.brace_token.span.close().start().line;

        self.units.push(SemanticUnit {
            kind: UnitKind::Function,
            name,
            qualified_name: qualified,
            line_start,
            line_end,
            calls,
            references,
            params,
            return_type,
            is_public,
            is_async,
            is_unsafe: is_unsafe_fn,
            has_body,
            statement_count,
            branch_count,
            loop_count,
            doc_comments,
        });

        // Push the function name as scope so nested items (a `fn inner`
        // inside this function) qualify as `outer::inner` rather than
        // colliding with same-named items nested in other functions.
        self.current_module.push(node.sig.ident.to_string());
        syn::visit::visit_item_fn(self, node);
        self.current_module.pop();
    }

    fn visit_item_struct(&mut self, node: &'ast ItemStruct) {
        let name = node.ident.to_string();
        let qualified = self.qualified_name(&name);

        let mut references = Vec::new();
        for field in node.fields.iter() {
            extend_dedup(&mut references, self.extract_references(&field.ty));
        }

        let doc_comments = self.extract_doc_comments(&node.attrs);
        let line_start = node.ident.span().start().line;
        let line_end = line_start + 1; // Approximate

        self.units.push(SemanticUnit {
            kind: UnitKind::Struct,
            name,
            qualified_name: qualified,
            line_start,
            line_end,
            calls: Vec::new(),
            references,
            params: Vec::new(),
            return_type: None,
            is_public: is_pub(&node.vis),
            is_async: false,
            is_unsafe: false,
            has_body: true,
            statement_count: node.fields.len(),
            branch_count: 0,
            loop_count: 0,
            doc_comments,
        });

        syn::visit::visit_item_struct(self, node);
    }

    fn visit_item_enum(&mut self, node: &'ast ItemEnum) {
        let name = node.ident.to_string();
        let qualified = self.qualified_name(&name);

        let doc_comments = self.extract_doc_comments(&node.attrs);
        let line_start = node.ident.span().start().line;
        let line_end = line_start + 1; // Approximate

        self.units.push(SemanticUnit {
            kind: UnitKind::Enum,
            name,
            qualified_name: qualified,
            line_start,
            line_end,
            calls: Vec::new(),
            references: Vec::new(),
            params: Vec::new(),
            return_type: None,
            is_public: is_pub(&node.vis),
            is_async: false,
            is_unsafe: false,
            has_body: true,
            statement_count: node.variants.len(),
            branch_count: node.variants.len(),
            loop_count: 0,
            doc_comments,
        });

        syn::visit::visit_item_enum(self, node);
    }

    fn visit_item_trait(&mut self, node: &'ast ItemTrait) {
        let name = node.ident.to_string();
        let qualified = self.qualified_name(&name);

        let doc_comments = self.extract_doc_comments(&node.attrs);
        let line_start = node.ident.span().start().line;
        let line_end = node.brace_token.span.close().start().line;

        self.units.push(SemanticUnit {
            kind: UnitKind::Trait,
            name,
            qualified_name: qualified,
            line_start,
            line_end,
            calls: Vec::new(),
            references: Vec::new(),
            params: Vec::new(),
            return_type: None,
            is_public: is_pub(&node.vis),
            is_async: false,
            is_unsafe: false,
            has_body: !node.items.is_empty(),
            statement_count: node.items.len(),
            branch_count: 0,
            loop_count: 0,
            doc_comments,
        });

        // Scope trait items under the trait name so a required method
        // `Display::fmt` and a default body `Display::detailed` qualify
        // distinctly from same-named methods on other traits.
        // Trait items inherit the trait's visibility (they carry none).
        self.trait_is_pub = is_pub(&node.vis);
        self.current_module.push(node.ident.to_string());
        syn::visit::visit_item_trait(self, node);
        self.current_module.pop();
        self.trait_is_pub = false;
    }

    fn visit_item_impl(&mut self, node: &'ast ItemImpl) {
        // The impl unit's display name prefers the trait (so
        // `impl Display for User` shows as `impl_Display`), but the
        // lexical scope for its methods is the *self type* — methods
        // belong to `User::display`, not `Display::display`.
        let self_name = if let syn::Type::Path(p) = &*node.self_ty {
            p.path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        } else {
            "unknown".to_string()
        };
        let name = node
            .trait_
            .as_ref()
            .map(|(p, _)| {
                p.segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default()
            })
            .unwrap_or_else(|| self_name.clone());

        let qualified = self.qualified_name(&format!("impl_{name}"));
        let line_start = if let syn::Type::Path(p) = &*node.self_ty {
            p.path
                .segments
                .last()
                .map(|s| s.ident.span().start().line)
                .unwrap_or(0)
        } else {
            0
        };
        let line_end = node.brace_token.span.close().start().line;

        self.units.push(SemanticUnit {
            kind: UnitKind::Impl,
            name: format!("impl_{name}"),
            qualified_name: qualified,
            line_start,
            line_end,
            calls: Vec::new(),
            references: vec![name],
            params: Vec::new(),
            return_type: None,
            is_public: false,
            is_async: false,
            is_unsafe: node.unsafety.is_some(),
            has_body: !node.items.is_empty(),
            statement_count: node.items.len(),
            branch_count: 0,
            loop_count: 0,
            doc_comments: Vec::new(),
        });

        // Scope impl items under the self type: `Foo::new` and
        // `Bar::new` become distinct qualified names instead of
        // colliding in the dependency graph.
        self.current_module.push(self_name);
        syn::visit::visit_item_impl(self, node);
        self.current_module.pop();
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        let name = node.sig.ident.to_string();
        let qualified = self.qualified_name(&name);

        let is_public = is_pub(&node.vis);
        let is_async = node.sig.asyncness.is_some();
        let is_unsafe_fn = is_unsafe(&node.sig.safety);
        let has_body = true; // an impl method always has a body block

        let params: Vec<String> = node
            .sig
            .inputs
            .iter()
            .filter_map(|arg| {
                if let syn::FnArg::Typed(p) = arg {
                    Some(type_to_string(&p.ty))
                } else {
                    None
                }
            })
            .collect();

        let return_type = return_type_string(&node.sig.output);

        let (statement_count, branch_count, loop_count) = self.count_statements(&node.block.stmts);
        let calls = self.extract_calls(&node.block.stmts);

        let mut references = Vec::new();
        for arg in &node.sig.inputs {
            if let syn::FnArg::Typed(p) = arg {
                extend_dedup(&mut references, self.extract_references(&p.ty));
            }
        }
        if let ReturnType::Type(_, ty) = &node.sig.output {
            extend_dedup(&mut references, self.extract_references(ty));
        }

        let doc_comments = self.extract_doc_comments(&node.attrs);
        let line_start = node.sig.ident.span().start().line;
        let line_end = node.block.brace_token.span.close().start().line;

        self.units.push(SemanticUnit {
            kind: UnitKind::Function,
            name,
            qualified_name: qualified,
            line_start,
            line_end,
            calls,
            references,
            params,
            return_type,
            is_public,
            is_async,
            is_unsafe: is_unsafe_fn,
            has_body,
            statement_count,
            branch_count,
            loop_count,
            doc_comments,
        });

        syn::visit::visit_impl_item_fn(self, node);
    }

    fn visit_trait_item_fn(&mut self, node: &'ast syn::TraitItemFn) {
        let line_end = node
            .default
            .as_ref()
            .map(|b| b.brace_token.span.close().start().line)
            .unwrap_or_else(|| node.sig.ident.span().start().line);
        self.visit_fn_like(
            &node.sig.ident,
            // Trait items have no visibility modifier — they are public
            // to implementors iff the trait itself is public.
            self.trait_is_pub,
            &node.sig,
            &node.attrs,
            node.default.as_ref(),
            line_end,
        );
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        // Emit a Module unit so `mod` declarations are first-class
        // semantic units, then recurse so items inside `mod foo { }`
        // qualify under `foo`.
        let name = node.ident.to_string();
        let qualified = self.qualified_name(&name);
        let line_start = node.ident.span().start().line;
        let line_end = node
            .content
            .as_ref()
            .map(|(brace, _)| brace.span.close().start().line)
            .unwrap_or(line_start);

        self.units.push(SemanticUnit {
            kind: UnitKind::Module,
            name,
            qualified_name: qualified,
            line_start,
            line_end,
            calls: Vec::new(),
            references: Vec::new(),
            params: Vec::new(),
            return_type: None,
            is_public: is_pub(&node.vis),
            is_async: false,
            is_unsafe: node.unsafety.is_some(),
            has_body: node.content.is_some(),
            statement_count: node.content.as_ref().map(|(_, items)| items.len()).unwrap_or(0),
            branch_count: 0,
            loop_count: 0,
            doc_comments: self.extract_doc_comments(&node.attrs),
        });

        self.current_module.push(node.ident.to_string());
        syn::visit::visit_item_mod(self, node);
        self.current_module.pop();
    }

    fn visit_item_type(&mut self, node: &'ast syn::ItemType) {
        let name = node.ident.to_string();
        let qualified = self.qualified_name(&name);
        let line = node.ident.span().start().line;
        let mut references = Vec::new();
        extend_dedup(&mut references, self.extract_references(&node.ty));
        self.units.push(SemanticUnit {
            kind: UnitKind::TypeAlias,
            name,
            qualified_name: qualified,
            line_start: line,
            line_end: line,
            calls: Vec::new(),
            references,
            params: Vec::new(),
            return_type: None,
            is_public: is_pub(&node.vis),
            is_async: false,
            is_unsafe: false,
            has_body: true,
            statement_count: 1,
            branch_count: 0,
            loop_count: 0,
            doc_comments: self.extract_doc_comments(&node.attrs),
        });
    }

    fn visit_item_const(&mut self, node: &'ast syn::ItemConst) {
        let name = node.ident.to_string();
        let qualified = self.qualified_name(&name);
        let line = node.ident.span().start().line;
        let mut references = Vec::new();
        extend_dedup(&mut references, self.extract_references(&node.ty));
        self.units.push(SemanticUnit {
            kind: UnitKind::Constant,
            name,
            qualified_name: qualified,
            line_start: line,
            line_end: line,
            calls: Vec::new(),
            references,
            params: Vec::new(),
            return_type: Some(type_to_string(&node.ty)),
            is_public: is_pub(&node.vis),
            is_async: false,
            is_unsafe: false,
            has_body: true,
            statement_count: 1,
            branch_count: 0,
            loop_count: 0,
            doc_comments: self.extract_doc_comments(&node.attrs),
        });
    }

    fn visit_item_static(&mut self, node: &'ast syn::ItemStatic) {
        let name = node.ident.to_string();
        let qualified = self.qualified_name(&name);
        let line = node.ident.span().start().line;
        let mut references = Vec::new();
        extend_dedup(&mut references, self.extract_references(&node.ty));
        self.units.push(SemanticUnit {
            kind: UnitKind::Static,
            name,
            qualified_name: qualified,
            line_start: line,
            line_end: line,
            calls: Vec::new(),
            references,
            params: Vec::new(),
            return_type: Some(type_to_string(&node.ty)),
            is_public: is_pub(&node.vis),
            is_async: false,
            is_unsafe: false,
            has_body: true,
            statement_count: 1,
            branch_count: 0,
            loop_count: 0,
            doc_comments: self.extract_doc_comments(&node.attrs),
        });
    }
}

// ─────────────────────────────────────────────────────────────────
//  Analysis functions
// ─────────────────────────────────────────────────────────────────

/// Analyze a single Rust source file.
///
/// Parses the file, extracts all semantic units, builds a dependency
/// graph, and computes metrics.
pub fn analyze_file(source: &str, file_path: &str) -> Result<FileAnalysis, String> {
    let file = syn::parse_file(source).map_err(|e| format!("parse error: {e}"))?;

    let mut visitor = UnitVisitor::new();
    visitor.visit_file(&file);

    let total_lines = source.lines().count();

    // Single-pass count of all unit kinds + collect function complexities
    let mut total_functions = 0usize;
    let mut total_structs = 0usize;
    let mut total_enums = 0usize;
    let mut total_traits = 0usize;
    let mut total_impls = 0usize;
    let mut complexities: Vec<usize> = Vec::new();

    for u in &visitor.units {
        match u.kind {
            UnitKind::Function => {
                total_functions += 1;
                complexities.push(u.branch_count + u.loop_count + 1);
            }
            UnitKind::Struct => total_structs += 1,
            UnitKind::Enum => total_enums += 1,
            UnitKind::Trait => total_traits += 1,
            UnitKind::Impl => total_impls += 1,
            _ => {}
        }
    }

    let max_complexity = complexities.iter().copied().max().unwrap_or(0);
    let avg_complexity = if complexities.is_empty() {
        0.0
    } else {
        complexities.iter().copied().sum::<usize>() as f64 / complexities.len() as f64
    };

    // Build the dependency graph
    let graph = build_dependency_graph(&visitor.units);

    Ok(FileAnalysis {
        file_path: file_path.to_string(),
        units: visitor.units,
        graph,
        total_lines,
        total_functions,
        total_structs,
        total_enums,
        total_traits,
        total_impls,
        max_complexity,
        avg_complexity,
    })
}

/// Build a dependency graph from semantic units.
///
/// Nodes are keyed by **qualified** name (`mod::Type::method`) so
/// same-named methods on different types stay distinct. Calls are
/// recorded as simple names (the AST doesn't carry the callee's
/// resolved path), so each call is resolved to a qualified target:
///
/// 1. Try scope-relative matches outward from the caller's scope —
///    a method `a::Foo::m` calling `helper` first tries `a::Foo::helper`,
///    then `a::helper`, then `helper`.
/// 2. If no scoped candidate matches but exactly one unit has that
///    simple name, link it (unambiguous top-level call).
/// 3. Otherwise the call is ambiguous or external — no edge.
fn build_dependency_graph(units: &[SemanticUnit]) -> DependencyGraph {
    let mut node_map: HashMap<String, GraphNode> = HashMap::with_capacity(units.len());
    // Simple name → all qualified names of function units with it.
    let mut by_simple: HashMap<&str, Vec<&str>> = HashMap::new();

    // Create nodes for all functions
    for unit in units {
        if unit.kind == UnitKind::Function {
            node_map.insert(
                unit.qualified_name.clone(),
                GraphNode {
                    qualified_name: unit.qualified_name.clone(),
                    kind: unit.kind,
                    call_count: 0, // filled in after edge resolution
                    caller_count: 0,
                },
            );
            by_simple
                .entry(unit.name.as_str())
                .or_default()
                .push(unit.qualified_name.as_str());
        }
    }

    // Resolve a simple call name to a qualified unit name.
    // `caller_qname` supplies the caller's scope chain.
    let resolve_call = |caller_qname: &str, call: &str| -> Option<&str> {
        let candidates = by_simple.get(call)?;
        if candidates.len() == 1 {
            return Some(candidates[0]);
        }
        // Multiple units share the simple name — walk the caller's
        // scope chain outward (innermost first) looking for
        // `scope::call` among the candidates.
        let mut scope = UnitVisitor::scope_of(caller_qname);
        loop {
            let scoped = if scope.is_empty() {
                call.to_string()
            } else {
                format!("{scope}::{call}")
            };
            if let Some(&hit) = candidates.iter().find(|&&q| q == scoped) {
                return Some(hit);
            }
            if scope.is_empty() {
                return None;
            }
            scope = UnitVisitor::scope_of(scope);
        }
    };

    // Create edges from resolved calls
    let mut edges = Vec::with_capacity(units.len() * 3);
    for unit in units {
        if unit.kind != UnitKind::Function {
            continue;
        }
        let mut resolved = 0usize;
        for call in &unit.calls {
            if let Some(target) = resolve_call(&unit.qualified_name, call) {
                // No self-edges — a recursive call adds no dependency.
                if target != unit.qualified_name {
                    edges.push(GraphEdge {
                        from: unit.qualified_name.clone(),
                        to: target.to_string(),
                    });
                    resolved += 1;
                }
            }
        }
        if let Some(node) = node_map.get_mut(&unit.qualified_name) {
            node.call_count = resolved;
        }
    }

    // Count callers
    for edge in &edges {
        if let Some(node) = node_map.get_mut(&edge.to) {
            node.caller_count += 1;
        }
    }

    let nodes: Vec<GraphNode> = node_map.into_values().collect();

    DependencyGraph { nodes, edges }
}

/// Analyze a file from disk.
pub fn analyze_file_path(path: &std::path::Path) -> Result<FileAnalysis, String> {
    let source = std::fs::read_to_string(path).map_err(|e| format!("failed to read file: {e}"))?;
    analyze_file(&source, path.to_str().unwrap_or("?"))
}

/// Find potential refactoring targets in a file analysis.
///
/// Returns a list of (unit_name, reason) pairs for units that could
/// benefit from refactoring.
pub fn find_refactoring_targets(analysis: &FileAnalysis) -> Vec<(String, String)> {
    let mut targets = Vec::with_capacity(analysis.units.len());

    for unit in &analysis.units {
        if unit.kind != UnitKind::Function {
            continue;
        }

        let complexity = unit.branch_count + unit.loop_count + 1;

        // High complexity
        if complexity > 10 {
            targets.push((
                unit.name.clone(),
                format!("high complexity ({complexity}): consider breaking into smaller functions"),
            ));
        }

        // Too many parameters
        if unit.params.len() > 5 {
            targets.push((
                unit.name.clone(),
                format!(
                    "too many parameters ({}): consider using a parameter struct",
                    unit.params.len()
                ),
            ));
        }

        // Too many calls (tight coupling)
        if unit.calls.len() > 15 {
            targets.push((
                unit.name.clone(),
                format!(
                    "too many calls ({}): tightly coupled, consider reducing dependencies",
                    unit.calls.len()
                ),
            ));
        }

        // Long function (many statements)
        if unit.statement_count > 50 {
            targets.push((
                unit.name.clone(),
                format!(
                    "long function ({} statements): consider splitting",
                    unit.statement_count
                ),
            ));
        }
    }

    // Leaf functions that could be inlined
    let leaves = analysis.graph.leaves();
    for leaf in leaves.iter().take(5) {
        if leaf.caller_count == 0 {
            targets.push((
                leaf.qualified_name.clone(),
                "dead code: no callers".to_string(),
            ));
        }
    }

    targets
}
