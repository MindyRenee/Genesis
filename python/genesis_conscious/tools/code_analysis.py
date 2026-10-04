"""Code analysis — structural understanding of Python source.

The ``CodeLearner`` builds the *semantic graph* — concepts and edges
for what exists. This module answers the questions a reasoning system
actually asks about code: what calls what, how complex is each body,
what does a change to X touch, where are the risky tangles.

It is deliberately read-only and dependency-free: ``ast`` does the
parsing, and everything returned is plain data. The cognition layer
composes whatever words it needs from the structures; nothing here
produces prose.

# What it computes

- **Scoped symbol table** — module, classes, functions, methods, and
  nested definitions with qualified names and line spans.
- **Call edges** — attributed to the innermost *named* callable, so a
  helper nested inside ``foo`` attributes its calls to ``foo`` (the
  concept that actually exists in the graph).
- **Cyclomatic complexity** per callable — decision points + 1.
- **Import dependencies** — resolved to dotted module paths.
- **Impact analysis** — given a symbol, which other symbols in the
  file depend on it transitively (the change-propagation surface).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

__all__ = [
    "CallEdge",
    "FileAnalysis",
    "SymbolInfo",
    "analyze_python_source",
    "analyze_python_tree",
    "impact_of",
    "summarize",
]

# Calls to these builtins carry no structural signal — they fire
# constantly and would drown the real call graph.
_BUILTIN_CALLS = frozenset({
    "print", "len", "range", "str", "int", "float", "bool", "list",
    "dict", "set", "tuple", "type", "isinstance", "issubclass",
    "enumerate", "zip", "map", "filter", "sorted", "min", "max",
    "sum", "abs", "round", "repr", "format", "open", "getattr",
    "setattr", "hasattr", "super", "any", "all", "next", "iter",
    "ord", "chr", "hex", "id", "hash", "callable", "vars", "dir",
})


@dataclass(slots=True)
class SymbolInfo:
    """One named definition in a Python file."""

    name: str                # short name ("parse_file")
    qualified: str           # dotted path ("module.Parser.parse_file")
    kind: str                # "module" | "class" | "function" | "method"
    line: int
    end_line: int
    params: list[str] = field(default_factory=list)
    returns: str = ""        # return annotation, unparsed
    decorators: list[str] = field(default_factory=list)
    docstring: str = ""
    complexity: int = 1      # cyclomatic: decision points + 1
    calls: list[str] = field(default_factory=list)   # dotted callee names
    raises: list[str] = field(default_factory=list)  # exception types raised

    @property
    def span(self) -> int:
        """Body length in lines."""
        return max(1, self.end_line - self.line + 1)


@dataclass(slots=True)
class CallEdge:
    """A resolved call relationship between two named symbols."""

    caller: str   # qualified name of the enclosing callable
    callee: str   # dotted callee name, as written ("self.run", "ast.parse")


@dataclass(slots=True)
class FileAnalysis:
    """Complete structural analysis of one Python file."""

    path: str
    language: str = "python"
    docstring: str = ""
    lines: int = 0
    symbols: list[SymbolInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    call_edges: list[CallEdge] = field(default_factory=list)

    @property
    def functions(self) -> list[SymbolInfo]:
        return [s for s in self.symbols if s.kind in ("function", "method")]

    @property
    def classes(self) -> list[SymbolInfo]:
        return [s for s in self.symbols if s.kind == "class"]

    @property
    def max_complexity(self) -> int:
        return max((s.complexity for s in self.functions), default=0)

    @property
    def avg_complexity(self) -> float:
        fns = self.functions
        if not fns:
            return 0.0
        return sum(s.complexity for s in fns) / len(fns)


class _ScopeVisitor(ast.NodeVisitor):
    """Walk a module AST maintaining lexical scope.

    Attributes every call to the innermost enclosing *named* callable —
    a function defined at module level or a method of a class. Nested
    functions inside a callable attribute their calls to that callable,
    matching the concept granularity the concept network uses.
    """

    def __init__(self, module_name: str) -> None:
        self.module = module_name
        self.symbols: list[SymbolInfo] = []
        self.call_edges: list[CallEdge] = []
        self.imports: list[str] = []
        # Stack of qualified-name prefixes (module, class, function…)
        self._scope: list[str] = [module_name]
        # The callable that currently owns calls, or None at module
        # level / inside class bodies between methods.
        self._callable: SymbolInfo | None = None
        # Stack of enclosing classes — distinguishes methods from
        # plain functions for qualified naming and kind tagging.
        self._class_stack: list[str] = []

    # ── scope management ─────────────────────────────────────────

    def _qualname(self, name: str) -> str:
        return ".".join([*self._scope, name])

    def _add_symbol(self, node: ast.AST, kind: str, **extra: object) -> SymbolInfo:
        docstring = (
            ast.get_docstring(node)
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            else None
        )
        info = SymbolInfo(
            name=getattr(node, "name", "?"),
            qualified=self._qualname(getattr(node, "name", "?")),
            kind=kind,
            line=getattr(node, "lineno", 0),
            end_line=getattr(node, "end_lineno", getattr(node, "lineno", 0)),
            docstring=docstring or "",
            **extra,  # type: ignore[arg-type]
        )
        self.symbols.append(info)
        return info

    # ── definitions ──────────────────────────────────────────────

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        info = self._add_symbol(node, "class")
        info.decorators = [ast.unparse(d) for d in node.decorator_list]
        self._scope.append(node.name)
        self._class_stack.append(node.name)
        # Class-level calls (default args, class attributes) belong to
        # the enclosing callable, not to the class itself.
        for child in node.body:
            self.visit(child)
        self._class_stack.pop()
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_callable(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_callable(node, is_async=True)

    def _visit_callable(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        is_async: bool,
    ) -> None:
        # A method is a function whose immediate enclosing scope is a
        # class — the class is the top of both stacks simultaneously.
        in_class = bool(self._class_stack) and self._scope[-1] == self._class_stack[-1]
        kind = "method" if in_class else "function"
        info = self._add_symbol(
            node, kind,
            params=self._params(node.args),
            returns=ast.unparse(node.returns) if node.returns else "",
            decorators=[ast.unparse(d) for d in node.decorator_list],
        )
        info.complexity = _complexity(node)

        outer_callable = self._callable
        # Only rebind the call owner when this callable is one that the
        # concept graph names — module functions and class methods.
        # Nested functions inside a callable keep the outer owner.
        owns_calls = outer_callable is None or in_class
        if owns_calls:
            self._callable = info
        self._scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self._scope.pop()
        if owns_calls:
            self._callable = outer_callable

    # ── statements that carry signal ─────────────────────────────

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = "." * node.level + (node.module or "")
        for alias in node.names:
            self.imports.append(f"{base}.{alias.name}" if base else alias.name)

    def visit_Call(self, node: ast.Call) -> None:
        callee = _callee_name(node.func)
        if (
            callee
            and self._callable is not None
            and callee.split(".")[-1].split("(")[0] not in _BUILTIN_CALLS
            and callee.split(".")[0] not in _BUILTIN_CALLS
        ):
            edge = CallEdge(caller=self._callable.qualified, callee=callee)
            self.call_edges.append(edge)
            if callee not in self._callable.calls:
                self._callable.calls.append(callee)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        if self._callable is not None and node.exc is not None:
            name = _callee_name(node.exc) or (
                node.exc.id if isinstance(node.exc, ast.Name) else None
            )
            if name and name not in self._callable.raises:
                self._callable.raises.append(name)
        self.generic_visit(node)

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _params(args: ast.arguments) -> list[str]:
        params = [a.arg for a in args.posonlyargs + args.args]
        if args.vararg:
            params.append(f"*{args.vararg.arg}")
        params.extend(a.arg for a in args.kwonlyargs)
        if args.kwarg:
            params.append(f"**{args.kwarg.arg}")
        return params


def _callee_name(node: ast.expr) -> str | None:
    """Dotted name of a call target, or None if not statically named."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _callee_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        # foo()() — the interesting call is the outer one's target.
        return _callee_name(node.func)
    if isinstance(node, ast.Subscript):
        return _callee_name(node.value)
    return None


def _complexity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Cyclomatic complexity: 1 + number of decision points."""
    score = 1
    decision_nodes = (
        ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler,
        ast.Assert, ast.IfExp, ast.Match,
    )
    for sub in ast.walk(node):
        if isinstance(sub, decision_nodes):
            score += 1
        elif isinstance(sub, ast.BoolOp):
            # each `and`/`or` operand past the first is a branch
            score += len(sub.values) - 1
        elif isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            score += len(sub.generators) + sum(len(g.ifs) for g in sub.generators)
    return score


def analyze_python_source(source: str, path: str = "<string>") -> FileAnalysis:
    """Analyze Python source text into a FileAnalysis.

    Raises ``SyntaxError`` on unparseable input — callers that want a
    soft failure should catch it (the caller knows whether a broken
    file is a warning or an answer).
    """
    tree = ast.parse(source, filename=path)
    analysis = analyze_python_tree(tree, path)
    analysis.lines = source.count("\n") + (
        1 if source and not source.endswith("\n") else 0
    )
    return analysis


def analyze_python_tree(tree: ast.Module, path: str = "<string>") -> FileAnalysis:
    """Analyze an already-parsed module AST."""
    module_name = _module_name_from_path(path)
    visitor = _ScopeVisitor(module_name)
    visitor.visit(tree)
    return FileAnalysis(
        path=path,
        docstring=ast.get_docstring(tree) or "",
        symbols=visitor.symbols,
        imports=visitor.imports,
        call_edges=visitor.call_edges,
    )


def _module_name_from_path(path: str) -> str:
    """Best-effort module name from a file path."""
    stem = path.rsplit("/", 1)[-1].removesuffix(".py")
    if stem == "__init__":
        parent = path.rsplit("/", 2)
        return parent[-2] if len(parent) > 1 else stem
    return stem or "module"


def impact_of(analysis: FileAnalysis, symbol: str) -> dict[str, object]:
    """What breaks if ``symbol`` changes — transitive dependents.

    Args:
        analysis: The analyzed file.
        symbol: A short name or qualified name to look up.

    Returns a dict with the matched symbol, direct dependents, and the
    transitive dependent set — the change-propagation surface.
    """
    match = _resolve_symbol(analysis, symbol)
    if match is None:
        return {"symbol": symbol, "found": False, "dependents": []}

    # caller → set of callee short names, then invert.
    callees_of: dict[str, set[str]] = {}
    for edge in analysis.call_edges:
        callee_short = edge.callee.split(".")[-1]
        callees_of.setdefault(edge.caller, set()).add(callee_short)

    callers_of: dict[str, set[str]] = {}
    for caller, callees in callees_of.items():
        for callee in callees:
            callers_of.setdefault(callee, set()).add(caller)

    direct = sorted(callers_of.get(match.name, set()) |
                    callers_of.get(match.qualified, set()))
    seen: set[str] = set(direct)
    queue = list(direct)
    while queue:
        current = queue.pop()
        current_short = current.split(".")[-1]
        for up in callers_of.get(current, set()) | callers_of.get(current_short, set()):
            if up not in seen and up != match.qualified:
                seen.add(up)
                queue.append(up)

    return {
        "symbol": match.qualified,
        "kind": match.kind,
        "line": match.line,
        "found": True,
        "direct_dependents": direct,
        "transitive_dependents": sorted(seen),
        "fan_in": len(direct),
        "fan_out": len(match.calls),
        "complexity": match.complexity,
    }


def _resolve_symbol(analysis: FileAnalysis, name: str) -> SymbolInfo | None:
    """Find a symbol by qualified name, then by short name."""
    for sym in analysis.symbols:
        if sym.qualified == name or sym.qualified.endswith(f".{name}"):
            return sym
    short = name.split(".")[-1]
    matches = [s for s in analysis.symbols if s.name == short]
    return matches[0] if matches else None


def summarize(analysis: FileAnalysis) -> dict[str, object]:
    """JSON-safe structural summary — the data a mind composes from."""
    by_kind: dict[str, int] = {}
    for sym in analysis.symbols:
        by_kind[sym.kind] = by_kind.get(sym.kind, 0) + 1
    hot_spots = sorted(
        analysis.functions, key=lambda s: -s.complexity,
    )[:5]
    return {
        "path": analysis.path,
        "language": analysis.language,
        "lines": analysis.lines,
        "docstring": analysis.docstring.split("\n")[0] if analysis.docstring else "",
        "symbol_counts": by_kind,
        "imports": sorted(set(analysis.imports)),
        "call_edge_count": len(analysis.call_edges),
        "max_complexity": analysis.max_complexity,
        "avg_complexity": round(analysis.avg_complexity, 2),
        "hot_spots": [
            {"name": s.qualified, "complexity": s.complexity, "line": s.line}
            for s in hot_spots if s.complexity > 1
        ],
        "symbols": [
            {
                "name": s.qualified,
                "kind": s.kind,
                "line": s.line,
                "complexity": s.complexity,
                "calls": len(s.calls),
            }
            for s in analysis.symbols
        ],
    }
