"""Tools bundle — the capabilities Genesis can invoke.

This package bundles the tool framework and Genesis's concrete
capabilities into a coherent unit. A tool is a callable capability
it can decide to invoke: narrow, inspectable, and safe.

Subsystems:
    Tool, ToolRegistry, ToolResult, get_tools — the tool framework
        itself (``framework.py``)
    web_search — read-only web search and page fetching
    source_registry — trusted-source registry and query APIs
    explorer — filesystem exploration for self-directed learning
    code_learner — introspection over its own codebase
    code_analysis — scoped structural analysis: call graphs, cyclomatic
        complexity, and change-impact surfaces for Python source
    project_creator, project_composer — composing real projects from
        its concept network
"""

from __future__ import annotations

from .framework import Tool, ToolRegistry, ToolResult, get_tools

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "get_tools",
]
