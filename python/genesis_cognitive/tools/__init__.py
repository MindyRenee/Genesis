"""Tools bundle — the capabilities Genesis can invoke.

This package bundles the tool framework and Genesis's concrete
capabilities into a coherent unit. A tool is a callable capability
she can decide to invoke: narrow, inspectable, and safe.

Subsystems:
    Tool, ToolRegistry, ToolResult, get_tools — the tool framework
        itself (``framework.py``)
    web_search — read-only web search and page fetching
    source_registry — trusted-source registry and query APIs
    explorer — filesystem exploration for self-directed learning
    code_learner — introspection over her own codebase
    project_creator, project_composer — composing real projects from
        her concept network
"""

from __future__ import annotations

from .framework import Tool, ToolRegistry, ToolResult, get_tools

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "get_tools",
]
