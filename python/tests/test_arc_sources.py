from __future__ import annotations

import json

import pytest

from parietal_lobe.arc_sources import (
    ARCSourceError,
    arc1_problem_view,
    arc3_legal_actions,
    list_arc1_tasks,
    load_arc1_task,
)


def test_arc1_problem_view_hides_test_outputs_by_default() -> None:
    task = {
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[2]], "output": [[3]]}],
    }
    view = arc1_problem_view(task)
    assert view["train"] == [{"input": [[0]], "output": [[1]]}]
    assert view["test"] == [{"input": [[2]]}]


def test_arc1_problem_view_can_include_test_outputs_explicitly() -> None:
    task = {"train": [], "test": [{"input": [[2]], "output": [[3]]}]}
    assert arc1_problem_view(task, include_test_outputs=True)["test"][0]["output"] == [[3]]


def test_load_arc1_task_rejects_non_rectangular_grid(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"train": [{"input": [[0, 1], [1]], "output": [[0]]}], "test": [{"input": [[0]]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ARCSourceError, match="rectangular"):
        load_arc1_task(path)


def test_load_arc1_task_rejects_out_of_range_colors(tmp_path) -> None:
    path = tmp_path / "bad-color.json"
    path.write_text(
        json.dumps({"train": [{"input": [[10]], "output": [[0]]}], "test": [{"input": [[0]]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ARCSourceError, match="0..9"):
        load_arc1_task(path)


def test_list_arc1_tasks_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="split"):
        list_arc1_tasks("private")



def test_arc3_legal_actions_reads_current_environment_action_space() -> None:
    class Action:
        def __init__(self, name: str) -> None:
            self.name = name

    class Environment:
        action_space = [Action("ACTION3"), Action("ACTION1")]

    assert arc3_legal_actions(Environment()) == ["ACTION1", "ACTION3"]


def test_arc3_legal_actions_rejects_missing_action_space() -> None:
    class Environment:
        pass

    with pytest.raises(ARCSourceError, match="action_space"):
        arc3_legal_actions(Environment())
