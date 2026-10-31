from __future__ import annotations
import hashlib, json
from typing import Any, Mapping, Sequence

def _step_map(family: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(step["step_id"]): step for step in family["steps"]}


def _atomic_prompt(family: Mapping[str, Any], step_index: int) -> str:
    step = family["steps"][step_index - 1]
    lines = [
        str(family["shared_context"]).strip(),
        "",
        f"Family ID: {family['family_id']}",
        f"Step: {step['step_id']}",
    ]
    dependencies = _step_map(family)
    if step["depends_on"]:
        lines.extend(["", "Use these verified predecessor answers:"])
        for dependency_id in step["depends_on"]:
            lines.append(f"{dependency_id}: {dependencies[dependency_id]['gold_answer']}")
    lines.extend(
        [
            "",
            str(step["question"]).strip(),
            "",
            "Return valid JSON only using this exact schema:",
            json.dumps(
                {"family_id": family["family_id"], "step_id": step["step_id"], "answer": "..."},
                separators=(",", ":"),
            ),
        ]
    )
    return "\n".join(lines)


def _bundle_schema(family_id: str, horizon: int) -> str:
    value = {
        "family_id": family_id,
        "horizon": horizon,
        "steps": [
            {"step_id": f"step_{position}", "answer": "..."} for position in range(1, horizon + 1)
        ],
        "final_answer": "...",
    }
    return json.dumps(value, separators=(",", ":"))


def _oracle_bundle_prompt(family: Mapping[str, Any], horizon: int) -> str:
    step_by_id = _step_map(family)
    lines = [
        str(family["shared_context"]).strip(),
        "",
        "Solve every listed step independently. For each step, use every supplied verified predecessor answer.",
        f"Family ID: {family['family_id']}",
        f"Horizon: {horizon}",
        "",
    ]
    for step in family["steps"][:horizon]:
        lines.append(f"{step['step_id']}: {str(step['question']).strip()}")
        for dependency_id in step["depends_on"]:
            lines.append(
                f"Verified {dependency_id} for this step: {step_by_id[dependency_id]['gold_answer']}"
            )
    lines.extend(
        [
            "",
            "Return an answer for every step and a final answer. Return valid JSON only using this exact schema:",
            _bundle_schema(str(family["family_id"]), horizon),
        ]
    )
    return "\n".join(lines)


def _composed_prompt(family: Mapping[str, Any], horizon: int) -> str:
    lines = [
        str(family["shared_context"]).strip(),
        "",
        "Solve every listed step in order. Use your computed answer from each step in any downstream step that depends on it.",
        "Do not omit a step.",
        f"Family ID: {family['family_id']}",
        f"Horizon: {horizon}",
        "",
    ]
    for step in family["steps"][:horizon]:
        dependencies = ", ".join(step["depends_on"]) if step["depends_on"] else "none"
        lines.append(
            f"{step['step_id']} [depends_on: {dependencies}]: {str(step['question']).strip()}"
        )
    lines.extend(
        [
            "",
            "Return an answer for every step and a final answer. Return valid JSON only using this exact schema:",
            _bundle_schema(str(family["family_id"]), horizon),
        ]
    )
    return "\n".join(lines)
