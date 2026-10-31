from typing import Mapping, Iterable
ARMS=("A","B","C","D")
def _render_segments(item: Mapping, model_history: Mapping[str, str]) -> str:
    missing = [
        segment["record_id"]
        for segment in item["prompt_segments"]
        if segment["kind"] == "answer" and segment["record_id"] not in model_history
    ]
    if missing:
        raise ValueError(f"Missing model history for retained answer slots: {missing}")
    output = []
    for segment in item["prompt_segments"]:
        if segment["kind"] == "text":
            output.append(segment["text"])
            continue
        if segment["kind"] != "answer":
            raise ValueError(f"Unknown prompt segment kind: {segment['kind']}")
        value = model_history[segment["record_id"]]
        if not isinstance(value, str):
            raise TypeError(f"Model history for {segment['record_id']} must be text")
        spacer = "" if not value or value[0].isspace() else " "
        output.append(f"{segment['prefix']}{spacer}{value}")
    return "".join(output)


def render_prompt(
    item: Mapping,
    arm: str,
    model_history: Mapping[str, str] | None = None,
) -> str:
    normalized_arm = arm.upper()
    if normalized_arm not in ARMS:
        raise ValueError(f"Unknown experiment arm: {arm}")
    if normalized_arm == "A":
        return item["prompt_a"]
    if normalized_arm == "B":
        return item["prompt_b"]
    if normalized_arm == "C":
        return item["prompt_c"]
    return _render_segments(item, model_history or {})


def trajectory_items(
    items: Iterable[Mapping],
    target_ids: Iterable[str] | None = None,
) -> list[dict]:
    items = list(items)
    item_map = {item["id"]: item for item in items}
    selected_ids = (
        {item["id"] for item in items if item["has_retained_gold_history"]}
        if target_ids is None
        else set(target_ids)
    )
    unknown = sorted(selected_ids - set(item_map))
    if unknown:
        raise ValueError(f"Unknown trajectory target IDs: {unknown[:10]}")
    selected_by_group = {}
    for record_id in selected_ids:
        item = item_map[record_id]
        selected_by_group.setdefault(item["group_id"], []).append(item)
    result = []
    for group_id in sorted(selected_by_group):
        targets = sorted(selected_by_group[group_id], key=lambda item: item["position"])
        deepest = targets[-1]
        selected_group_ids = {item["id"] for item in targets}
        for trajectory_item in deepest["trajectory"]:
            value = dict(trajectory_item)
            value["group_id"] = group_id
            value["is_scored_target"] = value["id"] in selected_group_ids
            value["selected_target_ids"] = [item["id"] for item in targets]
            result.append(value)
    return result
