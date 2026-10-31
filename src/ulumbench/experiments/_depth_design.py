from __future__ import annotations
import random
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from ulumbench.scoring.typed import score_answer
from ulumbench.util import canonical, digest

def task(family, position, slot):
    return {"task_id": f"task_{slot}", "family_id": family["family_id"], "position": position}


def predecessors(family, position):
    steps = {s["step_id"]: s for s in family["steps"]}
    return [steps[s]["gold_answer"] for s in family["steps"][position - 1]["depends_on"]]


def collision_count(left, right, family_index):
    a, b = family_index[left["family_id"]], family_index[right["family_id"]]
    a_step, b_step = a["steps"][left["position"] - 1], b["steps"][right["position"] - 1]
    return sum(
        score_answer(v, a_step)["correct"] for v in predecessors(b, right["position"])
    ) + sum(score_answer(v, b_step)["correct"] for v in predecessors(a, left["position"]))


def assign_block(block, seed):
    """One permutation per step position; no family repeats within a bundle.

    Matching is entirely outcome-blind. Prefer similar operation answer types and
    topics, and minimize explicit predecessor-value collisions. Each source task
    occurs exactly once per condition/horizon across this block.
    """
    rng = random.Random(seed)
    n, depth = len(block), block[0]["chain_length"]
    index = {f["family_id"]: f for f in block}
    assignments = [[task(f, 1, 1)] for f in block]
    for position in range(2, depth + 1):
        costs = np.zeros((n, n))
        for i, anchor in enumerate(block):
            used = {t["family_id"] for t in assignments[i]}
            target = anchor["steps"][position - 1]
            for j, candidate in enumerate(block):
                proposed = task(candidate, position, position)
                if candidate["family_id"] in used:
                    costs[i, j] = 1e9
                    continue
                collisions = sum(collision_count(t, proposed, index) for t in assignments[i])
                costs[i, j] = (
                    100 * collisions
                    + 10
                    * (target["answer_type"] != candidate["steps"][position - 1]["answer_type"])
                    + 3 * (anchor["topic"] != candidate["topic"])
                    + rng.random()
                )
        rows, cols = linear_sum_assignment(costs)
        for i, j in zip(rows, cols, strict=True):
            if costs[i, j] >= 1e9:
                raise ValueError("Independent matching has no valid assignment")
            assignments[i].append(task(block[j], position, position))
    return assignments


def prompt_for(request, family_index):
    dependent = request["condition"] == "composed"
    lines = [
        "Solve the tasks below in the listed order.",
        (
            "Use your computed answers where a later task depends on them."
            if dependent
            else "Each task is independent. Its case and supplied inputs apply only to that task."
        ),
        "Local references such as step 1 refer only to the numbered steps of that case.",
    ]
    if dependent:
        lines += ["", "Case:", family_index[request["family_id"]]["shared_context"]]
    for entry in request["tasks"]:
        family = family_index[entry["family_id"]]
        step = family["steps"][entry["position"] - 1]
        lines += ["", f"{entry['task_id']} (local {step['step_id']}):"]
        if not dependent:
            lines += ["Case:", family["shared_context"]]
            step_index = {s["step_id"]: s for s in family["steps"]}
            for dependency in step["depends_on"]:
                lines.append(
                    f"Supplied local {dependency}: {step_index[dependency]['gold_answer']}"
                )
        lines.append(step["question"])
    schema = {
        "request_id": request["request_id"],
        "answers": [{"task_id": t["task_id"], "answer": "..."} for t in request["tasks"]],
    }
    lines += [
        "",
        "Return exactly this JSON structure, with answers in task order:",
        canonical(schema),
    ]
    return "\n".join(lines)


def build_requests(families, protocol):
    size = protocol["panel"]["block_size"]
    if size < 8:
        raise ValueError("Matching blocks must contain at least eight families")
    index = {f["family_id"]: f for f in families}
    requests = []
    by_depth = defaultdict(list)
    for family in families:
        by_depth[family["chain_length"]].append(family)
    for depth, group in sorted(by_depth.items()):
        group.sort(key=lambda f: (f["family_class"], f["topic"], f["family_id"]))
        if len(group) % size:
            raise ValueError("Each native-length stratum must divide into complete matching blocks")
        for offset in range(0, len(group), size):
            block = group[offset : offset + size]
            block_id = f"length{depth}_block{offset // size:02d}"
            assignments = assign_block(
                block, protocol["panel"]["matching_seed"] + depth * 1000 + offset
            )
            for i, family in enumerate(block):
                fid = family["family_id"]
                horizons = protocol["panel"]["horizons"] if depth == 8 else [depth]
                for position in range(1, depth + 1):
                    requests.append(
                        {
                            "request_id": f"{fid}:atomic:s{position}",
                            "condition": "atomic",
                            "horizon": 1,
                            "family_id": fid,
                            "native_length": depth,
                            "block_id": block_id,
                            "tasks": [task(family, position, 1)],
                        }
                    )
                for horizon in horizons:
                    for condition in ("composed", "independent"):
                        entries = (
                            [task(family, p, p) for p in range(1, horizon + 1)]
                            if condition == "composed"
                            else assignments[i][:horizon]
                        )
                        requests.append(
                            {
                                "request_id": f"{fid}:{condition}:h{horizon}",
                                "condition": condition,
                                "horizon": horizon,
                                "family_id": fid,
                                "native_length": depth,
                                "block_id": block_id,
                                "tasks": entries,
                            }
                        )
    for request in requests:
        entries = request["tasks"]
        request["predecessor_answer_collisions"] = (
            sum(
                collision_count(a, b, index)
                for i, a in enumerate(entries)
                for b in entries[i + 1 :]
            )
            if request["condition"] == "independent"
            else 0
        )
        request["distinct_topics"] = len({index[t["family_id"]]["topic"] for t in entries})
        request["system_prompt"] = protocol["generation"]["system_prompt"]
        request["user_prompt"] = prompt_for(request, index)
        request["prompt_bytes"] = len((request["system_prompt"] + request["user_prompt"]).encode())
        request["prompt_sha256"] = digest([request["system_prompt"], request["user_prompt"]])
    validate_requests(requests)
    return requests


def validate_requests(requests):
    ids = [r["request_id"] for r in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate request IDs")
    axes = defaultdict(lambda: defaultdict(Counter))
    for r in requests:
        if r["condition"] == "atomic":
            continue
        if len(r["tasks"]) != r["horizon"]:
            raise ValueError("Horizon differs from task count")
        if r["condition"] == "independent":
            if len({t["family_id"] for t in r["tasks"]}) != len(r["tasks"]):
                raise ValueError("A family repeats inside an independent batch")
        axes[r["block_id"], r["horizon"]][r["condition"]].update(
            (t["family_id"], t["position"]) for t in r["tasks"]
        )
    for value in axes.values():
        if value["independent"] != value["composed"]:
            raise ValueError("Condition task pools are not exactly balanced")
