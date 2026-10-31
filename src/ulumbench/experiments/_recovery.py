import json
from ulumbench.scoring.typed import score_answer
def prompt(family, record, arm):
    k = record["position"]
    supplied = record["wrong_answer"] if arm in {"B", "D"} else record["gold_answer"]
    instruction = (
        "Before continuing, verify the candidate calculation from the original "
        "context and preceding steps, and correct it if necessary."
        if arm in {"C", "D"}
        else "Continue solving the remaining steps."
    )
    lines = ["Case:", family["shared_context"], "", "Operations:"]
    lines.extend(f"{s['step_id']}: {s['question']}" for s in family["steps"])
    lines += ["", "Earlier reference values:"]
    lines.extend(f"{s['step_id']}: {s['gold_answer']}" for s in family["steps"][: k - 1])
    lines += [
        f"Candidate calculation for step_{k}: {supplied}",
        instruction,
        (
            "Return the state you use for the candidate step and your answers for every "
            "remaining step. Return JSON only; every answer must be a string."
        ),
    ]
    # No requested error-detection judgment: doing so would itself cue verification in A/B.
    schema = {
        "request_id": f"{record['id']}:{arm}",
        "answers": [{"step_id": f"step_{i}", "answer": "..."} for i in range(k, 9)],
    }
    lines.append(json.dumps(schema))
    return "\n".join(lines)

def score_response(raw, family, injection, arm):
    expected = [f"step_{i}" for i in range(injection["position"], 9)]
    values, error = {}, None
    try:

        def unique_pairs(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError("duplicate JSON key")
                obj[key] = value
            return obj

        obj = json.loads(raw, object_pairs_hook=unique_pairs)
        if not isinstance(obj, dict) or set(obj) != {"request_id", "answers"}:
            raise ValueError("wrong response fields")
        if obj["request_id"] != f"{injection['id']}:{arm}":
            raise ValueError("wrong request ID")
        if not isinstance(obj["answers"], list):
            raise TypeError("answers must be an array")
        ids = []
        for item in obj["answers"]:
            if not isinstance(item, dict) or set(item) != {"step_id", "answer"}:
                raise ValueError("wrong answer fields")
            if not isinstance(item["answer"], str) or not item["answer"].strip():
                raise ValueError("answer must be nonempty string")
            ids.append(item["step_id"])
            values[item["step_id"]] = item["answer"]
        if ids != expected:
            raise ValueError("missing, duplicate, or unordered steps")
    except (ValueError, TypeError, KeyError) as exc:
        values, error = {}, str(exc)
    trace = injection["trajectory_check"]
    comparisons = []
    for i in range(injection["position"], 9):
        step = family["steps"][i - 1]
        candidate = values.get(step["step_id"])
        gold_correct = score_answer(candidate, step)["correct"]
        alt = trace.get("wrong_state_trajectory", {}).get(step["step_id"])
        matches_wrong = (
            None
            if alt is None
            else score_answer(candidate, {**step, "gold_answer": alt})["correct"]
        )
        distinguishable = None if alt is None else not score_answer(alt, step)["correct"]
        comparisons.append(
            {
                "step_id": step["step_id"],
                "gold_correct": gold_correct,
                "matches_wrong_trajectory": matches_wrong,
                "trajectory_distinguishable": distinguishable,
            }
        )
    return {
        "format_valid": error is None,
        "format_error": error,
        "candidate_state_correct": comparisons[0]["gold_correct"],
        "terminal_correct": comparisons[-1]["gold_correct"],
        "steps": comparisons,
        "interpretation": "Correct final answers alone do not establish detection or repair.",
    }
