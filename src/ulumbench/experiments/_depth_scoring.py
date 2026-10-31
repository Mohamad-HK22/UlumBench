from ulumbench.scoring.typed import score_answer
from ulumbench.util import strict_json
def parse_response(raw, request):
    try:
        data = strict_json(raw)
        if not isinstance(data, dict) or set(data) != {"request_id", "answers"}:
            raise ValueError("Expected exactly request_id and answers")
        if data["request_id"] != request["request_id"]:
            raise ValueError("Wrong request_id")
        if not isinstance(data["answers"], list):
            raise ValueError("answers must be an array")
        expected = [t["task_id"] for t in request["tasks"]]
        observed = []
        values = {}
        for item in data["answers"]:
            if not isinstance(item, dict) or set(item) != {"task_id", "answer"}:
                raise ValueError("Unexpected answer fields")
            if not isinstance(item["answer"], str) or not item["answer"].strip():
                raise ValueError("Answers must be nonempty strings")
            observed.append(item["task_id"])
            values[item["task_id"]] = item["answer"]
        if observed != expected:
            raise ValueError("Missing, duplicate, unknown, or out-of-order task IDs")
        return {"valid": True, "answers": values, "error": None}
    except (ValueError, TypeError, KeyError) as exc:
        return {"valid": False, "answers": {}, "error": str(exc)}


def score_response(raw, request, families):
    parsed = parse_response(raw, request)
    scores = []
    for task in request["tasks"]:
        family = families[task["family_id"]]
        step = family["steps"][task["position"] - 1]
        result = score_answer(parsed["answers"].get(task["task_id"]), step)
        scores.append({**task, **result})
    return {
        "format_valid": parsed["valid"],
        "format_error": parsed["error"],
        "steps": scores,
        "terminal_correct": scores[-1]["correct"],
        "whole_correct": all(x["correct"] for x in scores),
    }
