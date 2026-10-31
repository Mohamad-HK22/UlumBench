from __future__ import annotations

import ast
import json
import math
import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Any, Mapping, Sequence

from .taxonomy import load_taxonomy

NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
NUMBER_RE = re.compile(rf"^{NUMBER_PATTERN}$")
NUMBER_UNIT_RE = re.compile(rf"^\s*({NUMBER_PATTERN})\s*(\D.+?)\s*$")
ORDERED_SEQUENCE_TYPES = {
    "array",
    "cache_state",
    "construction_state",
    "coordinate",
    "coordinate_vector",
    "deque_state",
    "heap_state",
    "linked_list",
    "mixed_state",
    "object_state",
    "parent_state",
    "parser_state",
    "point",
    "probe_state",
    "proof_state",
    "radix_state",
    "recurrence_state",
    "traversal_state",
    "vector",
}
SYMBOL_TYPES = {"bitstring", "dfa_state", "sentential_form", "tape_word"}


def _diagnostic(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _parse_json_object(raw: Any) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    diagnostics: list[dict[str, str]] = []
    if not isinstance(raw, str):
        return None, [_diagnostic("response_not_string", "Raw response is not a string")]
    if not raw.strip():
        return None, [_diagnostic("empty_response", "Raw response is empty")]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [
            _diagnostic("invalid_json", f"Invalid JSON at line {exc.lineno}, column {exc.colno}")
        ]
    if not isinstance(parsed, dict):
        diagnostics.append(_diagnostic("root_not_object", "JSON root must be an object"))
        return None, diagnostics
    return parsed, diagnostics


def _parse_result(
    parsed: dict[str, Any] | None, diagnostics: list[dict[str, str]]
) -> dict[str, Any]:
    return {
        "valid": not diagnostics,
        "format_error": diagnostics[0]["code"] if diagnostics else None,
        "format_diagnostics": diagnostics,
        "parsed_response": parsed,
    }


def parse_atomic_response(
    raw: Any,
    expected_family_id: str | None = None,
    expected_step_id: str | None = None,
) -> dict[str, Any]:
    parsed, diagnostics = _parse_json_object(raw)
    if parsed is None:
        return _parse_result(parsed, diagnostics)
    required = {"family_id", "step_id", "answer"}
    missing = sorted(required - set(parsed))
    extra = sorted(set(parsed) - required)
    if missing:
        diagnostics.append(_diagnostic("missing_fields", "Missing fields: " + ", ".join(missing)))
    if extra:
        diagnostics.append(
            _diagnostic("unexpected_fields", "Unexpected fields: " + ", ".join(extra))
        )
    if not isinstance(parsed.get("family_id"), str) or not parsed.get("family_id", "").strip():
        diagnostics.append(_diagnostic("invalid_family_id", "family_id must be a nonempty string"))
    if not isinstance(parsed.get("step_id"), str) or not re.fullmatch(
        r"step_\d+", parsed.get("step_id", "")
    ):
        diagnostics.append(_diagnostic("invalid_step_id", "step_id must use the step_N form"))
    if not isinstance(parsed.get("answer"), str):
        diagnostics.append(_diagnostic("answer_not_string", "answer must be a string"))
    elif not parsed["answer"].strip():
        diagnostics.append(_diagnostic("empty_answer", "answer must not be empty"))
    if expected_family_id is not None and parsed.get("family_id") != expected_family_id:
        diagnostics.append(
            _diagnostic("family_id_mismatch", "family_id does not match the request")
        )
    if expected_step_id is not None and parsed.get("step_id") != expected_step_id:
        diagnostics.append(_diagnostic("step_id_mismatch", "step_id does not match the request"))
    return _parse_result(parsed, diagnostics)


def parse_bundle_response(
    raw: Any,
    expected_family_id: str | None = None,
    expected_horizon: int | None = None,
    expected_step_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    parsed, diagnostics = _parse_json_object(raw)
    if parsed is None:
        return _parse_result(parsed, diagnostics)
    required = {"family_id", "horizon", "steps", "final_answer"}
    missing = sorted(required - set(parsed))
    extra = sorted(set(parsed) - required)
    if missing:
        diagnostics.append(_diagnostic("missing_fields", "Missing fields: " + ", ".join(missing)))
    if extra:
        diagnostics.append(
            _diagnostic("unexpected_fields", "Unexpected fields: " + ", ".join(extra))
        )
    if not isinstance(parsed.get("family_id"), str) or not parsed.get("family_id", "").strip():
        diagnostics.append(_diagnostic("invalid_family_id", "family_id must be a nonempty string"))
    horizon = parsed.get("horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        diagnostics.append(_diagnostic("invalid_horizon", "horizon must be an integer"))
    if expected_family_id is not None and parsed.get("family_id") != expected_family_id:
        diagnostics.append(
            _diagnostic("family_id_mismatch", "family_id does not match the request")
        )
    if expected_horizon is not None and horizon != expected_horizon:
        diagnostics.append(_diagnostic("horizon_mismatch", "horizon does not match the request"))
    expected_ids = (
        list(expected_step_ids)
        if expected_step_ids is not None
        else [f"step_{position}" for position in range(1, horizon + 1)]
        if isinstance(horizon, int) and not isinstance(horizon, bool) and horizon > 0
        else []
    )
    steps = parsed.get("steps")
    if not isinstance(steps, list):
        diagnostics.append(_diagnostic("steps_not_array", "steps must be an array"))
    else:
        actual_ids: list[Any] = []
        for position, item in enumerate(steps, start=1):
            if not isinstance(item, dict):
                diagnostics.append(
                    _diagnostic("step_not_object", f"steps[{position - 1}] must be an object")
                )
                continue
            required_step_fields = {"step_id", "answer"}
            missing_step_fields = sorted(required_step_fields - set(item))
            extra_step_fields = sorted(set(item) - required_step_fields)
            if missing_step_fields:
                diagnostics.append(
                    _diagnostic(
                        "missing_step_fields",
                        f"steps[{position - 1}] is missing: " + ", ".join(missing_step_fields),
                    )
                )
            if extra_step_fields:
                diagnostics.append(
                    _diagnostic(
                        "unexpected_step_fields",
                        f"steps[{position - 1}] has unexpected fields: "
                        + ", ".join(extra_step_fields),
                    )
                )
            step_id = item.get("step_id")
            actual_ids.append(step_id)
            if not isinstance(step_id, str) or not re.fullmatch(r"step_\d+", step_id):
                diagnostics.append(
                    _diagnostic("invalid_step_id", f"steps[{position - 1}].step_id is invalid")
                )
            answer = item.get("answer")
            if not isinstance(answer, str):
                diagnostics.append(
                    _diagnostic(
                        "answer_not_string", f"steps[{position - 1}].answer must be a string"
                    )
                )
            elif not answer.strip():
                diagnostics.append(
                    _diagnostic("empty_answer", f"steps[{position - 1}].answer must not be empty")
                )
        if len(actual_ids) != len(set(value for value in actual_ids if isinstance(value, str))):
            diagnostics.append(
                _diagnostic("duplicate_step_id", "steps contains a duplicate step_id")
            )
        if expected_ids and actual_ids != expected_ids:
            diagnostics.append(
                _diagnostic(
                    "step_order_or_count_mismatch",
                    "steps must contain every expected step exactly once and in order",
                )
            )
    final_answer = parsed.get("final_answer")
    if not isinstance(final_answer, str):
        diagnostics.append(_diagnostic("final_answer_not_string", "final_answer must be a string"))
    elif not final_answer.strip():
        diagnostics.append(_diagnostic("empty_final_answer", "final_answer must not be empty"))
    return _parse_result(parsed, diagnostics)


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return Decimal(str(value))
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"(?<=\d),(?=\d)", "", value.strip())
    if not NUMBER_RE.fullmatch(normalized):
        return None
    try:
        parsed = Decimal(normalized)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    normalized = value.normalize()
    return (
        format(normalized, "f")
        if normalized.adjusted() >= -6 and normalized.adjusted() <= 20
        else str(normalized)
    )


def _unit_key(value: str) -> str:
    normalized = value.replace("CO₂", "CO2").replace("³", "^3").strip().lower()
    normalized = re.sub(r"\s+per\s+", "/", normalized)
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    normalized = re.sub(r"\s*\^\s*", "^", normalized)
    return re.sub(r"\s+", " ", normalized)


@lru_cache(maxsize=1)
def _unit_aliases() -> dict[str, str]:
    taxonomy = load_taxonomy()
    aliases: dict[str, str] = {}
    for canonical, values in taxonomy["units"].items():
        for value in values:
            key = _unit_key(str(value))
            if key in aliases and aliases[key] != canonical:
                raise ValueError(f"Ambiguous unit alias: {value}")
            aliases[key] = str(canonical)
    return aliases


def _parse_number_unit(value: Any) -> tuple[Decimal, str] | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"(?<=\d),(?=\d)", "", value.strip())
    match = NUMBER_UNIT_RE.fullmatch(normalized)
    if match is None:
        return None
    number = _decimal(match.group(1))
    canonical_unit = _unit_aliases().get(_unit_key(match.group(2)))
    if number is None or canonical_unit is None:
        return None
    return number, canonical_unit


def _literal_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        raise ValueError("Unsupported literal")
    if isinstance(value, (int, float)):
        parsed = _decimal(value)
        if parsed is None:
            raise ValueError("Invalid numeric literal")
        return parsed
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return tuple(_literal_value(item) for item in value)
    raise ValueError("Unsupported literal")


def _parse_sequence(value: Any) -> tuple[Any, ...] | None:
    if isinstance(value, (list, tuple)):
        raw = value
    elif isinstance(value, str):
        try:
            raw = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError):
            return None
    else:
        return None
    if not isinstance(raw, (list, tuple)):
        return None
    try:
        parsed = _literal_value(raw)
    except ValueError:
        return None
    return parsed if isinstance(parsed, tuple) else None


def _parse_set(value: Any) -> frozenset[Any] | None:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in {"∅", "{}", "set()"}:
            return frozenset()
        try:
            raw = ast.literal_eval(normalized)
        except (SyntaxError, ValueError):
            return None
    else:
        raw = value
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return None
    try:
        return frozenset(_literal_value(item) for item in raw)
    except (TypeError, ValueError):
        return None


def _normalized_literal_text(value: Any) -> Any:
    if isinstance(value, Decimal):
        return _decimal_text(value)
    if isinstance(value, tuple):
        return [_normalized_literal_text(item) for item in value]
    return value


def _parse_schedule(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    fields: dict[str, str] = {}
    for component in value.strip().split(";"):
        if "=" not in component:
            return None
        key, raw_value = component.split("=", 1)
        normalized_key = key.strip().lower()
        if normalized_key in fields:
            return None
        fields[normalized_key] = raw_value.strip()
    if set(fields) != {"selected", "elapsed", "remaining"}:
        return None
    elapsed = _decimal(fields["elapsed"])
    if elapsed is None:
        return None
    remaining = tuple(item.strip() for item in fields["remaining"].split(",") if item.strip())
    return {"selected": fields["selected"], "elapsed": elapsed, "remaining": remaining}


def _normalized_schedule(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selected": value["selected"],
        "elapsed": _decimal_text(value["elapsed"]),
        "remaining": list(value["remaining"]),
    }


def _candidate_for_record(candidate: Any) -> Any:
    if isinstance(candidate, (str, int, float, list, tuple)) or candidate is None:
        return candidate
    try:
        return json.dumps(candidate, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(candidate)


def score_answer(candidate: Any, step: Mapping[str, Any]) -> dict[str, Any]:
    gold = step["gold_answer"]
    answer_type = str(step["answer_type"])
    method = str(step["evaluation_method"])
    tolerance = Decimal(str(step["tolerance"]))
    result: dict[str, Any] = {
        "candidate_answer": _candidate_for_record(candidate),
        "gold_answer": gold,
        "answer_type": answer_type,
        "evaluation_method": method,
        "tolerance": float(tolerance),
        "correct": False,
        "score_error": None,
        "normalized_candidate": None,
        "normalized_gold": None,
        "absolute_error": None,
        "candidate_unit": None,
        "gold_unit": None,
    }
    if candidate is None:
        result["score_error"] = "missing_answer"
        return result

    if method == "exact_tolerance" and answer_type == "number":
        parsed_candidate = _decimal(candidate)
        parsed_gold = _decimal(gold)
        if parsed_candidate is None or parsed_gold is None:
            result["score_error"] = "invalid_number"
            return result
        absolute_error = abs(parsed_candidate - parsed_gold)
        result["normalized_candidate"] = _decimal_text(parsed_candidate)
        result["normalized_gold"] = _decimal_text(parsed_gold)
        result["absolute_error"] = float(absolute_error)
        result["correct"] = absolute_error <= tolerance
        return result

    if method == "numeric_units" and answer_type == "number_with_unit":
        parsed_candidate = _parse_number_unit(candidate)
        parsed_gold = _parse_number_unit(gold)
        if parsed_candidate is None:
            result["score_error"] = "invalid_number_or_unit"
            return result
        if parsed_gold is None:
            result["score_error"] = "invalid_gold_number_or_unit"
            return result
        candidate_number, candidate_unit = parsed_candidate
        gold_number, gold_unit = parsed_gold
        absolute_error = abs(candidate_number - gold_number)
        result["normalized_candidate"] = f"{_decimal_text(candidate_number)} {candidate_unit}"
        result["normalized_gold"] = f"{_decimal_text(gold_number)} {gold_unit}"
        result["absolute_error"] = float(absolute_error)
        result["candidate_unit"] = candidate_unit
        result["gold_unit"] = gold_unit
        result["correct"] = candidate_unit == gold_unit and absolute_error <= tolerance
        if candidate_unit != gold_unit:
            result["score_error"] = "unit_mismatch"
        return result

    if method != "exact_text":
        result["score_error"] = "unsupported_method_type_pair"
        return result

    if answer_type == "integer":
        parsed_candidate = _decimal(candidate)
        parsed_gold = _decimal(gold)
        if (
            parsed_candidate is None
            or parsed_gold is None
            or parsed_candidate != parsed_candidate.to_integral_value()
        ):
            result["score_error"] = "invalid_integer"
            return result
        result["normalized_candidate"] = _decimal_text(parsed_candidate)
        result["normalized_gold"] = _decimal_text(parsed_gold)
        result["correct"] = parsed_candidate == parsed_gold
        return result

    if answer_type in ORDERED_SEQUENCE_TYPES:
        parsed_candidate = _parse_sequence(candidate)
        parsed_gold = _parse_sequence(gold)
        if parsed_candidate is None or parsed_gold is None:
            result["score_error"] = "invalid_ordered_state"
            return result
        result["normalized_candidate"] = _normalized_literal_text(parsed_candidate)
        result["normalized_gold"] = _normalized_literal_text(parsed_gold)
        result["correct"] = parsed_candidate == parsed_gold
        return result

    if answer_type == "set":
        parsed_candidate = _parse_set(candidate)
        parsed_gold = _parse_set(gold)
        if parsed_candidate is None or parsed_gold is None:
            result["score_error"] = "invalid_set"
            return result
        result["normalized_candidate"] = sorted(
            (_normalized_literal_text(value) for value in parsed_candidate), key=str
        )
        result["normalized_gold"] = sorted(
            (_normalized_literal_text(value) for value in parsed_gold), key=str
        )
        result["correct"] = parsed_candidate == parsed_gold
        return result

    if answer_type in SYMBOL_TYPES:
        if not isinstance(candidate, str):
            result["score_error"] = "invalid_symbolic_answer"
            return result
        if answer_type in {"bitstring", "tape_word"}:
            normalized_candidate = re.sub(r"\s+", "", candidate)
            normalized_gold = re.sub(r"\s+", "", str(gold))
            if not re.fullmatch(r"[01]+", normalized_candidate):
                result["score_error"] = "invalid_binary_word"
                return result
        elif answer_type == "sentential_form":
            normalized_candidate = re.sub(r"\s+", "", candidate)
            normalized_gold = re.sub(r"\s+", "", str(gold))
        else:
            normalized_candidate = re.sub(r"\s+", "", candidate)
            normalized_gold = re.sub(r"\s+", "", str(gold))
        result["normalized_candidate"] = normalized_candidate
        result["normalized_gold"] = normalized_gold
        result["correct"] = normalized_candidate == normalized_gold
        return result

    if answer_type == "schedule_state":
        parsed_candidate = _parse_schedule(candidate)
        parsed_gold = _parse_schedule(gold)
        if parsed_candidate is None or parsed_gold is None:
            result["score_error"] = "invalid_schedule_state"
            return result
        result["normalized_candidate"] = _normalized_schedule(parsed_candidate)
        result["normalized_gold"] = _normalized_schedule(parsed_gold)
        result["correct"] = parsed_candidate == parsed_gold
        return result

    result["score_error"] = "unsupported_answer_type"
    return result


def _row_response(row: Mapping[str, Any]) -> Any:
    for key in ("raw_response", "response_text", "response", "output"):
        if key in row:
            return row[key]
    return None


def score_call(row: Mapping[str, Any], family: Mapping[str, Any]) -> dict[str, Any]:
    condition = str(row.get("condition") or row.get("condition_id") or "")
    family_id = str(family["family_id"])
    raw_response = _row_response(row)
    step_by_id = {str(step["step_id"]): step for step in family["steps"]}

    if condition == "atomic":
        step_id = str(row.get("step_id") or f"step_{int(row['horizon'])}")
        parsed = parse_atomic_response(raw_response, family_id, step_id)
        parsed_response = parsed["parsed_response"]
        raw_candidate = parsed_response.get("answer") if isinstance(parsed_response, dict) else None
        candidate = raw_candidate if isinstance(raw_candidate, str) else None
        score = score_answer(candidate, step_by_id[step_id])
        score.update({"step_id": step_id, "position": int(step_id.split("_")[1])})
        return {
            "request_id": row.get("request_id"),
            "family_id": family_id,
            "condition": condition,
            "horizon": None,
            "format_valid": parsed["valid"],
            "format_error": parsed["format_error"],
            "format_diagnostics": parsed["format_diagnostics"],
            "parsed_response": parsed_response,
            "step_scores": [score],
            "final_score": None,
            "whole_chain_correct": bool(score["correct"]),
            "all_outputs_correct": bool(score["correct"]),
        }

    if condition not in {"oracle_bundle", "composed"}:
        raise ValueError(f"Unknown condition: {condition}")
    horizon = int(row["horizon"])
    expected_step_ids = [f"step_{position}" for position in range(1, horizon + 1)]
    parsed = parse_bundle_response(raw_response, family_id, horizon, expected_step_ids)
    parsed_response = parsed["parsed_response"]
    answer_by_step: dict[str, Any] = {}
    if isinstance(parsed_response, dict) and isinstance(parsed_response.get("steps"), list):
        for item in parsed_response["steps"]:
            if isinstance(item, dict) and isinstance(item.get("step_id"), str):
                answer = item.get("answer")
                if item["step_id"] not in answer_by_step and isinstance(answer, str):
                    answer_by_step[item["step_id"]] = answer
    step_scores: list[dict[str, Any]] = []
    for position, step_id in enumerate(expected_step_ids, start=1):
        step_score = score_answer(answer_by_step.get(step_id), step_by_id[step_id])
        step_score.update({"step_id": step_id, "position": position})
        step_scores.append(step_score)
    raw_final_candidate = (
        parsed_response.get("final_answer") if isinstance(parsed_response, dict) else None
    )
    final_candidate = raw_final_candidate if isinstance(raw_final_candidate, str) else None
    final_score = score_answer(final_candidate, step_by_id[expected_step_ids[-1]])
    final_score.update({"step_id": expected_step_ids[-1], "position": horizon})
    whole_chain_correct = all(bool(value["correct"]) for value in step_scores)
    return {
        "request_id": row.get("request_id"),
        "family_id": family_id,
        "condition": condition,
        "horizon": horizon,
        "format_valid": parsed["valid"],
        "format_error": parsed["format_error"],
        "format_diagnostics": parsed["format_diagnostics"],
        "parsed_response": parsed_response,
        "step_scores": step_scores,
        "final_score": final_score,
        "whole_chain_correct": whole_chain_correct,
        "all_outputs_correct": whole_chain_correct and bool(final_score["correct"]),
    }
