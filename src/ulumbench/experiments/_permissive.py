import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ulumbench.scoring.academic import ParseResult, normalize_text, parse_mcq_answer

ChoiceInput = Mapping[str, str] | Sequence[str | Mapping[str, object]]

PERMISSIVE_SOURCES = frozenset(
    {
        "permissive_explicit_answer",
        "permissive_boxed_label",
        "permissive_leading_delimited",
        "permissive_leading_literal",
        "permissive_terminal_literal",
        "permissive_leading_label_text",
        "permissive_boundary_choice_text",
    }
)

_TOKEN = r"(?:true|false|[A-Za-z])"
_CUE_PATTERN = re.compile(
    rf"\b(?:final\s+answer|correct\s+(?:answer|option|choice)|answer)"
    rf"\s*(?:is|=|:|[-–—])\s*"
    rf"(?P<prefix>(?:\*\*|__|`|~~)?\s*[\(\[]?)"
    rf"(?P<token>{_TOKEN})(?![A-Za-z])"
    rf"(?P<suffix>[\)\]]?(?:\*\*|__|`|~~)?)",
    re.IGNORECASE,
)
_BOXED_PATTERN = re.compile(
    rf"\\boxed\s*\{{\s*(?P<token>{_TOKEN})\s*\}}",
    re.IGNORECASE,
)
_LEADING_DELIMITED_PATTERN = re.compile(
    rf"^\s*(?:#{{1,6}}\s*)?(?:[-*+]\s+)?(?:\*\*|__|`|~~)?\s*"
    rf"[\(\[]?(?P<token>{_TOKEN})(?![A-Za-z])[\)\]]?"
    rf"(?=[\s*_`~]*(?:[\)\].:]|[-–—]))",
    re.IGNORECASE,
)
_LEADING_LITERAL_PATTERN = re.compile(
    rf"^(?P<token>{_TOKEN})(?![A-Za-z])(?:\s+|$)",
    re.IGNORECASE,
)
_TERMINAL_LITERAL_PATTERN = re.compile(
    rf"^(?:(?:final\s+)?answer\s*(?:is|=|:|[-–—])\s*)?"
    rf"[\(\[]?(?P<token>{_TOKEN})[\)\]]?$",
    re.IGNORECASE,
)
_LEADING_LABEL_TEXT_PATTERN = re.compile(
    rf"^(?P<token>{_TOKEN})(?![A-Za-z])\s+(?P<text>.+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ChoiceEntry:
    label: str
    text: str


@dataclass(frozen=True)
class _Evidence:
    values: tuple[str, ...] = ()
    reason: str | None = None


def _choice_entries(choices: ChoiceInput) -> list[_ChoiceEntry]:
    if isinstance(choices, Mapping):
        return [
            _ChoiceEntry(normalize_text(str(label)), str(text)) for label, text in choices.items()
        ]
    if isinstance(choices, str):
        raise TypeError("choices must be a mapping or a sequence of choices")
    entries = []
    for index, choice in enumerate(choices):
        if isinstance(choice, Mapping):
            if "label" not in choice or "text" not in choice:
                raise ValueError("ordered choice records require label and text")
            label = normalize_text(str(choice["label"]))
            text = str(choice["text"])
        elif isinstance(choice, str):
            if index >= 26:
                raise ValueError("unlabeled choice sequences cannot exceed 26 options")
            label = chr(ord("a") + index)
            text = choice
        else:
            raise TypeError("choice entries must be strings or mappings")
        entries.append(_ChoiceEntry(label, text))
    return entries


def _allowed_labels(
    entries: Sequence[_ChoiceEntry],
    allowed_answers: Sequence[str] | None,
) -> set[str]:
    choice_labels = {entry.label for entry in entries}
    allowed = (
        {normalize_text(str(value)) for value in allowed_answers}
        if allowed_answers is not None
        else choice_labels
    )
    labels = choice_labels & allowed
    return labels or choice_labels or allowed


def _nonempty_lines(response: str) -> list[str]:
    return [line.strip() for line in response.splitlines() if line.strip()]


def _normalize_display_line(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip()
    value = re.sub(r"^#{1,6}\s*", "", value)
    value = re.sub(r"^[-*+]\s+", "", value)
    value = re.sub(r"\*\*|__|~~|`", "", value)
    value = value.strip("*_ ")
    value = value.replace("\\(", "").replace("\\)", "")
    value = value.replace("\\[", "").replace("\\]", "")
    value = value.replace("$", "")
    return normalize_text(value).rstrip(".;:")


def _strict_result(result: ParseResult) -> ParseResult:
    source = f"strict_{result.source}" if result.source else "strict"
    return ParseResult(result.value, result.status, source, result.reason)


def _cue_token_is_marked(response: str, match: re.Match[str]) -> bool:
    token = match.group("token")
    if len(normalize_text(token)) > 1 or token.isupper():
        return True
    if match.group("prefix").strip() or match.group("suffix").strip():
        return True
    tail = response[match.end() :]
    if re.match(r"\s*[\)\].,:;]", tail):
        return True
    return not tail.strip()


def _explicit_answer_evidence(response: str) -> _Evidence:
    values = tuple(
        match.group("token")
        for match in _CUE_PATTERN.finditer(response)
        if _cue_token_is_marked(response, match)
    )
    return _Evidence(values)


def _boxed_evidence(response: str) -> _Evidence:
    return _Evidence(tuple(match.group("token") for match in _BOXED_PATTERN.finditer(response)))


def _leading_delimited_evidence(first_line: str) -> _Evidence:
    match = _LEADING_DELIMITED_PATTERN.match(first_line)
    return _Evidence((match.group("token"),)) if match else _Evidence()


def _leading_literal_evidence(first_line: str) -> _Evidence:
    normalized = _normalize_display_line(first_line)
    match = _LEADING_LITERAL_PATTERN.match(normalized)
    if not match:
        return _Evidence()
    value = normalize_text(match.group("token"))
    if normalized == value or value in {"true", "false"}:
        return _Evidence((value,))
    return _Evidence()


def _terminal_literal_evidence(last_line: str) -> _Evidence:
    normalized = _normalize_display_line(last_line)
    match = _TERMINAL_LITERAL_PATTERN.fullmatch(normalized)
    return _Evidence((match.group("token"),)) if match else _Evidence()


def _leading_label_text_evidence(
    first_line: str,
    entries: Sequence[_ChoiceEntry],
) -> _Evidence:
    normalized = _normalize_display_line(first_line)
    match = _LEADING_LABEL_TEXT_PATTERN.match(normalized)
    if not match:
        return _Evidence()
    label = normalize_text(match.group("token"))
    matching_entries = [entry for entry in entries if entry.label == label]
    if len(matching_entries) > 1:
        return _Evidence((label,))
    if not matching_entries:
        return _Evidence()
    expected = _normalize_display_line(matching_entries[0].text)
    remainder = normalize_text(match.group("text")).rstrip(".;:")
    if expected and (remainder == expected or remainder.startswith(f"{expected} ")):
        return _Evidence((label,))
    return _Evidence()


def _boundary_choice_text_evidence(
    first_line: str,
    last_line: str,
    entries: Sequence[_ChoiceEntry],
) -> _Evidence:
    text_labels: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        normalized = _normalize_display_line(entry.text)
        if normalized:
            text_labels[normalized].append(entry.label)
    values = []
    for line in dict.fromkeys((first_line, last_line)):
        labels = text_labels.get(_normalize_display_line(line), [])
        if len(set(labels)) > 1:
            return _Evidence(reason="ambiguous_choice_text")
        if labels:
            values.append(labels[0])
    return _Evidence(tuple(values))


def _resolve_evidence(
    evidence: _Evidence,
    source: str,
    entries: Sequence[_ChoiceEntry],
    allowed_labels: set[str],
) -> ParseResult | None:
    if evidence.reason:
        return ParseResult(None, "invalid", reason=evidence.reason)
    if not evidence.values:
        return None
    values = tuple(normalize_text(value) for value in evidence.values)
    if any(value not in allowed_labels for value in values):
        return ParseResult(None, "invalid", reason="out_of_range")
    if len(set(values)) > 1:
        return ParseResult(None, "invalid", reason="conflicting_selection_claims")
    value = values[0]
    if sum(entry.label == value for entry in entries) > 1:
        return ParseResult(None, "invalid", reason="ambiguous_choice_label")
    return ParseResult(value, "valid", source=source)


def _selection_conflict(
    evidence_items: Sequence[_Evidence],
    entries: Sequence[_ChoiceEntry],
    allowed_labels: set[str],
) -> ParseResult | None:
    values = tuple(
        normalize_text(value) for evidence in evidence_items for value in evidence.values
    )
    if not values:
        return None
    if any(value not in allowed_labels for value in values):
        return ParseResult(None, "invalid", reason="out_of_range")
    if len(set(values)) > 1:
        return ParseResult(None, "invalid", reason="conflicting_selection_claims")
    value = values[0]
    if sum(entry.label == value for entry in entries) > 1:
        return ParseResult(None, "invalid", reason="ambiguous_choice_label")
    return None


def parse_permissive_mcq_answer(
    response: str,
    choices: ChoiceInput,
    allowed_answers: Sequence[str] | None = None,
) -> ParseResult:
    strict = parse_mcq_answer(response, choices, allowed_answers)
    if strict.status != "invalid":
        return _strict_result(strict)
    entries = _choice_entries(choices)
    allowed_labels = _allowed_labels(entries, allowed_answers)
    lines = _nonempty_lines(response)
    first_line = lines[0] if lines else ""
    last_line = lines[-1] if lines else ""
    explicit = _explicit_answer_evidence(response)
    boxed = _boxed_evidence(response)
    leading_delimited = _leading_delimited_evidence(first_line)
    leading_literal = _leading_literal_evidence(first_line)
    terminal_literal = _terminal_literal_evidence(last_line)
    leading_label_text = _leading_label_text_evidence(first_line, entries)
    conflict = _selection_conflict(
        (
            explicit,
            boxed,
            leading_delimited,
            leading_literal,
            terminal_literal,
            leading_label_text,
        ),
        entries,
        allowed_labels,
    )
    if conflict is not None:
        return conflict
    tiers = (
        ("permissive_explicit_answer", explicit),
        ("permissive_boxed_label", boxed),
        ("permissive_leading_delimited", leading_delimited),
        ("permissive_leading_literal", leading_literal),
        ("permissive_terminal_literal", terminal_literal),
        ("permissive_leading_label_text", leading_label_text),
        (
            "permissive_boundary_choice_text",
            _boundary_choice_text_evidence(first_line, last_line, entries),
        ),
    )
    for source, evidence in tiers:
        result = _resolve_evidence(evidence, source, entries, allowed_labels)
        if result is not None:
            return result
    return _strict_result(strict)


parse_mcq_answer_permissive = parse_permissive_mcq_answer
