# Scoring contracts

## Academic answers

Multiple-choice scoring uses the curated target, allowed answers and ordered choices on Hugging Face. The parser accepts supported label forms, True/False literals and uniquely matching choice text, while rejecting ambiguity and conflicting selections. Choice-text targets are resolved through the selected choice. Native `evaluation_method` annotations are descriptive; every eligible multiple-choice record follows this parser and every other academic record follows binary judging.

The open-ended judge receives the canonical original question and images, reference answer and candidate answer, with an instruction to treat them as data. It must return exactly `CORRECT` or `INCORRECT`, at temperature 0 with an eight-token output cap. Supply and report the judge ID explicitly. Invalid judge output leaves the run incomplete and is saved for inspection; it is not silently converted to a wrong candidate answer. `--retry-errors` can retry an invalid judge response without repeating the candidate generation. Valid correctness verdicts are never retried.

## Diagnostic answers

The typed grader retains the source implementation: scalar decimal tolerance is absolute; integers must be integral; ordered structured values are parsed as Python literals and compared exactly; sets are order independent; bit strings preserve leading zeros; symbolic state normalization removes whitespace. Unit aliases are enumerated in `scoring/units.json`; arbitrary unit conversion is not performed. Fraction syntax, semantically equivalent alternative representations and coordinate reordering are not generally accepted.

The requested JSON envelope is validated separately from semantic answer comparison. For state assistance, `final_answer` and every intermediate step are checked separately; `whole_correct` requires the full output to be correct. Invalid envelopes gate overall correctness even if some extracted individual strings match references. Duplicate keys are invalid. For dependency depth, task IDs must appear in the requested order. For error recovery, suffix step IDs must appear in order and all answers must be nonempty strings.

This is the **reference-format scorer**, not the separately performed representation-normalization analysis. Do not label its scores as normalized paper results. Preserve per-step details to inspect representation-sensitive failures.

## Missingness and failure

Candidate refusals, non-stop finishes (including truncation) and malformed answers count as failures. Infrastructure errors stop execution and do not enter scored denominators. An interrupted group's accuracy is withheld until all selected calls are scored. A deliberately limited selection is labeled partial even when all of its selected calls finish.

Only final answer text and provider metadata are saved; provider reasoning content is excluded. Returned token counts, route, generation ID and cost are retained when available. Missing cost or reasoning telemetry is not interpreted as zero.
