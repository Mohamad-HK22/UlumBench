# Academic, answer-format and context experiments

This contribution adds three experiment runners to the shared UlumBench package. The dataset is already hosted on Hugging Face; no question or gold-answer files belong in this contribution.

| Experiment | Full candidate calls per model | Design |
| --- | ---: | --- |
| `academic` | 6,167 | 4,158 MCQs with a label-only instruction; 2,009 open-ended final-answer questions |
| `answer_format` | 8,316 | Baseline and label-only calls for each MCQ; the baseline is also rescored by a gold-blind permissive parser without another generation |
| `context` | 5,064 | A/B on 1,185 later-part targets, C on 1,102 independently answerable targets, D on 1,592 trajectory calls including 1,081 scored targets |

All use one repetition, low effort, temperature 0 and a 2,048-token output cap. Add an explicit judge model for academic or context runs with open-ended targets. The shared scorer always sends the canonical question and images to the judge, even when the model saw a different context.

Context A retains the original supplied history, B removes earlier answer spans while retaining earlier parts, C keeps the target and available shared context/images, and D substitutes the same model's earlier generated answers into retained answer slots. The D schedule respects dependencies and retains unscored prerequisite calls. A/B/D share 1,081 history-bearing primary targets; the full A/B and C subsets have different eligibility and should not be compared as if they were identical. The available C eligibility labels come from deterministic automated review rules; the release makes no claim of independent human adjudication.

A sample plan:

```bash
ulumbench plan --config configs/context.json --data datasets/ulumbench --output plans/context
```

For a live evaluation, replace `plan` with `run`, use a new output folder and add `--model PROVIDER/MODEL --judge-model PROVIDER/JUDGE_MODEL`.

The permissive parser requires a unique supported answer signal and rejects conflicting selections. It never sees the reference target during extraction. Its accuracy is reported beside the strict baseline, while the label-only arm is strict-scored.
