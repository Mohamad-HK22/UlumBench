<p align="center">
  <a href="https://ulumbench.com"><img src="branding/ulumbench.png" width="480" alt="UlumBench logo"></a>
</p>

<h1 align="center">UlumBench</h1>
<p align="center"><strong>A Benchmark for Evaluating Academic and Multi-Step Problem Solving in LLMs</strong><br>MusIML Workshop · NeurIPS 2026</p>
<p align="center">
  <a href="https://ulumbench.com">Website</a> ·
  <a href="https://openreview.net/forum?id=XwNQ9m9mem">Paper</a> ·
  <a href="https://huggingface.co/datasets/MHK-22/UlumBench">Dataset</a> ·
  <a href="LICENSE">MIT license</a>
</p>

UlumBench connects university-level academic problem solving with controlled tests of multi-step execution. Its academic evaluation panel contains **6,167 questions across 140 topics**, including diagrams and other image-supported inputs. A separate collection contains **500 source-linked diagnostic families**, with two to eight operations and reference answers for intermediate states.

This repository evaluates **new models**. The questions, answers, images and experimental metadata are hosted on Hugging Face. Historical model responses, failed runs and paper-analysis outputs are excluded.

## Experiments

| Experiment | What it evaluates | Full candidate calls per model |
| --- | --- | ---: |
| `state_assistance` | Atomic operations, oracle-assisted bundles and composed prefixes on 100 eight-step families | 1,600 |
| `dependency_depth` | Matched independent versus dependent operation pools across all 500 families; three repetitions | 12,000 |
| `error_recovery` | Continue or verify correct and perturbed intermediate states; three repetitions | 3,600 |
| `reasoning_effort` | Low, medium and high effort on a fixed 658-question computer-science panel | 1,974 + open-ended grading |

The academic, answer-format and context experiment runners will be contributed separately. `ulumbench list` shows what is installed. Shared academic scoring is included because the reasoning-effort experiment uses it.

## Install

Python 3.11 or newer is required.

```bash
git clone https://github.com/Mohamad-HK22/UlumBench.git
cd UlumBench
python -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-tested.txt .
```

On Windows, activate with `.venv\Scripts\activate`. The dependency ranges are in `pyproject.toml`; `requirements-tested.txt` records the versions used for release verification.

## Download the dataset

```bash
ulumbench download --output datasets/ulumbench
ulumbench validate --data datasets/ulumbench
```

The downloader uses a pinned Hugging Face commit. Local files are checked against the dataset manifest before evaluation. A different revision can be selected explicitly with `download --revision COMMIT`.

## Plan an evaluation

Generate and inspect the call plan before using an API key. This command makes no model calls:

```bash
ulumbench plan --config configs/state_assistance.json \
  --data datasets/ulumbench --output plans/state-assistance
```

The plan reports the number of candidate calls and the maximum number of judge calls. `--limit 4` selects a small smoke test; its results are marked partial and must not be reported as the full benchmark.

## Evaluate a new model

Set `OPENROUTER_API_KEY` in your environment, then supply an exact model ID supported by your provider:

```bash
ulumbench run --config configs/state_assistance.json \
  --data datasets/ulumbench --model PROVIDER/MODEL \
  --output runs/state-assistance-my-model
```

For `reasoning_effort`, add `--judge-model PROVIDER/JUDGE_MODEL` to grade open-ended answers. The selected model must support the configured reasoning effort and, for academic images, image input. The dependency-depth configuration also requires native JSON-schema output. Unsupported parameters fail explicitly. An exact candidate provider endpoint can be selected with `--provider ENDPOINT_SLUG`.

The full commands incur provider charges. No model evaluations or paid API calls were made during release validation.

## Results and resuming

Each run saves its identity, request plan, provider-attempt journal, scores and descriptive summary. Repeat the **same command** to resume: completed generations and judge calls are reused. A changed code version, runtime dependencies, dataset, configuration, model or selection requires a new output directory. Transport failures stop the run; unresolved attempts require an explicit `--retry-errors` after checking whether the earlier attempt was charged.

```bash
ulumbench summarize runs/state-assistance-my-model
python -m unittest discover -s tests -v
```

Refusals, truncated answers and malformed requested output count as unsuccessful answers. Provider errors and invalid judge output leave the run incomplete. Missing telemetry stays missing. Reasoning effort is an ordinal control within a model, not a promise of equal computation across models.

## Scoring and interpretation

Multiple-choice answers use curated targets and a deterministic parser. Other academic answers use a configurable binary reference-answer judge. Diagnostic answers use the supplied typed scorer, with stored numeric tolerances and declared unit aliases. Terminal correctness and complete-sequence correctness are reported separately.

This release retains the repository's reference-format diagnostic scoring. It does **not** implement the separately conducted representation-normalization replay or reproduce historical paper tables. Newly generated results should identify this scorer version and the precise data revision. See [protocols](docs/protocols.md), [scoring](docs/scoring.md) and [data](docs/data.md) for details and limits.

## Citation and licenses

Please cite the [UlumBench paper](https://openreview.net/forum?id=XwNQ9m9mem) when using the benchmark. Repository citation metadata is provided in [CITATION.cff](CITATION.cff).

Code is released under the [MIT License](LICENSE). The Hugging Face dataset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), subject to the source-material notice in its dataset card. The UlumBench name and logo identify the project; the software license does not grant trademark rights.
