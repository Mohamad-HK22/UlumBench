# Data organization

The canonical release lives at [MHK-22/UlumBench](https://huggingface.co/datasets/MHK-22/UlumBench). Code contains no benchmark questions, gold answers, question images, context manifests or injection records.

| Dataset component | Contents |
| --- | --- |
| `academic/records.jsonl` | All 6,295 records with explicit evaluation eligibility and exclusion reasons |
| `academic/evaluation.jsonl` | The 6,167 eligible records, also exposed as the default academic HF configuration |
| `assets/` | 605 unique PNGs referenced by academic records |
| `diagnostic/families.jsonl` | 500 families and 2,400 reference-scored steps; 100 primary eight-step families |
| `diagnostic/error_injections.jsonl` | 300 frozen perturbations, alternative trajectories and eligibility flags |
| `metadata/` | Curated MCQ targets, neutral provenance, group membership, exclusions, context manifests and the 658-question effort panel |
| `manifest.json` | Per-file SHA-256 hashes and expected counts |

The 6,167 panel excludes 44 input-issue records, 73 reference-review records and 11 further ambiguous MCQs. These disjoint exclusions remain visible in the full collection. The panel contains 4,158 MCQs and 2,009 open-ended questions, including 743 questions with images. No training split is defined.

Stable question and family IDs are preserved. Source-derived group names are replaced with deterministic opaque IDs without changing grouping. Historical outputs and model performance used in panel selection are not redistributed as fresh benchmark labels. Source questions and constructed diagnostics remain distinct data components.

The diagnostic release preserves the scientific content (context, operations, dependencies, references and tolerances), class labels and quality flags. Redundant pre-rendered prompt copies and unsupported per-record human-review assertions are omitted. This package is not an independent re-audit of data construction; separate reviews are not reconstructed. The context eligibility metadata identifies the available deterministic review method explicitly.

Use `ulumbench download` when evaluating images: it retrieves the entire immutable snapshot. The Hugging Face `datasets` academic configurations expose asset paths and choice metadata; their generic JSON loader does not itself decode those nested image references. The evaluator resolves and checks the PNG files locally.
