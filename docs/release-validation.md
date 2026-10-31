# Release validation

The release was verified locally without paid model calls.

- All 6,295 original academic records retain their original question, solution, topic, type and asset metadata exactly. All 605 image files retain their original bytes.
- All 500 diagnostic families retain their original scientific content: source IDs, context, operations, dependencies, answers and tolerances. All 2,400 reference step answers pass the retained typed scorer.
- The default academic panel has 6,167 questions, including 4,158 MCQs, 2,009 open-ended questions, 140 topics and 743 image-supported questions.
- The four initial experiment schedules have 1,600 / 12,000 / 3,600 / 1,974 candidate calls. Full reference-response scoring passed for every diagnostic request shape and every distinct diagnostic request in one repetition.
- The three separately contributed schedules have 6,167 / 8,316 / 5,064 candidate calls. Generated-history prerequisites are ordered correctly, with 1,081 scored D targets and 1,102 C targets.
- Tests cover malformed and duplicate JSON, task order, numeric tolerances, units, leading zeros, MCQ ambiguity, refusals/truncation, interrupted calls, cached generations and changed run identities.
- Academic, academic-full and diagnostic JSON files load through the Hugging Face `datasets` library with the expected row counts.

To run the bundled tests and optional full-data checks:

```bash
python -m unittest discover -s tests -v
ULUMBENCH_DATA=datasets/ulumbench python -m unittest discover -s tests -v
```

The initial repository has 18 tests, including two optional full-data tests. Adding the separate contribution yields 21 tests, including its context integration test. All 21 passed together against the full exported data. These checks validate software and data integrity, not current provider availability, model accuracy, the scientific validity of every source reference, or historical paper statistics.
