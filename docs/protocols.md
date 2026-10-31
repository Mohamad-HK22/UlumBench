# Evaluation protocols

All configurations are for prospective evaluation of a newly specified model. No historical models or response caches are bundled. The schedule is a seeded topological shuffle (seed 314159) so conditions are interleaved while generated-history dependencies precede their consumers. There is one generation per scheduled call and no answer-based retries.

| Protocol | Panel and design | Generation settings |
| --- | --- | --- |
| State assistance | 100 eight-step families; eight atomic calls, four oracle-assisted prefixes and four composed prefixes per family | One repetition; low effort; 8,192-token output cap; temperature omitted |
| Dependency depth | 500 families at native lengths; nested horizons 2/4/6/8 for the 100 eight-step families; atomic, independent and composed conditions | Three repetitions; low effort; 8,192-token cap; temperature omitted; native JSON schema |
| Error recovery | 100 eight-step families × positions 2/4/6 × four arms; all 300 frozen injections retained | Three repetitions; low effort; 8,192-token cap; temperature omitted |
| Reasoning effort | Fixed 658-question computer-science panel; low/medium/high effort | One repetition per effort; 16,384-token cap; temperature omitted |

State assistance supplies direct reference predecessors to atomic and oracle-assisted requests. Composed requests supply operations and shared context without predecessor references. Atomic requests are grouped by their step position in summaries; oracle and composed requests by horizon. The 11 flagged families remain in the full panel and their flags remain in request records for sensitivity analysis.

Dependency-depth matching retains the original outcome-blind algorithm: blocks of 10 families within native length, sorted by terminal answer class, topic and family ID; matching seed 271828, adjusted by length and block offset. Matching avoids repeated families within an independent bundle and penalizes predecessor-answer collisions, answer-type mismatch and topic mismatch. Every operation occurs equally often in the composed and independent pools at each block/horizon. Independent bundles change context and topic composition as well as dependency structure. Main and shorter-chain panels are reported separately.

Error-recovery arms are A: correct state/continue; B: wrong state/continue; C: correct state/verify; D: wrong state/verify. The original 100-family schedule is retained; the primary panel flags exclude one family, giving 99 families and 3,564 candidate calls. State repair, terminal correctness and consistency with a wrong-state trajectory are distinct measurements. Correct final answers alone do not establish detection or repair. The frozen corrected injection and washout flags are carried in the dataset.

Reasoning-effort settings are provider controls. Their actual effect should be checked using returned reasoning-token telemetry. The summary reports telemetry coverage and medians but does not automatically certify manipulation checks, perform inferential tests or claim equal compute across model families. The fixed panel was originally selected using earlier model performance; its selection is not independent of those original models.

## Execution differences from the research archive

The cleaned package uses one explicit OpenRouter client and a common durable journal instead of experiment-specific launchers. It preserves the model-facing task prompts and core answer graders. The new release uses a common strict output contract: malformed response envelopes, duplicate JSON keys, refusals and length-truncated responses cannot receive successful overall scores. It records failures explicitly and does not reproduce historical exception or retry policies. A single model/provider is requested with provider fallback disabled; actual returned route/model metadata are saved when available.

The code does not execute submitted code answers, give models external tools, reconstruct paper plots, replay normalization, recreate historical confidence intervals or reproduce cross-model consensus analysis. Summaries are descriptive; use the saved family, matching-block, group and repetition identities for any subsequent statistical analysis.

API fields follow the [OpenRouter reference](https://openrouter.ai/docs/api/reference/overview). Provider support and pricing can change; inspect a small new-model pilot before starting a full run.
