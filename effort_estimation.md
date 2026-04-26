# Assignment 2 (RAG) - Effort Estimation

Estimates for each task in the RAG assignment. The "manual" column assumes a competent grad student familiar with Python but new to RAG/Ragas. The Opus and Sonnet columns assume you're driving the model (reviewing diffs, pasting errors, re-running cells) - runtime for embedding and API calls is a floor that doesn't shrink with model choice, so a lot of the Opus/Sonnet figures are wall-clock, not "model thinking" time.

## Per-task estimates

| # | Task | Manual | Opus | Sonnet | What drives the spread |
|---|---|---|---|---|---|
| 1 | Naive generation (10 Qs + xlsx + discussion) | 1.5-2 h | 20-30 min | 30-40 min | Manual: Nebius client setup, dataset filtering (drop metrics-generated, dead URL fix), hand-judging verdicts. Model-driven: boilerplate is trivial; hand-judging still on you either way (~15 min). |
| 2 | RAG theory write-up | 30-45 min | ~10 min | ~15 min | Short prose; no code. |
| 3 | Embed documents (PyPDF → chunk → FAISS/BGE + spot-check) | 2-3 h | 30-45 min | 45-60 min | Metadata plumbing (0-indexed page_number), BGE model download, `.save_local`. Embedding runtime itself (~5-10 min) is constant. |
| 4 | `answer_with_rag` pipeline + prompts | 1-1.5 h | ~20 min | ~30 min | Prompt design + empty-retrieval handling. Opus tends to get prompt/citation format right first try. |
| 5 | Run & compare 10 Qs + xlsx + discussion | 1-1.5 h | ~30 min | 30-40 min | Mostly narrative/analysis. Manual judgment dominates in all modes. |
| 6 | Evaluation (LLM judge + Ragas faithfulness + page-hit@k, **full dataset**) | 3-4 h | 1-1.5 h | 1.5-2.5 h | Ragas + Nebius via `llm_factory` with `AsyncOpenAI` is the known pain point. Ragas runtime on 20 examples is real (10-20 min). Full-dataset correctness/page-hit on ~100+ Qs also adds wall-clock. |
| 7 | Improvement cycles (≥3 experiments, incl. re-embed + reranker) | 4-6 h | 1.5-2 h | 2-3 h | Re-embedding per chunk size is runtime-bound. Reranker (bge-reranker-base) wiring + download. Each experiment re-runs full-dataset eval. Hypothesis/interpret prose on you. |
| B | Bonus - multi-scale chunking (2-3 indices, page-hit@5) | 1.5-2 h | 30-45 min | 45-60 min | Mostly re-using Task 3 + Task 6 scaffolding at new chunk sizes. |
| | **Total (with bonus)** | **~15-20 h** | **~5-7 h** | **~7-10 h** |  |
| | **Total (without bonus)** | **~13-18 h** | **~4.5-6 h** | **~6-9 h** |  |

## Notes on the spread

- **Runtime floor (same for all modes):** ~30-60 min cumulative across embedding runs, Ragas calls on 20 examples (≈10-20 min per experiment), full-dataset correctness (≈5-10 min per experiment × ~4 experiments), and reranker download. This is why Opus doesn't collapse total time to under ~4 h.
- **Opus vs Sonnet gap is concentrated in Task 6 & 7.** The Ragas + Nebius async wiring and the reranker integration are where Sonnet typically needs an extra debug loop or two. Tasks 1-5 are close to a wash.
- **Manual judgment doesn't go away.** Verdict labels for Task 1/5 and hypothesis/interpretation prose for Task 7 are yours regardless of model. Budget ~1 h of your own time for these even in the fastest mode.
- **Biggest variance items:** Task 6 (Ragas quirks) and Task 7 (number of experiments you actually run - picking 3 vs 5 is a big delta).
