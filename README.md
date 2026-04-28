# FinanceBench-RAG

*A diagnosed RAG pipeline for SEC 10-K Q&A, benchmarked on FinanceBench.*

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Stack](https://img.shields.io/badge/stack-Nebius%20%C2%B7%20LangChain%20%C2%B7%20FAISS%20%C2%B7%20Ragas-purple)

Most RAG demos report a single accuracy number and stop. This repo takes the opposite approach: a baseline pipeline, three independent evaluation axes, and a set of one-variable-at-a-time experiments designed to tell you *which component* is actually costing you accuracy. The benchmark is [FinanceBench](https://arxiv.org/abs/2311.11944) - SEC 10-K question answering, adversarial by design.

---

## Headline Results

Baseline is a standard dense-retrieval RAG pipeline (k=4, chunk_size=1000, no reranker). "Best" is the configuration from the improvement cycles that wins on correctness without regressing the other two axes.

| Configuration | Correctness | Faithfulness (n=20) | page-hit@1 | page-hit@3 | page-hit@5 |
| --- | :---: | :---: | :---: | :---: | :---: |
| Naive generation (no retrieval, n=10 manual) | 0.20 | - | - | - | - |
| RAG baseline (k=4, chunk_size=1000, no reranker) | 0.28 | 0.79 | 0.20 | 0.33 | 0.40 |
| Best experiment (E1, k=8) | **0.34** | 0.75 | 0.20 | 0.33 | 0.40 |

*Naive generation correctness counts only `verdict == "correct"` (2 of 10) - the other 8 split as 6 refused, 1 partially correct, 1 wrong. Best experiment is E1 at k=8 (more retrieved chunks fed to the generator); the retriever and index are unchanged from baseline so page-hit columns match. Per-question data lives in `artifacts/assignment2_evaluation.xlsx` and `artifacts/assignment2_improvement_cycles.xlsx`.*

---

## Architecture

![architecture](docs/architecture.png)

The pipeline is three boxes, each of which is independently measurable:

- **Indexing** *(offline)*: PDFs are loaded page-by-page with `PyPDFLoader`, tagged with 0-indexed `page_number` metadata, split with `RecursiveCharacterTextSplitter`, embedded with `BAAI/bge-small-en-v1.5`, and persisted as a FAISS index.
- **Retrieval** *(per query)*: dense similarity search against FAISS, optionally reranked by `BAAI/bge-reranker-base`.
- **Generation** *(per query)*: retrieved chunks are formatted with `doc_name` labels and passed to `meta-llama/Llama-3.3-70B-Instruct` via Nebius Token Factory with a system prompt that forbids answering outside the retrieved context.

---

## Quickstart

```bash
git clone <this-repo>
cd financebench-rag
pip install -r requirements.txt
cp .env.example .env        # then fill in NEBIUS_API_KEY
jupyter lab notebooks/financebench_rag.ipynb
```

Then **Run All**. First-run cost is dominated by:

- Hugging Face model downloads (~130 MB for the embedder, ~1 GB for the reranker)
- Embedding the corpus (~5-10 min)
- Ragas faithfulness on 20 rows per experiment (~10-20 min per experiment)
- Correctness judge across the full dataset (~5-10 min per experiment)

A cold end-to-end run sits around 45-60 min. Subsequent runs reuse the FAISS index under `indices/` and the Hugging Face cache.

---

## What's Inside

- **[notebooks/financebench_rag.ipynb](notebooks/financebench_rag.ipynb)** - the full pipeline. Setup helpers at the top, one section per task below. All code lives here.
- **[SPEC.md](SPEC.md)** - engineering contract. Start here if you want to understand *what* the pipeline is required to do (separate from *how* the notebook does it).
- **[artifacts/](artifacts/)** - the graded deliverables (naive generation, run-and-compare, full evaluation, improvement cycles) as `.xlsx` files.
- **[indices/](indices/)** - saved FAISS stores per chunk size. Gitignored; rebuilt from the notebook's Setup section.
- **[docs/architecture.png](docs/architecture.png)** - the three-box diagram shown above.

---

## Evaluation Methodology

Three axes, chosen because they fail independently. A single correctness number tells you the pipeline is broken; it can't tell you *where*.

| Axis | What it measures | Why it's here |
| --- | --- | --- |
| **Correctness** | Does the final answer match ground truth? Binary verdict from DeepSeek-V3.2. | End-to-end quality - what a user actually feels. |
| **Faithfulness** | Does the answer stay within the retrieved context? Ragas `Faithfulness`. | Catches hallucination even when the answer looks right. Evaluated on a fixed 20-row sample (Ragas is slow). |
| **Retrieval page-hit@k** | Did retrieval surface the page cited as evidence, for `k ∈ {1,3,5}`? | Isolates retrieval. If page-hit is low, no amount of prompt engineering will save you. |

See [SPEC.md §5](SPEC.md#5-evaluation-contract) for the exact protocol, including the Ragas + Nebius wiring.

---

## Experiments & Findings

Task 7 runs at least three one-variable-at-a-time experiments against the Task 6 baseline. Each varies exactly one of: `k`, chunk size, reranker, generation prompt, or generation model.

All deltas are vs the RAG baseline (correctness 0.28, faithfulness 0.79, page-hit@5 0.40). Each experiment row reports the best variant; full per-variant numbers live in `artifacts/assignment2_improvement_cycles.xlsx`.

| Experiment | Change | Correctness Δ | Faithfulness Δ | page-hit@5 Δ |
| --- | --- | :---: | :---: | :---: |
| E1 - k sweep (best: k=8) | k ∈ {1, 3, 5, 8} | **+0.06** | -0.04 | 0.00 |
| E2 - chunk size (best: 2000) | 300 / 1000 / 2000 | -0.01 | -0.04 | -0.06 |
| E3 - reranker | + `bge-reranker-base` (20 → 4) | +0.03 | -0.10 | -0.10 |

**Where does the pipeline fail most?** Retrieval is the binding constraint: baseline page-hit@5 is 0.40, so 60 of 100 questions have no path to a correct answer at k=5. E1's monotonic correctness gain (k=1: 0.15 → k=8: 0.34) shows the generator does extract better answers when given more context, but at k=8 correctness still lands at 0.34 against a hit@8 ceiling of 0.45 - leaving ~11 questions where the right page was retrieved but the system answered wrong. Both axes have headroom; retrieval sets the harder upper bound. The E3 paradox (page-hit dropped, correctness rose) also exposes a measurement gap: 10-Ks repeat the same numbers across summary tables and footnotes, so a chunk pulled from a non-labeled page can still answer correctly even though page-hit scores it as a miss.

**Bonus - multi-scale chunking:** chunk_size=1000 wins on `page_hit@5` (0.40 vs 0.35 for 300 and 0.34 for 2000). Of 53 "covered" questions (at least one chunk size hit), 13 have a per-question best chunk size that differs from the overall winner - **disagreement rate 13/53 = 0.245**. Dominant on average but partly query-dependent; full discussion in the notebook's bonus section.

---

## Limits & Honest Caveats

- **FinanceBench is hard on purpose.** The paper reports that state-of-the-art commercial systems struggle on it. Absolute numbers here will look modest; treat them as a baseline for *your own* improvements, not as a leaderboard.
- **Faithfulness sample is small.** Ragas runs ≥1 LLM call per sample; at 20 samples this is already 10-20 min per experiment. Confidence intervals on the faithfulness column are wide.
- **No UI, no server.** This is a notebook + a saved index. Serving is roadmap territory.
- **No incremental ingest.** Changing the corpus means rebuilding the index.
- **Citation granularity is page-level.** The system cites `doc_name + page_number`, not clause spans. Good enough for 10-Ks; not good enough for contract review.
- **Retrieval is dense-only at baseline.** BM25 fallback / hybrid retrieval is on the roadmap, not the baseline.

---

## Roadmap

Ordered by expected impact on the axes that currently hurt most:

1. **Hybrid retrieval (BM25 + dense, reciprocal rank fusion).** FinanceBench questions contain rare tickers and dollar figures that dense embeddings blur; a lexical channel should recover them.
2. **Query rewriting / HyDE.** Novel-generated questions often phrase things a filing never would; rewriting the query into filing-ese before retrieval is cheap.
3. **Clause-level citations.** Extend the answer schema to return `(doc_name, page_number, char_span)` so the UI can highlight the source sentence.
4. **Stronger embedder.** Swap `bge-small` for `bge-large` or `e5-mistral` and measure - the ceiling on retrieval is probably higher than the current setup suggests.
5. **Streamlit demo.** Thin wrapper over the saved FAISS index so the pipeline is usable without Jupyter.
6. **Eval harness + regression CI.** Nightly run over a sampled subset, with a GitHub Action that fails the PR if any axis regresses by more than a threshold.

---

## Credits

- **Dataset:** [FinanceBench: A New Benchmark for Financial Question Answering](https://arxiv.org/abs/2311.11944) - Islam et al., Patronus AI.
- **Models:** `meta-llama/Llama-3.3-70B-Instruct` and `deepseek-ai/DeepSeek-V3.2` served via [Nebius Token Factory](https://nebius.ai/).
- **Libraries:** [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss), [Ragas](https://docs.ragas.io/), [Hugging Face](https://huggingface.co/) (`BAAI/bge-small-en-v1.5`, `BAAI/bge-reranker-base`).
- Originally built as coursework for the **Nebius Academy - From AI Model to AI Agent** course.

---

## License

MIT.
