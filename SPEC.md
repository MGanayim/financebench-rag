# FinanceBench-RAG - Engineering Specification

Internal engineering contract for the FinanceBench-RAG pipeline. This document is the source of truth for what the implementation must do; the notebook is the source of truth for how it does it. Where the two disagree, this spec wins - update it before changing behavior.

---

## 1. Overview

FinanceBench-RAG is a grounded question-answering pipeline over SEC 10-K filings, benchmarked on the **FinanceBench** dataset. The system indexes a fixed corpus of filings, retrieves evidence for each question, and generates an answer constrained to the retrieved context. Evaluation measures the pipeline on three independent axes - **answer correctness**, **faithfulness to retrieved context**, and **retrieval page-hit@k** - so each component can be diagnosed in isolation.

The target is not a single accuracy number. FinanceBench is adversarial by construction; the deliverable is a defensible diagnosis of where the pipeline fails and a set of experiments that attempt to move the needle on the weakest axis.

---

## 2. Scope & Non-Goals

**In scope**

- Corpus indexing (PDF → chunks → FAISS vector store).
- Dense retrieval with optional cross-encoder reranking.
- Prompt-constrained generation with per-fact citation.
- Three-axis evaluation (correctness / faithfulness / page-hit@k).
- At least three improvement cycles, one variable per experiment.
- Multi-scale chunking bonus study.

**Out of scope**

- UI or HTTP serving layer.
- Multi-tenancy, auth, rate limiting.
- Streaming responses.
- Agentic tool use or multi-hop planning.
- Incremental index updates (the index is rebuilt from scratch).
- Answers outside the FinanceBench domain.

---

## 3. Dataset Contract

**Source.** The [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) dataset on Hugging Face.

**Required transformations before any task consumes the data:**

1. **Drop `question_type == "metrics-generated"`** rows. Keep only `domain-relevant` and `novel-generated`.
2. **Replace dead `doc_link` values** with the mirror supplied in the brief (the `financebench` repo linked from the assignment's dataset notes). The mirror folder contains more documents than the dataset references; embed only the PDFs whose `doc_name` appears in the filtered dataset.
3. **Ignore `dataset_subset_label`**.

**Invariants the pipeline relies on:**

- `evidence_page_num` is 0-indexed in the dataset. The indexing step must keep its `page_number` metadata 0-indexed too.
- `evidence` is a list - a single question can cite multiple pages. Retrieval scoring treats a hit on **any** evidence page as a hit.

---

## 4. Component Specs

### 4.1 Indexing (offline, once per chunk configuration)

| Item | Value |
| --- | --- |
| Loader | `langchain_community.document_loaders.PyPDFLoader`, one `Document` per page |
| Metadata attached before splitting | `{doc_name, company, doc_period, page_number}` with `page_number` **0-indexed** |
| Splitter | `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)` for the baseline; bonus builds at 300 and 2000 |
| Embedding model | `BAAI/bge-small-en-v1.5` via `langchain_huggingface.HuggingFaceEmbeddings` |
| Vector store | LangChain FAISS |
| Persistence | `vectorstore.save_local("indices/faiss_chunk{size}")` |

**Invariants.** Chunks inherit the page's metadata verbatim. Only PDFs whose `doc_name` is in the filtered dataset are loaded. Re-embedding is required whenever `chunk_size`, `chunk_overlap`, or the embedding model changes - each variant is saved under a distinct directory (`faiss_chunk1000`, `faiss_chunk300`, `faiss_chunk2000`) so experiments can swap indices without rebuilding.

### 4.2 Retrieval (per query)

- `vectorstore.similarity_search(query, k=k)` returns a `list[Document]` with full metadata intact.
- For the reranker experiment: retrieve top-20 with FAISS, then rerank with `BAAI/bge-reranker-base` (cross-encoder), keep top-`k` (default 4).
- Empty retrieval is a legitimate state (e.g., if the index is empty or filters exclude everything) and must be handled downstream.

### 4.3 Generation

```python
def answer_with_rag(query: str, k: int = 4) -> dict:
    """
    Returns:
        {
            "answer": str,
            "retrieved_chunks": list[{"text": str, "doc_name": str, "page_number": int}]
        }
    """
```

**Generation model.** `meta-llama/Llama-3.3-70B-Instruct` via Nebius Token Factory. Temperature `0` for reproducibility.

**Prompt construction.**

- Retrieved chunks are formatted with an explicit separator between them and each chunk is labeled with its `doc_name`.
- The user message contains the query followed by the context block.
- If retrieval returns zero chunks, the context block is replaced with an explicit `"No relevant context was found."` sentinel - never an empty block.

**System prompt requirements.** The system prompt must instruct the model to:

1. Answer **only** from the provided context.
2. Say explicitly when the context does not contain the answer, rather than guessing.
3. Keep answers concise and cite the `doc_name` each fact came from.

---

## 5. Evaluation Contract

Three independent measures. Each isolates a different failure mode.

### 5.1 Correctness (full dataset)

- **Judge model.** `deepseek-ai/DeepSeek-V3-0324` via Nebius Token Factory - a different model family from the generator, to avoid self-preference bias.
- **Prompt.** Produces a binary verdict (`correct` / `incorrect`) plus a one-sentence justification.
- **Output column.** `correctness ∈ {0, 1}`.
- **Aggregation.** Mean across the full filtered dataset.

### 5.2 Faithfulness (20-row subsample)

- **Library.** [Ragas](https://docs.ragas.io/) `Faithfulness` from the collections API.
- **LLM wiring.** `ragas.llms.llm_factory(...)` wrapping `AsyncOpenAI(base_url=<Nebius base URL>, api_key=...)` pointed at DeepSeek-V3-0324.
- **Method.** Use `.score()` (sync). Do **not** use `.ascore()`.
- **Sample.** First 20 rows sorted by `financebench_id`. Ragas is slow (multiple LLM calls per sample); 20 is the fixed cap across all experiments so faithfulness numbers remain comparable.
- **Output column.** `faithfulness ∈ [0, 1]`.

### 5.3 Retrieval page-hit@k (full dataset)

- For each question, retrieve top-`k` chunks and record `page_hit@k = 1` if **any** retrieved chunk's `page_number` matches **any** entry in the question's `evidence_page_num` list; else `0`.
- Report `k ∈ {1, 3, 5}`.
- Multi-evidence questions count as a hit on any match (union semantics).

### 5.4 Baseline numbers

The Task 6 numbers (baseline pipeline, no experiments) are the reference row that every Task 7 experiment is compared against. They are written as the first row of `assignment2_improvement_cycles.xlsx`.

---

## 6. Experiment Matrix (Task 7)

Minimum three experiments. One variable changes per experiment; everything else stays at baseline (k=4, chunk_size=1000, no reranker, unchanged prompt and model).

| # | Variable | Values / Change | Notes |
| --- | --- | --- | --- |
| E1 | `k` (retrieval depth) | {1, 3, 5, 8} | k is the number of chunks shown to the generator. |
| E2 | `chunk_size` | {300, 1000, 2000} | Requires re-embedding. Each variant saved under its own `indices/faiss_chunk{size}` directory. |
| E3 | Reranker | Add `BAAI/bge-reranker-base` (retrieve 20 → rerank to 4) | Keep generator `k=4`. |
| E4 *(optional)* | Generation prompt **or** generation model | Stricter citation prompt **or** swap Llama-3.3 for another Nebius model | Exactly one change, not both. |

**Per-experiment deliverables:** hypothesis (1-2 sentences), the change, the three metrics re-run, interpretation (did the predicted metric move? anything else?). Correctness and page-hit@k run on the full dataset; faithfulness stays on the fixed 20-row subsample.

**Wrap-up deliverable:** a paragraph answering "where does the pipeline fail most - retrieval, generation, or both?" and "what would one more week unlock?"

---

## 7. Bonus - Multi-Scale Chunking

Test whether any single chunk size dominates on FinanceBench.

1. Build indices at 2-3 chunk sizes (e.g., `{300, 1000, 2000}`). Embedding model, chunking method, and overlap policy fixed; chunk size is the only variable.
2. For each question, record `page_hit@5` for each index.
3. Report:
   - Summary table: `page_hit@5` per chunk size.
   - **Disagreement rate**: fraction of questions where the best chunk size differs from the overall winner.
   - Short discussion: is there a dominant winner, or is the best size query-dependent?

---

## 8. Deliverables Checklist

Every filename below is grading-sensitive. Do not rename.

| Artifact | Location | Columns / Content |
| --- | --- | --- |
| `assignment2_naive_generation.xlsx` | `artifacts/` | `financebench_id`, `question_type`, `question`, `naive_answer`, `ground_truth`, `verdict ∈ {correct, partially correct, wrong, refused}` |
| `assignment2_run_and_compare.xlsx` | `artifacts/` | `financebench_id`, `question_type`, `question`, `naive_answer`, `RAG_answer`, `ground_truth` |
| `assignment2_evaluation.xlsx` | `artifacts/` | `financebench_id`, `question`, `correctness`, `faithfulness`, `page_hit_at_1`, `page_hit_at_3`, `page_hit_at_5` |
| `assignment2_improvement_cycles.xlsx` | `artifacts/` | `experiment`, `change`, `correctness`, `faithfulness`, `page_hit_at_1`, `page_hit_at_3`, `page_hit_at_5`. Row 1 = Task 6 baseline. |
| Notebook | `notebooks/financebench_rag.ipynb` | Code and markdown for all tasks, outputs present. |
| Markdown discussions | inside the notebook | One per task where the brief asks for it (Tasks 1, 2, 3, 5, 7, Bonus). |
| Headline metrics table | `README.md` §3 | Baseline vs best experiment on all three axes. |

---

## 9. Reproducibility

- **Secrets.** `NEBIUS_API_KEY` loaded from `.env` (copy `.env.example`). Never committed.
- **Determinism.** Retrieval is deterministic. Generation uses `temperature=0`. Ragas and the judge are LLM-based, so their outputs are approximately - not exactly - reproducible; the spread is small enough that conclusions hold across reruns.
- **Rebuilds.** `indices/` is gitignored. The notebook's "Setup" section contains a cell that rebuilds any missing FAISS index from scratch.
- **Model cache.** BGE embedder (~130 MB) and reranker (~1 GB) cache to the default Hugging Face cache; document the cache path in the README quickstart.
- **Runtime floor.** Embedding ~5-10 min; Ragas on 20 rows ~10-20 min per experiment; correctness judge on full dataset ~5-10 min per experiment. Budget accordingly.

---

## 10. Known Pain Points

These have bitten every prior run - the notebook should handle them explicitly, not defensively hope they don't happen.

1. **Ragas + Nebius wiring.** Faithfulness scoring fails silently if the Ragas LLM is not wrapped correctly. Use `ragas.llms.llm_factory(...)` over an `AsyncOpenAI(base_url=<Nebius URL>, api_key=...)` client; verify with a single-row smoke test before running the 20-row batch.
2. **0-indexed pages.** `evidence_page_num` is 0-indexed. `PyPDFLoader` attaches its own `page` metadata field - we ignore it and write our own `page_number` to be explicit. An off-by-one here drops `page_hit@k` to near zero without any visible error.
3. **Metrics-generated questions.** Must be dropped before indexing, evaluation, and experiments. Leaving them in inflates the denominator and distorts every metric.
4. **BGE model download.** First run downloads ~130 MB; later runs hit the HF cache. Slow networks should warm the cache before the notebook runs end-to-end.
5. **Dead `doc_link` URLs.** Replace with the mirror repo supplied in the brief before any download step.

---

## 11. Notebook Structure

The notebook is the single place where code lives. Organize it so a grader can skim it top-to-bottom and see each task's deliverable in place.

```
Setup
  - Imports, env loading, Nebius client
  - Dataset load + filter + URL fix
  - FAISS build helper (used for baseline + bonus chunk sizes)
  - answer_with_rag definition
  - Judge / Ragas / page-hit helpers

Task 1 - Naive Generation
  - Code → assignment2_naive_generation.xlsx → discussion

Task 2 - RAG Theory
  - Markdown only (indexing / retrieval / generation write-up)

Task 3 - Embed Documents
  - Code → spot-check retrievals → discussion

Task 4 - answer_with_rag
  - Code (imported into later tasks) + prompt text

Task 5 - Run & Compare
  - Code → assignment2_run_and_compare.xlsx → discussion

Task 6 - Evaluation
  - Code (correctness + faithfulness + page-hit@k) → assignment2_evaluation.xlsx → aggregates

Task 7 - Improvement Cycles
  - One subsection per experiment: hypothesis → code → metrics → interpretation
  - assignment2_improvement_cycles.xlsx → wrap-up

Bonus - Multi-Scale Chunking
  - Code → summary table → discussion
```

Shared helpers are defined **once** in Setup and reused. No redefinitions in later sections. Each task produces its xlsx via `pandas.DataFrame.to_excel(...)` into `artifacts/` and the markdown discussion lives in the same section as the code that produced it.
