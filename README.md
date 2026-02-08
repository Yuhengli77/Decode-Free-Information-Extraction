# Decode-Free Web Extraction for Search Agents

This project studies **index-based (decode-light) web evidence extraction** as a low-latency alternative to **generative summarization** in search agents.  
Instead of generating long summaries token-by-token, the extractor outputs **short indices/spans** over a block-indexed webpage, and a deterministic post-processor reconstructs the extracted evidence.

We (1) reproduce the core pipeline from the reference work, (2) improve extraction behavior via **reinforcement learning / preference optimization** with **agent-utility–oriented rewards**, and (3) integrate the module into a simple search-agent workflow to measure **end-to-end latency vs task quality**.

>
> **Terminology note**: the referenced paper names its method **Index-based Web Content Extraction**.  
> In this repo, we use **decode-free (decode-light)** as an umbrella engineering term. When citing that paper directly, we keep the original method name to avoid ambiguity.

---

## Papers That Inspired This Project

- **Chen et al. (2025)**, *An Index-based Approach for Efficient and Effective Web Content Extraction* (arXiv:2512.06641).  
  Core inspiration for reframing extraction as **index prediction + deterministic reconstruction** rather than long-form generation.  
  Link: https://arxiv.org/html/2512.06641v1


---

## Project Goals

### 1) Reproduction (baseline)
- Reproduce webpage preprocessing → block indexing → index-based extraction → reconstruction.
- Establish latency breakdown for extraction:
  - **Generative summary** (slow due to long decoding) vs
  - **Index output + reconstruction** (fast due to short outputs + Python post-processing)

### 2) RL / Preference Optimization (improvement)
- Train the index extractor with **agent-utility–aligned objectives**, e.g.:
  - evidence sufficiency for downstream answering/action
  - robustness to boilerplate/navigation noise
  - coherence/compactness of selected spans (without relying on long-form generation)

### 3) Agent Integration & Evaluation
- Plug the extractor into a lightweight search agent (retrieve pages → extract evidence -> action → answer).
- Evaluate:
  - **Task quality** (e.g., EM/F1 / success rate)
  - **Latency** (end-to-end + breakdown: fetch / extract / reason)

---

## System Achitecture

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                         INPUT: Raw HTML Page + URL/Title + User Query                        │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                             DECODE-FREE EXTRACTION PIPELINE                                  │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│  1) HTML CLEANING + CHUNKING + BLOCK INDEXING                                                │
│     ┌─────────────────────────────────────────────┐   ┌───────────────────────────────────┐  │
│     │ Noise Removal                               │   │ Chunk & Index Builder             │  │
│     │ - remove script/style/noscript              │   │ - split into blocks               │  │
│     │ - normalize whitespace                      │   │ - assign block ids (0..N-1)       │  │
│     └─────────────────────────────────────────────┘   └───────────────────────────────────┘  │
│                                      │                                                       │
│                                      ▼                                                       │
│     page_payload = {url, title, indexed_blocks: [{"id": i, "text": ...}, ...]}               │
│                                                                                              │
│  2) QUERY-AWARE INDEX EXTRACTION  (main/index_extractor.py)                                  │
│     ┌─────────────────────────────────────────────┐   ┌───────────────────────────────────┐  │
│     │ Prompt Construction                         │   │ LLM Response Parser               │  │
│     │ - QE mode: query provided                   │   │ - parse JSON / fenced JSON        │  │
│     │ - ME mode: query empty (main extraction)    │   │ - expect "block_intervals"        │  │
│     └─────────────────────────────────────────────┘   └───────────────────────────────────┘  │
│                                      │                                                       │
│                                      ▼                                                       │
│     model output: {"block_intervals": [[start_id, end_id], ...]} OR {"block_intervals":"NA"} │
│                                                                                              │
│  3) DETERMINISTIC ID EXPANSION + RECONSTRUCTION  (main/reconstructor.py)                     │
│     - normalize intervals (swap when start > end)                                            │
│     - expand intervals to valid ids, deduplicate, optional top_k                             │
│     - keep page order, rebuild extracted HTML fragment/document                              │
│                                                                                              │
│  4) EVIDENCE FORMATTING  (main/formatter.py)                                                 │
│     - reconstruct extracted HTML                                                             │
│     - convert HTML to markdown evidence                                                      │
│                                                                                              │
│  5) OPTIONAL QA ON EXTRACTED DOCS  (main/answerer.py)                                        │
│     - aggregate extracted markdown docs + query                                              │
│     - output JSON: {"answer": "..."}                                                         │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                    OUTPUT: selected_ids + extracted HTML + extracted Markdown                │
│                          (+ optional final answer from answerer)                             │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```
