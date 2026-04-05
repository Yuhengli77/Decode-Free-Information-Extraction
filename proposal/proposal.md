# Single-Pass Evidence Extraction via Hidden State Classification

## Research Proposal

### 1. Background and Motivation

Evidence selection — given a query and a set of candidate text passages, identify which passages are relevant — is a core component of RAG pipelines, open-domain QA, and multi-hop reasoning systems. Prior approaches fall into three categories, each with notable limitations:

| Method | Approach | Key Limitation |
|--------|----------|----------------|
| Dual-Tower | Encode query and passages independently, rank by cosine similarity | No cross-attention between query and passage; no inter-passage interaction; fast but weak expressiveness |
| Cross-Encoder | Concatenate query with **each** passage, score independently | Rich query-passage interaction, but **zero inter-passage interaction**; requires N forward passes. Could concatenate all passages, but encoder context windows (e.g., BERT's 512 tokens) are too small to fit them |
| LLM Generation | Prompt an LLM to judge each passage's relevance via generated text | Powerful reasoning, but **autoregressive decoding** is slow — even outputting a single "yes/no" per passage requires sequential token generation across N passages |

The first two approaches share a fundamental limitation: **passages are scored in isolation**. For multi-hop questions where the relevance of one passage depends on information in another, independent scoring is theoretically insufficient. The third approach can leverage LLM reasoning but pays a heavy cost in decoding latency.

#### Key Observation

Encoder-based models (cross-encoders) excel at discriminative scoring but are constrained by short context windows, preventing joint processing of all passages. LLMs offer large context windows (32K+ tokens) that easily accommodate a query and all candidate passages in a single sequence, but their standard usage requires autoregressive generation to produce judgments — even reducing the output to passage indices only alleviates, not eliminates, the decoding overhead.

**Our insight:** we can combine the best of both worlds — use the LLM's large context window to jointly process all passages in a single prefill pass, then directly classify each passage from its hidden state representation, completely bypassing autoregressive decoding. Concretely, both Qwen3-0.6B and Qwen3-Embedding-0.6B support 32K-token contexts, which is large enough to fit a query and all candidate documents at once — far exceeding the context limits of encoder models (512 tokens for BERT-family, up to 8K for recent models like ModernBERT).

### 2. Proposed Method: Hidden State Classification

#### 2.1 Architecture

Given a user query Q and N candidate passages, the input sequence is:

```
Question: {query} \n\n Passage 1 | Title: ... \n {text} \n [EOS] Passage 2 | ... [EOS] ... Passage N | ... [EOS]
```

A single forward pass produces hidden states at every position. For each passage i, we take the hidden state of its **last token (EOS)** as the passage representation, and apply a binary classification head:

```
label_i = sigmoid(W · h(EOS_i) + b)  →  0 (irrelevant) or 1 (relevant)
```

We investigate two backbone variants:

**Variant A — Causal LM (Qwen3-0.6B):** The causal attention mask is kept intact. Each passage's EOS token attends to the query and all preceding passages, but **not** to passages appearing after it. This provides **unidirectional** inter-passage interaction, leveraging the LLM's general reasoning capabilities from next-token-prediction pretraining.

**Variant B — Bidirectional Encoder (Qwen3-Embedding-0.6B):** Built from the same Qwen3 architecture but with unlocked bidirectional attention, this model allows each passage's EOS token to attend to all other passages in both directions, providing **full** inter-passage interaction. Note that this model has a different pretraining objective (contrastive learning for embeddings), which may affect the nature of its hidden state representations.

#### 2.2 Training

**Frozen backbone:** In the primary experiments, the LLM backbone is frozen. Only the classification head (W, b) is trained with binary cross-entropy loss:

```
L = - Σ_i [ y_i · log(p_i) + (1 - y_i) · log(1 - p_i) ]
```

This tests whether pretrained hidden states already encode sufficient relevance information without any fine-tuning, using the simplest possible classifier to isolate the contribution of the backbone's representations.

**Fine-tuned backbone:** To further explore the potential of the hidden state classification approach, we additionally experiment with LoRA and full fine-tuning of the LLM backbone. After establishing the frozen-backbone baseline, these experiments investigate how much task-specific adaptation can improve evidence selection performance.

### 3. Baselines

**Dual-Tower (Embedding Similarity):** Encode the query and each passage independently using a sentence embedding model (`all-MiniLM-L6-v2`). Score each passage by cosine similarity with the query. This is the fastest method but has no cross-attention between query and passage, and no inter-passage interaction at all.

**Cross-Encoder:** Concatenate the query with each passage individually and feed into a pretrained cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2`). This provides rich query-passage interaction through bidirectional attention, but each passage is scored independently — there is no inter-passage interaction.

### 4. Experiments

#### 4.1 Dataset: HotpotQA (Distractor Setting)

Each example contains a multi-hop question and 10 paragraphs (2 gold + 8 distractors), with sentence-level supporting fact annotations. We convert to paragraph-level binary labels: a paragraph is labeled 1 if it contains at least one supporting fact, 0 otherwise.

This dataset is ideal because multi-hop questions naturally require cross-paragraph reasoning, directly testing whether inter-passage interaction improves evidence selection.

#### 4.2 Core Comparison

| Method | Query-Passage Interaction | Inter-Passage Interaction | Decoding Cost |
|--------|--------------------------|--------------------------|---------------|
| Dual-Tower | None (independent encoding) | None | None (similarity only) |
| Cross-Encoder | Full (bidirectional) | None | None (score output) |
| LLM Generation | Full (causal) | Unidirectional | O(T) autoregressive tokens |
| Ours (Causal LM) | Full (causal) | Unidirectional (left-to-right) | None (hidden state classification) |
| Ours (Bidirectional) | Full (bidirectional) | Full (bidirectional) | None (hidden state classification) |

#### 4.3 Fine-Tuning Experiments

After establishing results with frozen backbones, we conduct additional experiments:

- **LoRA fine-tuning:** Low-rank adaptation of the causal LM backbone with the classification head, to evaluate whether task-specific adaptation of the hidden states improves evidence selection.
- **Full fine-tuning:** End-to-end training of the backbone and classification head, representing the upper bound of the hidden state classification approach.

#### 4.4 Evaluation

- **Metrics:** Paragraph-level F1, Precision, Recall
- **Breakdown by question type:** Bridge vs. comparison questions, to analyze where inter-passage interaction and LLM reasoning capabilities matter most
- **Threshold tuning:** Optimal classification threshold is selected on the validation set and applied to the test set

### 5. Summary

We propose a method that leverages LLM hidden states for single-pass evidence extraction, combining the large context window of LLMs with the efficiency of direct classification — no autoregressive decoding required. By classifying frozen hidden states from the EOS token of each passage, we test whether pretrained LLM representations already encode sufficient information for relevance judgments. We compare causal (unidirectional) and bidirectional backbone variants, evaluate against dual-tower and cross-encoder baselines, and further explore whether fine-tuning the backbone can improve performance on multi-hop reasoning tasks.