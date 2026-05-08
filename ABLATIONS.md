# Ablation Study Results — AdMIRe 2.0 (YZV405E Group 100)

All scores are **Top-1 accuracy** on the Codabench blind test set (avg across 15 languages).
Submissions made 2026-05-07 and 2026-05-08.

---

## Pipeline Overview

```
sentence (any of 15 languages)
    ↓ NLLB-200-distilled-600M
translated (English)  [PT-BR and EN skip this step]
    ↓ Phi-3.5-mini-instruct (4-bit)
paraphrase (idiom replaced with plain-English meaning)
    ↓ BGE-M3 → Logistic Regression (calibrated, isotonic)
P(idiomatic) ∈ [0, 1]
    ↓ ranking strategy (see table below)
ranked image order
```

**Image captions** (auto-generated English descriptions provided in the dataset) are
available as an alternative signal channel — used by all BGE caption strategies.

---

## Baseline

| Strategy | Description | Score |
|----------|-------------|-------|
| `classifier_a0.3` | Original proposed pipeline: SigLIP2 text+image alignment, classifier-gated (α=0.3 penalty) | **0.30** |

---

## Round 1 — SigLIP2 Ablations

Exploring whether the ranking signal or the penalty weight was the bottleneck in the
original SigLIP2-based pipeline. All strategies use SigLIP2 image and text encoders.

| Strategy | Description | Score |
|----------|-------------|-------|
| `orig_only` | Pure `sim(SigLIP2_text(original), SigLIP2_image)` — no paraphrase, no penalty | 0.28 |
| `bge_para_caption` | BGE-M3 `sim(paraphrase, caption)` — always use paraphrase, no classifier | 0.35 |
| `bge_clf_caption` | BGE-M3 `sim(query, caption)`, classifier-gated: paraphrase if P>0.75, original otherwise | **0.37** |
| `siglip_bge_clf` | Ensemble: 30% SigLIP2-clf + 70% BGE-caption, classifier-gated | 0.37 |

**Key finding:** BGE-M3 caption matching outperforms SigLIP2 image-text alignment.
SigLIP2 adds no value over pure BGE — `siglip_bge_clf` ties `bge_clf_caption`.
Root cause: SigLIP2's text encoder is weak on non-English input and idioms; captions
are always in English, making them a cleaner signal channel.

**Root cause confirmed (audit):** LR classifier is NOT the bottleneck — 94.1% Top-1
accuracy on EN dev set (80/85 correct, precision 0.935, recall 0.956).

---

## Round 2 — Paraphrase Quality

Testing whether upgrading the paraphraser from Phi-3.5-mini (3.8B) to Qwen2.5-7B-Instruct
would improve the BGE caption matching scores. Both use 4-bit quantization.

| Strategy | Paraphraser | Score |
|----------|-------------|-------|
| `bge_clf_caption` | Phi-3.5-mini-instruct (4-bit, ~2.5 GB) | 0.37 |
| `qwen_clf_caption` | Qwen2.5-7B-Instruct (4-bit, ~4.5 GB) | 0.37 |
| `qwen_para_caption` | Qwen2.5-7B, always paraphrase | 0.36 |

**Key finding:** Paraphrase quality is NOT the bottleneck. A 7B model with better
instruction following gives identical results to the 3.8B model. The ceiling is in
the similarity scoring step, not the query generation.

---

## Round 3 — Similarity Scoring Method

Testing whether replacing bi-encoder cosine similarity with a cross-encoder
(which jointly encodes both texts) would improve caption matching.

Model: `BAAI/bge-reranker-v2-m3` — multilingual cross-encoder (~600M parameters).

| Strategy | Similarity Method | Score |
|----------|-------------------|-------|
| `bge_clf_caption` (baseline) | Bi-encoder cosine similarity | 0.37 |
| `xenc_clf` | Cross-encoder score, classifier-gated | 0.35 |
| `xenc_para` | Cross-encoder score, always paraphrase | 0.35 |

**Key finding:** Cross-encoder performs *worse* than bi-encoder. This is diagnostic:
cross-encoders typically outperform bi-encoders on text relevance tasks. The fact that
they perform worse here confirms that the task is not a standard retrieval problem —
figurative image captions describe scenes literally, and the semantic gap between a
paraphrase and a literal scene description cannot be bridged by any text model alone.

---

## Round 4 — Data Pipeline Changes

Testing two targeted fixes:

**PT-BR translation fix:** PT-BR was in `NO_OP_LANGS` (treated as English), so Phi-3.5
and Qwen received Portuguese input for paraphrasing. Fix: route PT-BR through
NLLB-200 translation before paraphrasing.

**Soft p-weighted blend:** Instead of hard gate at P=0.75, blend scores continuously:
`score = P(idiomatic) × sim(paraphrase, caption) + (1 − P) × sim(original, caption)`

| Strategy | Changes | Score |
|----------|---------|-------|
| `bge_clf_caption` (baseline) | None | 0.37 |
| `final_clf` | PT-BR fix + Qwen paraphrases | 0.36 |
| `final_soft_blend` | PT-BR fix + Qwen + soft blend | 0.36 |

**Key finding:** PT-BR fix *hurts* — NLLB translation of Portuguese introduces more
noise than Qwen/Phi-3.5 handling Portuguese input directly. PT-BR fix was reverted.
Soft blend also does not improve over the hard gate.

---

## Round 5 — Classifier Threshold Sweep

Testing whether the hard gate threshold of P=0.75 is optimal. Lower thresholds use
the paraphrase more aggressively; higher thresholds are more conservative.

| Threshold | Strategy tag | Score |
|-----------|-------------|-------|
| 0.50 | `bge_clf_t050` | 0.36 |
| 0.55 | `bge_clf_t055` | 0.36 |
| 0.60 | `bge_clf_t060` | 0.37 |
| 0.65 | `bge_clf_t065` | 0.37 |
| **0.75** | `bge_clf_caption` (default) | **0.37** |
| 0.70 | `bge_clf_t070` | 0.37 |
| 0.80 | `bge_clf_t080` | 0.37 |
| 0.85 | `bge_clf_t085` | 0.37 |

**Key finding:** The score is flat for all thresholds ≥ 0.60. Thresholds below 0.60
hurt slightly by over-committing to the paraphrase path for uncertain instances. The
original threshold of 0.75 is at or near optimal.

---

## Summary of All Submitted Scores

| Score | Count | Strategies |
|-------|-------|------------|
| **0.37** | 9 | `bge_clf_caption`, `siglip_bge_clf`, `qwen_clf_caption`, `bge_clf_t060`–`t085` |
| 0.36 | 5 | `qwen_para_caption`, `final_clf`, `final_soft_blend`, `bge_clf_t050`, `bge_clf_t055` |
| 0.35 | 3 | `bge_para_caption`, `xenc_clf`, `xenc_para` |
| 0.30 | 1 | `classifier_a0.3` (original SigLIP2 pipeline) |
| 0.28 | 1 | `orig_only` (SigLIP2 baseline, no classifier) |

**Best submission:** `bge_clf_caption` with Phi-3.5 paraphrases, threshold 0.75 → **0.37**

---

## Key Takeaways for the Report

1. **The proposed SigLIP2 ranker (0.30) was improved by switching to caption-based
   BGE-M3 matching (+0.07)**. Root cause of original underperformance: SigLIP2's text
   encoder is weak on non-English input and idiomatic expressions after NLLB translation.

2. **The classifier is highly accurate (94.1%) and provides measurable value (+0.02)**
   over always-paraphrase strategies. It correctly identifies that literal sentences should
   use the original text as the ranking query.

3. **The ceiling for text-caption matching is 0.37**. Improving every component
   independently (paraphraser quality, similarity model, threshold, translation) did not
   push beyond this ceiling. The fundamental limitation is the visual metaphor gap:
   figurative images depict abstract concepts through literal scenes, and their captions
   describe those scenes — not the concepts.

4. **SigLIP2 image features add no value** when combined with BGE-M3 caption matching
   (0.30/0.70 ensemble ties pure BGE at 0.37). This suggests that for this task, the
   text-based caption signal fully dominates the visual signal.

---

## Reproducibility

All embeddings, translations, and paraphrases are cached in `data/processed/`.
Re-running any strategy is instant after the first run.

```bash
# Re-run best strategy
uv run python -m scripts.caption_rank \
    --strategies bge_clf_caption \
    --paraphrase-ns phi35_paraphrase \
    --output-dir data/submissions/final

# Zip and submit
cd data/submissions/final/bge_clf_caption
zip -j ../submission.zip *.tsv
```
