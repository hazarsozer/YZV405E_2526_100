# AdMIRe 2.0 — YZV405E Group 100

Image ranking pipeline for the [AdMIRe 2.0 shared task](https://www.codabench.org/competitions/10547/) (MWE-2026 @ EACL 2026), Subtask A (Images + Text).

**Team:** Hazar Utku Sözer · Faruk Rıza Öz · Enis Furkan Kırmızı
**Course:** YZV405E, Istanbul Technical University, Spring 2026

---

## Task

Given a sentence containing a potentially idiomatic nominal compound (e.g. *"bad apple"*) and 5 candidate images, rank the images by how well they represent the meaning of the compound **in that context**.

- **Idiomatic** context → best image depicts the figurative meaning
- **Literal** context → best image depicts the literal meaning
- Gold ranking is always `[figurative-strong, figurative-mild, literal-mild, literal-strong, distractor]` for idiomatic sentences (reversed for literal)
- **Metric:** Top-1 accuracy averaged across 15 languages (13 zero-shot)
- **Supervised languages:** EN, PT-BR only

---

## Pipeline

```
sentence (any of 15 languages)
    │
    ▼ NLLB-200-distilled-600M  (skipped for EN / PT-BR)
translated sentence (English)
    │
    ▼ Phi-3.5-mini-instruct 4-bit
paraphrase  (idiom replaced with plain-English meaning)
    │
    ▼ BGE-M3 embeddings → Logistic Regression (calibrated, isotonic, cv=5)
P(idiomatic) ∈ [0, 1]
    │
    ├─ P > 0.75  →  sim(BGE-M3(paraphrase), BGE-M3(caption))
    ├─ P < 0.25  →  sim(BGE-M3(original),   BGE-M3(caption))
    └─ otherwise →  sim(BGE-M3(paraphrase), BGE-M3(caption))   [bypass]
                                        │
                                        ▼
                              ranked image order (1–5)
```

Captions are auto-generated English descriptions provided in the dataset for each image. They are the ranking signal — not raw image pixels.

---

## Models

| Model | Role | VRAM |
|-------|------|------|
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | Text encoder (LR features + ranking) | ~2.3 GB fp16 |
| [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) | Vision encoder (ablations only) | ~3.5 GB fp16 |
| [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | Translation to English | ~1.2 GB fp16 |
| [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | Paraphrasing (4-bit) | ~2.5 GB |

All encoders are **frozen** — no fine-tuning. Models are loaded and unloaded sequentially to fit within 12 GB VRAM.

---

## Results

Best Codabench score: **0.37 Top-1 accuracy** (`bge_clf_caption`, Phi-3.5, threshold 0.75).

| Strategy | Description | Score |
|----------|-------------|-------|
| `orig_only` | SigLIP2 baseline, no classifier | 0.28 |
| `classifier_a0.3` | Original SigLIP2 heuristic ranker | 0.30 |
| `bge_para_caption` | BGE caption, always paraphrase | 0.35 |
| `siglip_bge_clf` | 30% SigLIP2 + 70% BGE caption, clf-gated | 0.37 |
| **`bge_clf_caption`** | **BGE caption, clf-gated at 0.75** | **0.37** |

See [ABLATIONS.md](ABLATIONS.md) for the full ablation study (19 strategies).

---

## Setup

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

**On Linux with a CUDA GPU**, torch installs from the PyTorch CUDA 12.1 index automatically (configured in `pyproject.toml`). On macOS, PyPI MPS wheels are used instead.

### Data

Place the AdMIRe 2.0 dataset under `data/raw/admire2_data/`:

```
data/raw/admire2_data/
├── Chinese/
├── Turkish/
├── ...   (one folder per language)
└── English/
    ├── bad_apple/
    │   ├── 12345678.png
    │   └── ...
    └── ...
```

Training/dev TSVs go in the same folder as each language directory. See `src/data/loader.py` for the expected structure.

---

## Running the Pipeline

### 1. Train the LR classifier

```bash
uv run python -m scripts.train_lr
```

Outputs `models/lr_sense_classifier.joblib`. Requires GPU for BGE-M3 embedding (cached to `data/processed/` after first run).

### 2. Run the full pipeline and generate submissions

```bash
uv run python -m scripts.run_inference \
    --output-dir data/submissions/output
```

Produces one TSV per language in the output directory.

### 3. Ranking-only (uses cached embeddings)

```bash
uv run python -m scripts.caption_rank \
    --strategies bge_clf_caption \
    --paraphrase-ns phi35_paraphrase \
    --output-dir data/submissions/final
```

Available strategies: `bge_clf_caption`, `bge_para_caption`, `siglip_bge_clf`, `soft_blend`.
Pass `--threshold-sweep` to test thresholds 0.50–0.85 in one run.

### 4. Regenerate paraphrases (e.g. with a different model)

```bash
uv run python -m scripts.regen_paraphrases \
    --namespace my_paraphrase_ns
```

### 5. Cross-encoder ranking (ablation)

```bash
uv run python -m scripts.cross_encoder_rank \
    --output-dir data/submissions/xenc
```

### 6. Audit classifier accuracy on EN dev set

```bash
uv run python -m scripts.audit_classifier
```

---

## Tests

```bash
uv run pytest
```

163 tests, all passing. Integration tests that need raw image data are skipped automatically if the dataset is not present.

---

## Project Structure

```
src/
  data/loader.py              # Dataset loading and validation
  models/
    frozen_encoders.py        # SigLIP2 and BGE-M3 wrappers
    translator.py             # NLLB-200 translation
    slm_paraphraser.py        # Phi-3.5-mini paraphraser
    qwen_paraphraser.py       # Qwen2.5-7B paraphraser (ablation)
    sense_classifier.py       # LR idiomatic/literal classifier
    ranker.py                 # Category-aware heuristic ranker
  utils/
    cache.py                  # Parquet embedding and text cache
    io.py                     # Path and hash helpers
  pipeline.py                 # End-to-end orchestrator
  eval.py                     # Top-1 accuracy, DCG, nDCG
scripts/
  run_inference.py            # Full pipeline CLI
  train_lr.py                 # Train and save LR classifier
  caption_rank.py             # Ranking-only script (uses cache)
  regen_paraphrases.py        # Regenerate paraphrases
  cross_encoder_rank.py       # Cross-encoder ranking (ablation)
  replay_ranking.py           # Replay any cached strategy
  audit_classifier.py         # Evaluate LR classifier on EN dev
tests/                        # 163 unit + integration tests
ABLATIONS.md                  # Full ablation study with scores
AI_USAGE.md                   # AI tool disclosure (course requirement)
```

---

## Caching

All embeddings, translations, and paraphrases are cached to `data/processed/` as Parquet files. After the first full run, every subsequent run of any script is near-instant for already-processed data. Cache keys are SHA-256 hashes of the input text.

---

## Competition Links

- Subtask A (Images + Text): https://www.codabench.org/competitions/10547/
- Subtask B (Text only): https://www.codabench.org/competitions/10548/
