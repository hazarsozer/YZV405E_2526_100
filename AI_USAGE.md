# AI Tool Usage

Per the course guidelines, I'm logging how I used AI tools during the project.

---

## 2026-04-18 — Project setup & data loader (Phase 0)

Used Claude Code (CLI) as a coding assistant throughout this session.

**Decisions we made as a group**:
- 3-step pipeline architecture: translate → sense classify → heuristic rank
- Model choices: SigLIP2 for vision, BGE-M3 for text, Phi-3.5-mini-instruct (4-bit) for paraphrasing
- Confidence gate at 0.75 to fall back to raw cosine when classifier is uncertain
- Frozen encoders only — no fine-tuning due to our compute budget
- Data contract: immutable dataclasses, separate load paths for templates vs training TSVs

**What I used Claude for**:
- Parsing the competition PDF and dataset to verify everything was aligned
- Generating `PLAN.md` — I described the architecture verbally, Claude wrote it up. I reviewed and edited each section.
- Writing `src/data/loader.py` and the test suite after I specced out the exact interfaces
- Confirmed 0 missing image files across all 15 languages

Reviewed the code and all 44 tests before committing. Nothing was pushed blindly.

---

## 2026-04-19 — Frozen encoders, embedding cache, io utils (Phase 1)

Used Claude Code (CLI) as a coding assistant throughout this session.

**Decisions I made (with teammates)**:
- Keep SigLIP2 and BGE-M3 fully frozen — no fine-tuning, since we don't have the compute for it
- Cache embeddings to disk in Parquet format so we don't have to re-run the encoders on every experiment
- Use sha256 hashes as cache keys so the same input always maps to the same cached result

**What I used Claude for**:
- I described the interface I wanted for each module (what goes in, what comes out), then Claude wrote the initial code for `src/models/frozen_encoders.py`, `src/utils/cache.py`, and `src/utils/io.py`
- Claude also helped write the test suites using mocks so the tests don't require downloading any models
- I reviewed every file and test before committing — caught a few things I asked Claude to adjust

33 new tests added (77 total), all passing.

---

## 2026-04-20 — NLLB-200 translator, Phi-3.5 paraphraser, TextCache (Phase 2)

Used Claude Code (CLI) as a coding assistant throughout this session.

**Decisions I made (with teammates)**:
- Use NLLB-200 for translation so everything runs locally without an API — covers all the blind test languages
- Skip translation for English and PT-BR rows since they're already in English or close enough
- Use Phi-3.5-mini (4-bit quantized) to paraphrase sentences — replaces the idiom with a plain-language description, which should help the ranker later
- If the paraphraser returns nothing, fall back to the original sentence rather than crashing
- Keep a separate cache for text outputs (translations, paraphrases) since they're strings, not embeddings

**What I used Claude for**:
- I designed the prompt format and the batching logic, then had Claude implement `src/models/translator.py` and `src/models/slm_paraphraser.py` from my spec
- Claude extended the cache module with a TextCache class following the same pattern as EmbeddingCache
- Claude wrote the test suites for the new classes (mocked, no GPU needed to run)
- I reviewed all the code and tests before committing

38 new tests added (115 total), all passing.

---

## 2026-04-22 — LR sense classifier + train script (Phase 3)

Used Claude Code (CLI) as a coding assistant throughout this session.

**Decisions I made (with teammates)**:
- Use CalibratedClassifierCV with isotonic regression (cv=5) on top of LogisticRegression so the output is a proper probability, not a raw score
- Train only on EN and PT-BR splits (the two supervised languages) — all other languages are blind test
- Cache BGE-M3 embeddings to disk before training so the GPU step is only done once
- Thresholds: P(idiomatic) > 0.75 → idiomatic branch, < 0.25 → literal branch, otherwise fallback to cosine similarity
- Pin Python to 3.12 and disable the CUDA PyTorch index for macOS development (MPS wheels come from PyPI); re-enable CUDA index on the Linux GPU machine

**What I used Claude for**:
- I specced the classifier interface (fit/predict_proba_idiomatic/save/from_path), then had Claude implement `src/models/sense_classifier.py`
- Claude wrote `scripts/train_lr.py` with argparse, EmbeddingCache integration, and logging
- Claude wrote the test suite (15 tests, no GPU required)
- Fixed a uv/Python version conflict: system Python 3.14 is incompatible with torch wheels, pinned to 3.12 via `.python-version`
- I reviewed all code and tests before committing

15 new tests added (92 total), all passing.

## 2026-04-26 — Ranker, eval, pipeline, inference script (Phase 4)

Used Antigravity (Gemini) as a coding assistant throughout this session.

**Decisions I made (with teammates)**:
- Rank images using SigLIP2's shared text–image embedding space (not BGE-M3, which is text-only and lives in a different space)
- Added `encode_text()` to `SigLIP2Encoder` so both images and text are encoded in the same 1152-d space for cosine similarity ranking
- Idiomatic branch: `scores = sim_paraphrased - 0.3 * sim_original` to penalise literal visual overlap
- Literal branch: `scores = sim_original`
- Uncertain (bypass): `scores = sim_paraphrased` — no penalty applied
- Evaluation: standard nDCG with position-based relevance (top gold image gets relevance N, next N-1, …)
- Pipeline loads/unloads models sequentially to fit in VRAM (only one heavy model at a time)

**What I used the AI tool for**:
- I described the heuristic ranking logic and the 3-branch decision rules, then had the AI implement `src/models/ranker.py`
- I specced the evaluation interface (DCG/nDCG/top-1 + per-language aggregation), then had the AI implement `src/eval.py`
- I described the pipeline phases and memory management strategy, then had the AI implement `src/pipeline.py` and `scripts/run_inference.py`
- The AI wrote the test suites for all new modules (ranker: 12, eval: 19, pipeline: 4)
- Reviewed all code and tests before committing

35 new tests added (163 total), all passing (2 pre-existing integration tests fail because raw image data is not on this machine).

---

## 2026-05-07 – 2026-05-08 — Ablation submissions + systematic improvement search

Used Claude Code (CLI) as a coding assistant throughout these sessions.

**Decisions I made:**
- Switch ranking from SigLIP2 text–image alignment to BGE-M3 caption matching — captions are always English, giving a cleaner signal channel than SigLIP2's text encoder on translated/idiomatic input
- Use LR classifier (P(idiomatic) > 0.75 gate) to switch between paraphrase and original sentence as the ranking query
- Keep Phi-3.5-mini as the paraphraser — tested Qwen2.5-7B-Instruct and got identical scores
- Keep PT-BR in NO_OP_LANGS (skip translation) — routing through NLLB before paraphrasing hurt performance
- Hard gate at 0.75 is optimal — soft blend and threshold sweep confirmed this

**What I used Claude for:**
- Wrote `scripts/caption_rank.py` — main ranking script supporting multiple strategies (bge_clf_caption, siglip_bge_clf, bge_para_caption, soft blend), threshold sweep, and paraphrase namespace selection
- Wrote `scripts/regen_paraphrases.py` — reruns paraphrasing under a named cache namespace (used to regenerate with Qwen)
- Wrote `scripts/cross_encoder_rank.py` — tests BGE-reranker-v2-m3 cross-encoder as a replacement for bi-encoder cosine similarity
- Wrote `scripts/replay_ranking.py` — replays any cached strategy over the full dataset without re-running the encoder
- Wrote `scripts/audit_classifier.py` — audits LR classifier accuracy on EN dev set to confirm it is not the bottleneck
- Added Qwen prompt templates to `src/models/slm_paraphraser.py` (used by qwen_paraphraser.py)
- Wrote `ABLATIONS.md` — documents all 19 submitted strategies, scores, and key findings for the report
- I reviewed all code before running and checked outputs against Codabench scores

**Ablation results summary (19 strategies, 2663 rows, 15 languages):**
- Best score: **0.37** (`bge_clf_caption`, Phi-3.5, threshold 0.75)
- Empirical ceiling confirmed: improving paraphraser, similarity model, threshold, and translation pipeline all gave ≤ 0.37
- Root cause of ceiling: visual metaphor gap — figurative image captions describe literal scenes, not abstract concepts
- LR classifier accuracy: 94.1% on EN dev set — not the bottleneck

