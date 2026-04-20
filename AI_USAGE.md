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

<!-- continue logging sessions below -->
