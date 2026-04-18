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

<!-- continue logging sessions below -->
