"""Category-aware heuristic ranker (Step 3 of the pipeline).

Ranks 5 candidate images based on SigLIP2 text-image cosine similarity,
guided by the sense classifier's P(idiomatic) output.

Decision rules
--------------
* P(idiomatic) > HIGH  → *Idiomatic branch*: rank by similarity to the
  paraphrased text; penalise images that match the literal (original) text.
* P(idiomatic) < LOW   → *Literal branch*: rank by similarity to the
  original text (literal visual overlap is desirable).
* Otherwise            → *Confidence-gated bypass*: fall back to raw cosine
  similarity with the paraphrased text (no heuristic penalty).
"""
from __future__ import annotations

import numpy as np

from src.models.sense_classifier import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
)

# How much to penalise literal visual overlap in the idiomatic branch.
# Setting this to 0.0 disables the penalty; 1.0 fully subtracts it.
LITERAL_PENALTY_WEIGHT = 0.3


def cosine_similarity(
    text_emb: np.ndarray, image_embs: np.ndarray
) -> np.ndarray:
    """Cosine similarity between one text vector and N image vectors.

    Both inputs should already be L2-normalised (from the frozen encoders).
    Returns shape (N,).
    """
    # Guard against un-normalised inputs
    t = text_emb / max(np.linalg.norm(text_emb), 1e-8)
    norms = np.linalg.norm(image_embs, axis=1, keepdims=True)
    imgs = image_embs / np.maximum(norms, 1e-8)
    return (imgs @ t).astype(np.float32)


def rank_images(
    image_embeddings: np.ndarray,
    original_text_emb: np.ndarray,
    paraphrased_text_emb: np.ndarray,
    p_idiomatic: float,
    image_names: list[str],
    *,
    high_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
    low_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    literal_penalty: float = LITERAL_PENALTY_WEIGHT,
) -> list[str]:
    """Return *image_names* sorted best-to-worst according to heuristic rules.

    Args:
        image_embeddings: (5, D) L2-normalised SigLIP2 image embeddings.
        original_text_emb: (D,) SigLIP2 text embedding of the *original*
            (translated but not paraphrased) sentence.
        paraphrased_text_emb: (D,) SigLIP2 text embedding of the
            *paraphrased* sentence (idiom replaced with plain meaning).
        p_idiomatic: Calibrated P(idiomatic) from the sense classifier.
        image_names: File-names of the 5 candidate images (same order as
            *image_embeddings*).
        high_threshold: P above which the idiomatic branch fires.
        low_threshold: P below which the literal branch fires.
        literal_penalty: Weight subtracted for literal overlap in the
            idiomatic branch.

    Returns:
        List of 5 image file-names ordered from best to worst match.
    """
    if len(image_names) != len(image_embeddings):
        raise ValueError(
            f"image_names length {len(image_names)} != "
            f"image_embeddings length {len(image_embeddings)}"
        )

    sim_orig = cosine_similarity(original_text_emb, image_embeddings)
    sim_para = cosine_similarity(paraphrased_text_emb, image_embeddings)

    if p_idiomatic > high_threshold:
        # Idiomatic: reward paraphrased match, penalise literal match
        scores = sim_para - literal_penalty * sim_orig
    elif p_idiomatic < low_threshold:
        # Literal: reward literal visual overlap
        scores = sim_orig
    else:
        # Uncertain: confidence-gated bypass — raw cosine only
        scores = sim_para

    ranked_indices = np.argsort(-scores)
    return [image_names[int(i)] for i in ranked_indices]
