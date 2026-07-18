from __future__ import annotations

import torch

from anaprior.models.afloc_mrsg.anchor import compute_afloc_phrase_anchor
from anaprior.models.afloc_mrsg.contracts import AFLocFeatureBatch, PhraseFeatureBatch


def test_anchor_is_detached_and_confidence_is_multiscale() -> None:
    image = AFLocFeatureBatch(
        img_emb_l2=torch.rand(2, 8, 4, 4, requires_grad=True),
        img_emb_l=torch.rand(2, 8, 2, 2, requires_grad=True),
        img_emb_lf=torch.rand(2, 8, 1, 1, requires_grad=True),
        image_gray=torch.rand(2, 1, 32, 32),
    )
    phrase = PhraseFeatureBatch(
        word_embeddings=torch.rand(2, 5, 8, requires_grad=True),
        sentence_embedding=torch.rand(2, 8),
        disease_description_embedding=torch.rand(2, 8),
        attention_mask=torch.tensor([[True, True, False, False, False], [True, True, True, False, False]]),
    )

    anchor, confidence, scales = compute_afloc_phrase_anchor(image, phrase)

    assert anchor.shape == (2, 1, 4, 4)
    assert confidence.shape == anchor.shape
    assert set(scales) == {"l2", "l", "lf"}
    assert not anchor.requires_grad
    assert not confidence.requires_grad
    assert torch.isfinite(anchor).all()
    assert ((confidence >= 0.0) & (confidence <= 1.0)).all()
