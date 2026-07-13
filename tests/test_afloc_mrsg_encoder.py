from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from anaprior.features.afloc_mrsg_encoder import FrozenAFLocMRSGEncoder
from tests.mrsg_test_utils import FakeAFLoc


class DeterministicFakeAFLoc(FakeAFLoc):
    def __init__(self) -> None:
        super().__init__(text_dim=3)

    def process_text(self, texts, device):
        encoded = []
        for text in texts:
            if text == "small right apical pneumothorax":
                encoded.append(([1, 2, 0, 0], [1, 1, 0, 0]))
            elif text == "air in the pleural space":
                encoded.append(([7, 8, 9, 0], [1, 1, 1, 0]))
            else:
                raise AssertionError(f"unexpected text: {text}")

        caption_ids = torch.tensor([row[0] for row in encoded], dtype=torch.long, device=device)
        attention_mask = torch.tensor([row[1] for row in encoded], dtype=torch.long, device=device)
        token_type_ids = torch.zeros_like(caption_ids)
        return {
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        del attention_mask, token_type_ids
        base = caption_ids.to(torch.float32)
        word_embeddings = torch.stack((base, base + 10.0, base + 20.0), dim=1)
        report_embeddings = torch.stack((base[:, 0], base[:, 1], base[:, 0] + base[:, 1]), dim=1)
        sent_embeddings = report_embeddings.unsqueeze(-1)
        return {
            "word_embeddings": word_embeddings,
            "sent_embeddings": sent_embeddings,
            "report_embeddings": report_embeddings,
            "report": [["finding"] for _ in range(caption_ids.shape[0])],
            "sent_units_report": [["finding"] for _ in range(caption_ids.shape[0])],
        }


class BadImageShapeAFLoc(FakeAFLoc):
    def image_encoder_forward(self, images: torch.Tensor):
        batch = images.shape[0]
        device = images.device
        return (
            torch.rand(batch, 64, 8, device=device),
            torch.rand(batch, 32, 16, 16, device=device),
            torch.rand(batch, 128, 4, 4, device=device),
            torch.rand(batch, 128, device=device),
        )


class BadWordShapeAFLoc(FakeAFLoc):
    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        del attention_mask, token_type_ids
        batch, tokens = caption_ids.shape
        device = caption_ids.device
        return {
            "word_embeddings": torch.rand(batch, tokens, device=device),
            "sent_embeddings": torch.rand(batch, self.text_dim, 1, device=device),
            "report_embeddings": torch.rand(batch, self.text_dim, device=device),
            "report": [["finding"] for _ in range(batch)],
            "sent_units_report": [["finding"] for _ in range(batch)],
        }


def test_encoder_freezes_afloc_and_returns_all_feature_levels() -> None:
    fake_afloc = FakeAFLoc()
    encoder = FrozenAFLocMRSGEncoder(fake_afloc)

    features = encoder.encode_images(torch.rand(2, 3, 224, 224))

    assert features.img_emb_l2.shape == (2, 32, 16, 16)
    assert features.img_emb_l.shape == (2, 64, 8, 8)
    assert features.img_emb_lf.shape == (2, 128, 4, 4)
    assert features.image_gray.shape == (2, 1, 224, 224)
    assert all(not parameter.requires_grad for parameter in fake_afloc.parameters())
    assert not fake_afloc.training


def test_encoder_outputs_are_detached_and_backbone_gets_no_gradients() -> None:
    fake_afloc = FakeAFLoc()
    encoder = FrozenAFLocMRSGEncoder(fake_afloc)

    image_features = encoder.encode_images(torch.rand(2, 3, 224, 224))
    phrase_features = encoder.encode_phrases(
        ["small right apical pneumothorax", "small right apical pneumothorax"],
        ["air in the pleural space", "air in the pleural space"],
        device=torch.device("cpu"),
    )
    head = nn.Conv2d(image_features.img_emb_l2.shape[1], 1, kernel_size=1)
    text_head = nn.Linear(phrase_features.word_embeddings.shape[-1], 1)

    loss = head(image_features.img_emb_l2).sum()
    loss = loss + text_head(phrase_features.word_embeddings).sum()
    loss.backward()

    assert not image_features.img_emb_l2.requires_grad
    assert not image_features.img_emb_l.requires_grad
    assert not image_features.img_emb_lf.requires_grad
    assert not image_features.image_gray.requires_grad
    assert not phrase_features.word_embeddings.requires_grad
    assert not phrase_features.sentence_embedding.requires_grad
    assert not phrase_features.disease_description_embedding.requires_grad
    assert not phrase_features.attention_mask.requires_grad
    assert all(parameter.grad is None for parameter in fake_afloc.parameters())


def test_encoder_train_cannot_switch_afloc_out_of_eval() -> None:
    fake_afloc = FakeAFLoc().train()
    encoder = FrozenAFLocMRSGEncoder(fake_afloc)

    encoder.train()

    assert not fake_afloc.training


def test_encoder_normalizes_word_token_layout_and_text_embeddings() -> None:
    encoded = FrozenAFLocMRSGEncoder(DeterministicFakeAFLoc()).encode_phrases(
        ["small right apical pneumothorax"],
        ["air in the pleural space"],
        device=torch.device("cpu"),
    )

    assert encoded.word_embeddings.shape == (1, 4, 3)
    assert torch.equal(
        encoded.word_embeddings[0],
        torch.tensor(
            [
                [1.0, 11.0, 21.0],
                [2.0, 12.0, 22.0],
                [0.0, 10.0, 20.0],
                [0.0, 10.0, 20.0],
            ]
        ),
    )
    assert torch.equal(encoded.sentence_embedding, torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(
        encoded.disease_description_embedding,
        torch.tensor([[7.0, 8.0, 15.0]]),
    )
    assert encoded.attention_mask.dtype == torch.bool
    assert torch.equal(encoded.attention_mask, torch.tensor([[True, True, False, False]]))


def test_encoder_forward_returns_image_and_phrase_feature_batches() -> None:
    encoder = FrozenAFLocMRSGEncoder(FakeAFLoc())

    image_features, phrase_features = encoder(
        torch.rand(2, 3, 224, 224),
        ["small right apical pneumothorax", "small right apical pneumothorax"],
        ["air in the pleural space", "air in the pleural space"],
    )

    assert image_features.img_emb_l2.shape[0] == 2
    assert phrase_features.word_embeddings.shape[0] == 2


def test_encoder_rejects_invalid_image_feature_shapes() -> None:
    encoder = FrozenAFLocMRSGEncoder(BadImageShapeAFLoc())

    with pytest.raises(ValueError, match="img_emb_l must have shape \\[B,C,H,W\\]"):
        encoder.encode_images(torch.rand(2, 3, 224, 224))


def test_encoder_rejects_invalid_word_embedding_shapes() -> None:
    encoder = FrozenAFLocMRSGEncoder(BadWordShapeAFLoc())

    with pytest.raises(ValueError, match="word_embeddings must have shape \\[B,D,T\\]"):
        encoder.encode_phrases(
            ["small right apical pneumothorax"],
            ["air in the pleural space"],
            device=torch.device("cpu"),
        )


def test_from_checkpoint_uses_afloc_builder_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = FakeAFLoc()

    def fake_load_model(ckpt_path: str, device: str):
        assert ckpt_path.endswith("afloc.ckpt")
        assert device == "cpu"
        return loaded

    monkeypatch.setattr("anaprior.features.afloc_mrsg_encoder.load_model", fake_load_model)

    encoder = FrozenAFLocMRSGEncoder.from_checkpoint(Path("weights") / "afloc.ckpt", device="cpu")

    assert encoder.afloc is loaded
