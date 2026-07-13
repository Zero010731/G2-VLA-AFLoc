from dataclasses import replace

import pytest
import torch

from anaprior.models.afloc_mrsg.contracts import (
    AFLocFeatureBatch,
    MRSGConfig,
    MRSGOutput,
    PhraseFeatureBatch,
)


def _valid_output() -> MRSGOutput:
    return MRSGOutput(
        final_heatmap=torch.rand(2, 1, 16, 16),
        query_heatmaps=torch.rand(2, 4, 16, 16),
        query_route_weights=torch.softmax(torch.rand(2, 4), dim=-1),
        query_reliability=torch.rand(2, 4),
        phrase_patch_logits=torch.rand(2, 8, 16, 16),
    )


def test_mrsg_config_has_exactly_four_query_operators() -> None:
    config = MRSGConfig()

    assert config.query_names == ("focal", "diffuse", "boundary", "structural")
    with pytest.raises(ValueError, match="exactly four"):
        replace(config, query_names=("focal", "diffuse"))


def test_mrsg_config_rejects_renamed_query_operators() -> None:
    with pytest.raises(ValueError, match="focal, diffuse, boundary, structural"):
        replace(
            MRSGConfig(),
            query_names=("focal", "diffuse", "boundary", "generic"),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"feature_dim": 15, "num_heads": 4}, "divisible"),
        ({"topk_fraction": 0.0}, "topk_fraction"),
        ({"topk_fraction": 1.1}, "topk_fraction"),
    ],
)
def test_mrsg_config_rejects_invalid_model_dimensions(changes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(MRSGConfig(), **changes)


def test_mrsg_output_validates_direct_heatmap_shapes() -> None:
    output = _valid_output()

    output.validate()

    assert output.final_heatmap.shape == (2, 1, 16, 16)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("final_heatmap", torch.rand(2, 2, 16, 16), "one channel"),
        ("query_heatmaps", torch.rand(2, 3, 16, 16), r"\[B,4,H,W\]"),
        ("query_route_weights", torch.rand(2, 3), r"\[B,4\]"),
        ("query_reliability", torch.rand(2, 3), r"\[B,4\]"),
        ("phrase_patch_logits", torch.rand(2, 8, 8, 8), "same spatial"),
    ],
)
def test_mrsg_output_rejects_inconsistent_shapes(field, value, message: str) -> None:
    output = replace(_valid_output(), **{field: value})

    with pytest.raises(ValueError, match=message):
        output.validate()


def test_feature_contracts_are_immutable_tensor_containers() -> None:
    image_features = AFLocFeatureBatch(
        img_emb_l2=torch.rand(2, 32, 16, 16),
        img_emb_l=torch.rand(2, 64, 8, 8),
        img_emb_lf=torch.rand(2, 128, 4, 4),
        image_gray=torch.rand(2, 1, 224, 224),
    )
    phrase_features = PhraseFeatureBatch(
        word_embeddings=torch.rand(2, 7, 24),
        sentence_embedding=torch.rand(2, 24),
        disease_description_embedding=torch.rand(2, 24),
        attention_mask=torch.ones(2, 7, dtype=torch.bool),
    )

    assert image_features.img_emb_l2.shape[0] == 2
    assert phrase_features.word_embeddings.shape[:2] == (2, 7)
    with pytest.raises(Exception):
        image_features.image_gray = torch.rand(2, 1, 224, 224)
