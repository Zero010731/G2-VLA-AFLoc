import json
from pathlib import Path

import torch

from anaprior.eval.disease_properties import DISEASE_PROPERTY_NAMES, disease_property_matrix
from anaprior.eval.phrase_subtype import PHRASE_SUBTYPE_TO_ID
from anaprior.train.train_dense_dp_msa_adapter import train_dense_dp_msa_adapter


def _write_cache(path: Path) -> None:
    torch.manual_seed(9)
    base_hmaps = torch.rand(4, 1, 6, 6)
    region_maps = torch.rand(4, 3, 6, 6)
    region_scores = torch.tensor(
        [[0.9, 0.2, 0.0], [0.1, 0.8, 0.3], [0.2, 0.1, 0.7], [0.7, 0.0, 0.4]],
        dtype=torch.float32,
    )
    target_hmaps = torch.sum(region_maps * region_scores[:, :, None, None], dim=1, keepdim=True)
    payload = {
        "base_hmaps": base_hmaps,
        "region_maps": region_maps,
        "region_scores": region_scores,
        "target_hmaps": target_hmaps,
        "disease_properties": disease_property_matrix(["Pneumonia", "Lung Opacity", "Pneumonia", "Lung Opacity"]),
        "disease_property_names": DISEASE_PROPERTY_NAMES,
        "subtype_ids": torch.tensor(
            [
                PHRASE_SUBTYPE_TO_ID["basilar"],
                PHRASE_SUBTYPE_TO_ID["multifocal_patchy"],
                PHRASE_SUBTYPE_TO_ID["opacity_like"],
                PHRASE_SUBTYPE_TO_ID["pneumonia_like"],
            ],
            dtype=torch.long,
        ),
        "spatial_features": torch.rand(4, 5, 3, 3),
        "finding_vocab": {"Pneumonia": 0, "Lung Opacity": 1},
        "subtype_vocab": dict(PHRASE_SUBTYPE_TO_ID),
        "region_names": ["left_lower_lung", "right_lower_lung", "bilateral_lungs"],
    }
    torch.save(payload, path)


def test_train_dense_dp_msa_adapter_saves_checkpoint_and_report(tmp_path: Path) -> None:
    train_cache = tmp_path / "train.pt"
    valid_cache = tmp_path / "valid.pt"
    _write_cache(train_cache)
    _write_cache(valid_cache)

    report = train_dense_dp_msa_adapter(
        train_cache=train_cache,
        valid_cache=valid_cache,
        outdir=tmp_path / "out",
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        hidden_channels=4,
        condition_dim=4,
        lambda_weight=0.05,
        residual_l1_weight=0.01,
        ranking_loss_weight=0.1,
        seed=5,
        device="cpu",
    )

    assert report["status"] == "ok"
    assert report["uses_spatial_features"] is True
    assert Path(report["checkpoint"]).exists()
    saved = json.loads(Path(report["report"]).read_text(encoding="utf-8"))
    assert saved["model_config"]["spatial_channels"] == 5
    assert saved["epoch_losses"]
