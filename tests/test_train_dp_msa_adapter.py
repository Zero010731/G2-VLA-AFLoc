import json
from pathlib import Path

import torch

from anaprior.eval.phrase_subtype import PHRASE_SUBTYPE_TO_ID
from anaprior.train.train_dp_msa_adapter import train_dp_msa_adapter


def _write_cache(path: Path) -> None:
    torch.manual_seed(7)
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
        "disease_ids": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "subtype_ids": torch.tensor(
            [
                PHRASE_SUBTYPE_TO_ID["basilar"],
                PHRASE_SUBTYPE_TO_ID["multifocal_patchy"],
                PHRASE_SUBTYPE_TO_ID["opacity_like"],
                PHRASE_SUBTYPE_TO_ID["pneumonia_like"],
            ],
            dtype=torch.long,
        ),
        "finding_vocab": {"Pneumonia": 0, "Lung Opacity": 1},
        "subtype_vocab": dict(PHRASE_SUBTYPE_TO_ID),
        "region_names": ["left_lower_lung", "right_lower_lung", "bilateral_lungs"],
    }
    torch.save(payload, path)


def test_train_dp_msa_adapter_saves_checkpoint_and_report(tmp_path: Path) -> None:
    train_cache = tmp_path / "train.pt"
    valid_cache = tmp_path / "valid.pt"
    _write_cache(train_cache)
    _write_cache(valid_cache)

    report = train_dp_msa_adapter(
        train_cache=train_cache,
        valid_cache=valid_cache,
        outdir=tmp_path / "out",
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        hidden_channels=4,
        embedding_dim=4,
        lambda_weight=0.1,
        residual_l1_weight=0.01,
        seed=3,
        device="cpu",
    )

    assert report["status"] == "ok"
    assert report["num_train_rows"] == 4
    assert report["num_regions"] == 3
    assert Path(report["checkpoint"]).exists()
    assert Path(report["report"]).exists()
    saved = json.loads(Path(report["report"]).read_text(encoding="utf-8"))
    assert saved["finding_vocab"] == {"Pneumonia": 0, "Lung Opacity": 1}
    assert saved["epoch_losses"]
