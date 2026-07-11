import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from anaprior.eval.disease_conditioned_gate import apply_disease_conditioned_gate, apply_disease_specific_pooling


def test_pneumothorax_blocks_central_top_region_and_bypasses_repair() -> None:
    regions = ["cardiac_silhouette", "left_upper_lung", "pleural_space_costophrenic"]
    scores = np.array([0.9, 0.4, 0.3], dtype=np.float32)

    gated = apply_disease_conditioned_gate("Pneumothorax", regions, scores)

    assert gated.status == "bypass"
    assert gated.reason == "blocked_top_region"
    assert gated.top_region == "cardiac_silhouette"
    assert np.allclose(gated.scores, np.zeros_like(scores))


def test_pleural_effusion_downweights_central_regions_but_keeps_pleural_evidence() -> None:
    regions = [
        "hilar_mediastinal",
        "pleural_space_costophrenic",
        "left_lower_lung",
    ]
    scores = np.array([0.9, 0.6, 0.5], dtype=np.float32)

    gated = apply_disease_conditioned_gate("Pleural Effusion", regions, scores)

    assert gated.status == "use"
    assert gated.reason == "passed"
    assert gated.scores[1] > gated.scores[0]
    assert gated.scores[2] > gated.scores[0]


def test_unknown_finding_passes_raw_scores_through() -> None:
    scores = np.array([0.1, 0.8], dtype=np.float32)

    gated = apply_disease_conditioned_gate("Unknown Finding", ["a", "b"], scores)

    assert gated.status == "use"
    assert gated.reason == "no_disease_config"
    assert np.allclose(gated.scores, scores)


def test_disease_specific_pooling_suppresses_pneumothorax_central_shortcut_without_bypass() -> None:
    regions = ["cardiac_silhouette", "left_upper_lung", "pleural_space_costophrenic"]
    scores = np.array([0.95, 0.4, 0.3], dtype=np.float32)

    pooled = apply_disease_specific_pooling("Pneumothorax", regions, scores)

    assert pooled.status == "use"
    assert pooled.reason == "pooled"
    assert pooled.scores[0] == 0.0
    assert pooled.scores[1] > 0.0
    assert pooled.scores[2] > 0.0


def test_disease_specific_pooling_keeps_edema_diffuse_bilateral_signal() -> None:
    regions = ["bilateral_lungs", "left_lower_lung", "cardiac_silhouette"]
    scores = np.array([0.5, 0.4, 0.9], dtype=np.float32)

    pooled = apply_disease_specific_pooling("Edema", regions, scores)

    assert pooled.status == "use"
    assert pooled.scores[0] > pooled.scores[1]
    assert pooled.scores[2] == 0.9
