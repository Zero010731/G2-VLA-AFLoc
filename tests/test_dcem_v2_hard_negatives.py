from anaprior.train.dcem_v2_hard_negatives import (
    hard_negative_regions_for,
    is_hard_negative,
    normalize_region_name,
    positive_prior_regions_for,
)


def test_pneumothorax_marks_central_regions_as_hard_negative() -> None:
    assert is_hard_negative("Pneumothorax", "cardiac_silhouette")
    assert is_hard_negative("pneumothorax", "hilar_mediastinal")
    assert not is_hard_negative("Pneumothorax", "left_upper_lung")


def test_cardiomegaly_uses_only_cardiac_region_as_positive_prior() -> None:
    assert positive_prior_regions_for("Cardiomegaly") == frozenset({"cardiac_silhouette"})
    assert is_hard_negative("Cardiomegaly", "left_lower_lung")
    assert not is_hard_negative("Cardiomegaly", "cardiac_silhouette")


def test_lung_findings_downrank_cardiac_and_hilar_shortcuts() -> None:
    for finding in ["Consolidation", "Lung Opacity", "Pneumonia"]:
        hard_negatives = hard_negative_regions_for(finding)
        assert "cardiac_silhouette" in hard_negatives
        assert "hilar_mediastinal" in hard_negatives
        assert not is_hard_negative(finding, "right_lower_lung")


def test_edema_keeps_diffuse_lung_prior_without_cardiac_hard_negative() -> None:
    positives = positive_prior_regions_for("Edema")

    assert "bilateral_lungs" in positives
    assert "left_lower_lung" in positives
    assert not is_hard_negative("Edema", "cardiac_silhouette")


def test_pleural_effusion_hard_negatives_do_not_block_pleural_region() -> None:
    assert "pleural_space_costophrenic" in positive_prior_regions_for("Pleural Effusion")
    assert not is_hard_negative("Pleural Effusion", "pleural_space_costophrenic")
    assert is_hard_negative("Pleural Effusion", "hilar_mediastinal")


def test_chest_imagenome_raw_region_names_are_normalized_to_dcem_regions() -> None:
    assert normalize_region_name("cardiac silhouette") == "cardiac_silhouette"
    assert normalize_region_name("right upper lung zone") == "right_upper_lung"
    assert normalize_region_name("left lower lung zone") == "left_lower_lung"
    assert normalize_region_name("left hilar structures") == "hilar_mediastinal"
    assert normalize_region_name("right costophrenic angle") == "pleural_space_costophrenic"
    assert normalize_region_name("left lung") == "bilateral_lungs"


def test_hard_negative_checks_accept_chest_imagenome_raw_region_names() -> None:
    assert is_hard_negative("Pneumothorax", "cardiac silhouette")
    assert is_hard_negative("Consolidation", "right hilar structures")
    assert is_hard_negative("Cardiomegaly", "right lung")
    assert not is_hard_negative("Pneumothorax", "right upper lung zone")
