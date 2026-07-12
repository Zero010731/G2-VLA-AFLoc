import pytest
import torch

from anaprior.eval.disease_properties import (
    DISEASE_PROPERTY_NAMES,
    DISEASE_PROPERTY_TABLE,
    disease_property_matrix,
    disease_property_vector,
)


def _property_index(name: str) -> int:
    return DISEASE_PROPERTY_NAMES.index(name)


def test_disease_property_table_is_frozen_and_normalized() -> None:
    assert DISEASE_PROPERTY_NAMES == (
        "focality_focal",
        "focality_regional",
        "focality_diffuse",
        "compartment_parenchymal",
        "compartment_pleural",
        "compartment_cardiac",
        "compartment_mediastinal",
        "laterality_unilateral",
        "laterality_bilateral",
        "vertical_apical",
        "vertical_lower",
        "boundary_dependence",
        "texture_dependence",
        "expected_extent_large",
        "central_shortcut_risk",
    )
    assert set(DISEASE_PROPERTY_TABLE) == {
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Lung Opacity",
        "Pleural Effusion",
        "Pneumonia",
        "Pneumothorax",
    }
    for values in DISEASE_PROPERTY_TABLE.values():
        assert len(values) == len(DISEASE_PROPERTY_NAMES)
        assert all(0.0 <= value <= 1.0 for value in values)


def test_disease_property_vector_encodes_spatial_properties_not_branches() -> None:
    pneumo = disease_property_vector("Pneumothorax")
    edema = disease_property_vector("Edema")
    cardio = disease_property_vector("Cardiomegaly")

    assert pneumo[_property_index("compartment_pleural")] > 0.8
    assert pneumo[_property_index("vertical_apical")] > 0.5
    assert pneumo[_property_index("boundary_dependence")] > 0.8
    assert pneumo[_property_index("central_shortcut_risk")] > 0.8

    assert edema[_property_index("focality_diffuse")] > 0.8
    assert edema[_property_index("laterality_bilateral")] > 0.8
    assert edema[_property_index("expected_extent_large")] > 0.8

    assert cardio[_property_index("compartment_cardiac")] > 0.8
    assert cardio[_property_index("boundary_dependence")] > 0.8
    assert cardio[_property_index("compartment_pleural")] < 0.2


def test_disease_property_matrix_handles_aliases_and_unknowns() -> None:
    matrix = disease_property_matrix(["lung opacity", "pleural_effusion", "PNEUMONIA"])

    assert isinstance(matrix, torch.Tensor)
    assert matrix.shape == (3, len(DISEASE_PROPERTY_NAMES))
    assert torch.allclose(matrix[0], torch.tensor(disease_property_vector("Lung Opacity")))
    assert torch.allclose(matrix[1], torch.tensor(disease_property_vector("Pleural Effusion")))

    with pytest.raises(ValueError, match="Unknown disease"):
        disease_property_vector("not a finding")
