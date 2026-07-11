from anaprior.eval.phrase_subtype import (
    PHRASE_SUBTYPE_TO_ID,
    PHRASE_SUBTYPES,
    infer_phrase_subtype,
    phrase_subtype_id,
)


def test_phrase_subtype_vocab_is_stable_and_has_other_fallback() -> None:
    assert PHRASE_SUBTYPES[0] == "basilar"
    assert PHRASE_SUBTYPES[-1] == "other"
    assert PHRASE_SUBTYPE_TO_ID["other"] == len(PHRASE_SUBTYPES) - 1


def test_infer_phrase_subtype_prioritizes_anatomic_scale_terms() -> None:
    assert infer_phrase_subtype("left basilar opacity").name == "basilar"
    assert infer_phrase_subtype("patchy multifocal airspace opacities").name == "multifocal_patchy"
    assert infer_phrase_subtype("diffuse bilateral opacities").name == "bilateral_diffuse"
    assert infer_phrase_subtype("right apical pneumothorax").name == "upper_apical"
    assert infer_phrase_subtype("right middle lobe opacity").name == "mid_lung"


def test_infer_phrase_subtype_detects_texture_and_disease_like_terms() -> None:
    assert infer_phrase_subtype("ground-glass opacity in the left upper lobe").name == "ground_glass"
    assert infer_phrase_subtype("airspace opacity suspicious for infection").name == "airspace"
    assert infer_phrase_subtype("retrocardiac consolidation").name == "retrocardiac_hilar"
    assert infer_phrase_subtype("opacity compatible with pneumonia").name == "pneumonia_like"
    assert infer_phrase_subtype("dense consolidation").name == "consolidation_like"
    assert infer_phrase_subtype("persistent opacity").name == "opacity_like"


def test_phrase_subtype_id_matches_vocab() -> None:
    subtype = infer_phrase_subtype("small focal area of consolidation")

    assert subtype.name == "focal"
    assert phrase_subtype_id("small focal area of consolidation") == PHRASE_SUBTYPE_TO_ID["focal"]
    assert subtype.matched_terms
