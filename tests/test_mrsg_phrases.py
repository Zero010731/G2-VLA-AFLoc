import json

import pytest

from anaprior.data.mrsg_phrases import (
    MinedPhrase,
    build_counterfactuals,
    load_disease_descriptions,
    mine_report_phrases,
)


def _phrase_for(phrases, finding):
    return next(item for item in phrases if item.finding == finding)


def test_phrase_miner_preserves_location_laterality_and_severity() -> None:
    phrases = mine_report_phrases(
        "There is a small right apical pneumothorax. No left pneumothorax."
    )

    positive = [item for item in phrases if not item.negated]
    assert len(positive) == 1
    assert positive[0].finding == "Pneumothorax"
    assert positive[0].laterality == ("right",)
    assert positive[0].location_terms == ("apical",)
    assert positive[0].severity_terms == ("small",)
    assert positive[0].uncertain is False


def test_phrase_miner_keeps_bilateral_finding_as_both_sides() -> None:
    phrase = mine_report_phrases("Small bilateral pleural effusions.")[0]

    assert phrase.finding == "Pleural Effusion"
    assert phrase.laterality == ("left", "right")
    assert phrase.severity_terms == ("small",)
    assert phrase.negated is False


def test_phrase_miner_marks_uncertainty_without_turning_it_into_negation() -> None:
    phrase = mine_report_phrases(
        "Patchy right basilar opacity may represent pneumonia."
    )[-1]

    assert phrase.finding == "Pneumonia"
    assert phrase.laterality == ("right",)
    assert phrase.location_terms == ("basilar",)
    assert phrase.uncertain is True
    assert phrase.negated is False


def test_phrase_miner_preserves_explicit_negation() -> None:
    phrases = mine_report_phrases("No focal consolidation or pleural effusion.")

    assert {item.finding for item in phrases} == {
        "Consolidation",
        "Pleural Effusion",
    }
    assert all(item.negated for item in phrases)


def test_phrase_matching_respects_word_boundaries() -> None:
    phrases = mine_report_phrases(
        "The patient has edematous soft tissues but no pulmonary edema."
    )

    assert len(phrases) == 1
    assert phrases[0].finding == "Edema"
    assert phrases[0].negated is True


def test_counterfactual_generator_rejects_report_supported_laterality() -> None:
    report = "Small bilateral pleural effusions."
    phrase = mine_report_phrases(report)[0]

    negatives = build_counterfactuals(phrase, full_report=report, max_negatives=8)

    texts = {item.text.lower() for item in negatives}
    assert "small left pleural effusions." not in texts
    assert "small right pleural effusions." not in texts
    assert all(item.text.lower() not in report.lower() for item in negatives)


def test_counterfactual_generator_rejects_any_finding_supported_elsewhere() -> None:
    report = (
        "Small right apical pneumothorax. "
        "There is also mild left basilar atelectasis."
    )
    phrase = _phrase_for(mine_report_phrases(report), "Pneumothorax")

    negatives = build_counterfactuals(phrase, full_report=report, max_negatives=16)

    assert all(item.finding != "Atelectasis" for item in negatives)


def test_counterfactuals_are_text_only_and_preserve_source_polarity() -> None:
    phrase = mine_report_phrases("Possible mild right lower lung pneumonia.")[0]

    negatives = build_counterfactuals(
        phrase,
        full_report="Possible mild right lower lung pneumonia.",
        max_negatives=6,
    )

    assert negatives
    assert all(isinstance(item, MinedPhrase) for item in negatives)
    assert all(item.uncertain is True and item.negated is False for item in negatives)
    forbidden = {"box", "bbox", "mask", "region", "coordinate", "heatmap"}
    assert forbidden.isdisjoint(vars(negatives[0]))


def test_load_disease_descriptions_returns_all_textual_findings(tmp_path) -> None:
    descriptions = load_disease_descriptions()

    assert set(descriptions) == {
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Lung Opacity",
        "Pleural Effusion",
        "Pneumonia",
        "Pneumothorax",
    }
    assert all(isinstance(value, str) and value.strip() for value in descriptions.values())
    assert all(not isinstance(value, (list, dict)) for value in descriptions.values())

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"Edema": [1, 2, 3]}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty text"):
        load_disease_descriptions(bad_path)
