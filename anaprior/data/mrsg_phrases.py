"""Text-only phrase mining for box-free AFLoc-MRSG training."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


_FINDING_PATTERNS: Mapping[str, Tuple[str, ...]] = {
    "Atelectasis": (r"\batelect(?:asis|atic)\b",),
    "Cardiomegaly": (
        r"\bcardiomegaly\b",
        r"\benlarged (?:cardiac|cardiomediastinal) silhouette\b",
    ),
    "Consolidation": (r"\bconsolidations?\b", r"\bairspace consolidation\b"),
    "Edema": (r"\b(?:pulmonary )?edema\b",),
    "Lung Opacity": (
        r"\b(?:lung|pulmonary|airspace) opacit(?:y|ies)\b",
        r"\bopacit(?:y|ies)\b",
    ),
    "Pleural Effusion": (r"\bpleural effusions?\b",),
    "Pneumonia": (r"\bpneumonia\b",),
    "Pneumothorax": (r"\bpneumothor(?:ax|aces)\b",),
}

_CANONICAL_REPLACEMENTS: Mapping[str, str] = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "pulmonary edema",
    "Lung Opacity": "lung opacity",
    "Pleural Effusion": "pleural effusion",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
}

_LATERALITY_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "left": re.compile(r"\bleft(?:-sided)?\b", re.IGNORECASE),
    "right": re.compile(r"\bright(?:-sided)?\b", re.IGNORECASE),
    "bilateral": re.compile(r"\bbilateral(?:ly)?\b|\bbibasilar\b", re.IGNORECASE),
}

_LOCATION_PATTERNS: Mapping[str, re.Pattern[str]] = {
    term: re.compile(pattern, re.IGNORECASE)
    for term, pattern in {
        "apical": r"\bapical\b|\bapex\b",
        "basilar": r"\bbasilar\b|\bbasal\b|\bbase\b|\bbases\b|\bbibasilar\b",
        "upper": r"\bupper\b",
        "middle": r"\bmiddle\b|\bmid[ -]?lung\b",
        "lower": r"\blower\b",
        "perihilar": r"\bperihilar\b|\bhilar\b",
        "peripheral": r"\bperipheral\b",
        "central": r"\bcentral\b",
        "retrocardiac": r"\bretrocardiac\b",
        "costophrenic": r"\bcostophrenic\b",
    }.items()
}

_SEVERITY_PATTERNS: Mapping[str, re.Pattern[str]] = {
    term: re.compile(rf"\b{term}\b", re.IGNORECASE)
    for term in (
        "trace",
        "tiny",
        "minimal",
        "small",
        "mild",
        "moderate",
        "large",
        "marked",
        "severe",
    )
}

_NEGATION_PATTERN = re.compile(
    r"\b(?:no|without|absent|absence of|negative for|free of)\b",
    re.IGNORECASE,
)
_UNCERTAINTY_PATTERN = re.compile(
    r"\b(?:possible|possibly|probable|probably|likely|questionable|suspected|"
    r"suggestive of|may represent|could represent|cannot exclude|can not exclude)\b",
    re.IGNORECASE,
)
_LOCAL_SCOPE_BOUNDARY_PATTERN = re.compile(
    r"(?:\b(?:and|or|but|while|whereas|however|although|yet)\b|[,;:])",
    re.IGNORECASE,
)
_SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")

_LOCATION_REPLACEMENTS: Mapping[str, str] = {
    "apical": "apical",
    "basilar": "basilar",
    "upper": "upper",
    "middle": "middle",
    "lower": "lower",
    "perihilar": "perihilar",
    "peripheral": "peripheral",
    "central": "central",
    "retrocardiac": "retrocardiac",
    "costophrenic": "costophrenic",
}


@dataclass(frozen=True)
class MinedPhrase:
    """A report phrase represented exclusively by text-derived attributes."""

    text: str
    finding: str
    laterality: Tuple[str, ...]
    location_terms: Tuple[str, ...]
    severity_terms: Tuple[str, ...]
    uncertain: bool
    negated: bool


def _sentences(report: str) -> Iterable[str]:
    for match in _SENTENCE_PATTERN.finditer(report):
        sentence = match.group(0).strip()
        if sentence:
            yield sentence


def _finding_matches(sentence: str) -> Iterable[Tuple[int, int, str]]:
    accepted: List[Tuple[int, int, str]] = []
    for finding, patterns in _FINDING_PATTERNS.items():
        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, sentence, re.IGNORECASE))
        selected: List[Tuple[int, int, str]] = []
        for match in sorted(
            matches,
            key=lambda item: (item.start(), -len(item.group(0))),
        ):
            span = (match.start(), match.end(), finding)
            if any(span[0] < item[1] and item[0] < span[1] for item in selected):
                continue
            selected.append(span)
        accepted.extend(selected)
    yield from sorted(accepted, key=lambda item: (item[0], item[1], item[2]))


def _local_scopes(
    sentence: str,
    mentions: Sequence[Tuple[int, int, str]],
) -> Iterable[Tuple[str, int, int, Tuple[Tuple[int, int, str], ...]]]:
    boundaries = []
    for boundary in _LOCAL_SCOPE_BOUNDARY_PATTERN.finditer(sentence):
        connector = boundary.group(0).strip().lower()
        is_hard_boundary = connector in {
            ";",
            ":",
            "but",
            "while",
            "whereas",
            "however",
            "although",
            "yet",
        }
        has_left_mention = any(end <= boundary.start() for _, end, _ in mentions)
        has_right_mention = any(start >= boundary.end() for start, _, _ in mentions)
        if is_hard_boundary or (has_left_mention and has_right_mention):
            boundaries.append(boundary)

    scope_start = 0
    connector = ""
    for boundary in (*boundaries, None):
        scope_end = boundary.start() if boundary is not None else len(sentence)
        scoped_mentions = tuple(
            mention
            for mention in mentions
            if mention[0] >= scope_start and mention[1] <= scope_end
        )
        if scoped_mentions:
            yield connector, scope_start, scope_end, scoped_mentions
        if boundary is not None:
            connector = boundary.group(0).strip().lower()
            scope_start = boundary.end()


def _nearest_terms(
    sentence: str,
    finding_start: int,
    patterns: Mapping[str, re.Pattern[str]],
) -> Tuple[str, ...]:
    candidates = []
    for term, pattern in patterns.items():
        for match in pattern.finditer(sentence):
            if match.end() <= finding_start:
                distance = finding_start - match.end()
                direction = 0
            else:
                distance = match.start() - finding_start
                direction = 1
            candidates.append((distance, direction, match.start(), term))
    if not candidates:
        return ()
    nearest_distance = min(item[0] for item in candidates)
    return tuple(
        item[3]
        for item in sorted(candidates, key=lambda value: (value[0], value[1], value[2]))
        if item[0] == nearest_distance
    )


def _laterality(sentence: str, finding_start: int) -> Tuple[str, ...]:
    bilateral = list(_LATERALITY_PATTERNS["bilateral"].finditer(sentence))
    if bilateral:
        nearest_bilateral = min(abs(match.start() - finding_start) for match in bilateral)
        unilateral = [
            match
            for side in ("left", "right")
            for match in _LATERALITY_PATTERNS[side].finditer(sentence)
        ]
        nearest_unilateral = min(
            (abs(match.start() - finding_start) for match in unilateral),
            default=None,
        )
        if nearest_unilateral is None or nearest_bilateral <= nearest_unilateral:
            return ("left", "right")
    left_matches = list(_LATERALITY_PATTERNS["left"].finditer(sentence))
    right_matches = list(_LATERALITY_PATTERNS["right"].finditer(sentence))
    for left in left_matches:
        for right in right_matches:
            first, second = sorted((left, right), key=lambda match: match.start())
            connector = sentence[first.end() : second.start()]
            both_before = second.end() <= finding_start
            both_after = first.start() >= finding_start
            if (
                re.fullmatch(r"\s*(?:and|or|/|&)\s*", connector, re.IGNORECASE)
                and (both_before or both_after)
            ):
                return ("left", "right")
    return _nearest_terms(
        sentence,
        finding_start,
        {key: value for key, value in _LATERALITY_PATTERNS.items() if key != "bilateral"},
    )


def mine_report_phrases(report: str) -> Tuple[MinedPhrase, ...]:
    """Extract study findings and their text attributes using frozen rules."""

    if not isinstance(report, str):
        raise TypeError("report must be text")
    phrases: List[MinedPhrase] = []
    for sentence in _sentences(report):
        mentions = tuple(_finding_matches(sentence))
        carry_or_negation = False
        for connector, scope_start, scope_end, scoped_mentions in _local_scopes(
            sentence,
            mentions,
        ):
            raw_scope = sentence[scope_start:scope_end]
            leading_whitespace = len(raw_scope) - len(raw_scope.lstrip())
            scope = raw_scope.strip()
            first_start = scoped_mentions[0][0] - scope_start - leading_whitespace
            leading_context = scope[:first_start]
            local_leading_negation = bool(_NEGATION_PATTERN.search(leading_context))
            inherited_negation = connector == "or" and carry_or_negation
            for start, _, finding in scoped_mentions:
                local_start = start - scope_start - leading_whitespace
                preceding_context = scope[:local_start]
                phrases.append(
                    MinedPhrase(
                        text=scope,
                        finding=finding,
                        laterality=_laterality(scope, local_start),
                        location_terms=_nearest_terms(
                            scope,
                            local_start,
                            _LOCATION_PATTERNS,
                        ),
                        severity_terms=_nearest_terms(
                            scope,
                            local_start,
                            _SEVERITY_PATTERNS,
                        ),
                        uncertain=bool(
                            _UNCERTAINTY_PATTERN.search(preceding_context)
                        ),
                        negated=(
                            inherited_negation
                            or bool(_NEGATION_PATTERN.search(preceding_context))
                        ),
                    )
                )
            carry_or_negation = local_leading_negation or inherited_negation
    return tuple(phrases)


def _replace_first(text: str, patterns: Sequence[str], replacement: str) -> Optional[str]:
    matches = []
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match is not None:
            matches.append(match)
    if not matches:
        return None
    match = min(matches, key=lambda item: (item.start(), -len(item.group(0))))
    return text[: match.start()] + replacement + text[match.end() :]


def _supported_phrases(report: str) -> Tuple[MinedPhrase, ...]:
    return tuple(item for item in mine_report_phrases(report) if not item.negated)


def _candidate_is_supported(
    candidate: MinedPhrase,
    supported: Sequence[MinedPhrase],
    replacement_kind: str,
) -> bool:
    for evidence in supported:
        if evidence.finding != candidate.finding:
            continue
        if replacement_kind == "finding":
            return True
        if (
            replacement_kind == "laterality"
            and set(candidate.laterality) & set(evidence.laterality)
        ):
            return True
        if (
            replacement_kind == "location"
            and set(candidate.location_terms) & set(evidence.location_terms)
        ):
            return True
    return False


def _append_candidate(
    output: List[MinedPhrase],
    candidate: MinedPhrase,
    supported: Sequence[MinedPhrase],
    replacement_kind: str,
    normalized_report: str,
    seen: set,
) -> None:
    normalized_text = " ".join(candidate.text.lower().split())
    if not normalized_text or normalized_text in normalized_report:
        return
    if normalized_text in seen or _candidate_is_supported(
        candidate,
        supported,
        replacement_kind,
    ):
        return
    seen.add(normalized_text)
    output.append(candidate)


def build_counterfactuals(
    phrase: MinedPhrase,
    full_report: str,
    max_negatives: int = 8,
) -> Tuple[MinedPhrase, ...]:
    """Build conservative text counterfactuals absent from the complete report."""

    if not isinstance(phrase, MinedPhrase):
        raise TypeError("phrase must be a MinedPhrase")
    if not isinstance(full_report, str):
        raise TypeError("full_report must be text")
    if max_negatives < 0:
        raise ValueError("max_negatives must be non-negative")
    if max_negatives == 0 or phrase.negated:
        return ()

    supported = _supported_phrases(full_report)
    normalized_report = " ".join(full_report.lower().split())
    candidates: List[MinedPhrase] = []
    seen = set()

    source_patterns = _FINDING_PATTERNS[phrase.finding]
    for finding, replacement_text in _CANONICAL_REPLACEMENTS.items():
        if finding == phrase.finding:
            continue
        replaced_text = _replace_first(phrase.text, source_patterns, replacement_text)
        if replaced_text is None:
            continue
        candidate = replace(phrase, text=replaced_text, finding=finding)
        _append_candidate(
            candidates,
            candidate,
            supported,
            "finding",
            normalized_report,
            seen,
        )
        if len(candidates) >= max_negatives:
            return tuple(candidates)

    if phrase.laterality:
        source_sides = set(phrase.laterality)
        source_patterns_laterality: List[str] = []
        if source_sides == {"left", "right"}:
            source_patterns_laterality.extend(
                (r"\bbilateral(?:ly)?\b", r"\bbibasilar\b")
            )
        else:
            source_patterns_laterality.extend(
                rf"\b{side}(?:-sided)?\b" for side in sorted(source_sides)
            )
        for target_side in ("left", "right"):
            if source_sides == {target_side}:
                continue
            replaced_text = _replace_first(
                phrase.text,
                source_patterns_laterality,
                target_side,
            )
            if replaced_text is None:
                continue
            candidate = replace(
                phrase,
                text=replaced_text,
                laterality=(target_side,),
            )
            _append_candidate(
                candidates,
                candidate,
                supported,
                "laterality",
                normalized_report,
                seen,
            )
            if len(candidates) >= max_negatives:
                return tuple(candidates)

    for source_location in phrase.location_terms:
        source_pattern = _LOCATION_PATTERNS.get(source_location)
        if source_pattern is None:
            continue
        for target_location, replacement_text in _LOCATION_REPLACEMENTS.items():
            if target_location == source_location:
                continue
            replaced_text = _replace_first(
                phrase.text,
                (source_pattern.pattern,),
                replacement_text,
            )
            if replaced_text is None:
                continue
            candidate = replace(
                phrase,
                text=replaced_text,
                location_terms=(target_location,),
            )
            _append_candidate(
                candidates,
                candidate,
                supported,
                "location",
                normalized_report,
                seen,
            )
            if len(candidates) >= max_negatives:
                return tuple(candidates)

    return tuple(candidates)


def load_disease_descriptions(
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Load descriptive disease text used by the later text encoder."""

    source = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[1]
        / "configs"
        / "mrsg_disease_descriptions.json"
    )
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Disease descriptions must be a JSON object")
    descriptions: Dict[str, str] = {}
    for finding, description in raw.items():
        if not isinstance(finding, str) or not finding.strip():
            raise ValueError("Disease description keys must be non-empty text")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Disease descriptions must contain non-empty text")
        descriptions[finding] = description.strip()
    return descriptions


__all__ = [
    "MinedPhrase",
    "build_counterfactuals",
    "load_disease_descriptions",
    "mine_report_phrases",
]
