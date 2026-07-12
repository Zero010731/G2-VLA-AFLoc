# DP-MSA-v2 Property-Conditioned Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build DP-MSA-v2 as a shared property-conditioned residual refinement adapter over DCEM-v3 heatmaps.

**Architecture:** Add a frozen disease spatial property vocabulary and encoder. Extend the existing DP-MSA adapter so disease names are represented by property vectors, and shared multi-scale branches choose repair behavior from disease properties plus phrase subtype rather than from disease-specific hand-written branches.

**Tech Stack:** Python 3.9, PyTorch, pytest, existing AnaPrior modules.

## Global Constraints

- Do not hand-write one repair branch per disease.
- Do not use MS-CXR boxes, masks, or oracle profiles as training labels.
- Keep DCEM-v3 as the base heatmap for DP-MSA-v2 evaluation.
- Preserve the current DP-MSA-v0 `disease_ids` interface for existing callers.
- Add tests before production code.

---

### Task 1: Frozen Disease Spatial Properties

**Files:**
- Create: `anaprior/eval/disease_properties.py`
- Test: `tests/test_disease_properties.py`

**Interfaces:**
- Produces: `DISEASE_PROPERTY_NAMES: tuple[str, ...]`
- Produces: `DISEASE_PROPERTY_TABLE: Mapping[str, tuple[float, ...]]`
- Produces: `disease_property_vector(category: str) -> tuple[float, ...]`
- Produces: `disease_property_matrix(categories: Sequence[str]) -> torch.Tensor`

- [ ] Write tests that properties are frozen, normalized, and disease names map through aliases.
- [ ] Run `python -m pytest tests/test_disease_properties.py -q` and confirm failure because the module does not exist.
- [ ] Implement the property vocabulary and table.
- [ ] Run `python -m pytest tests/test_disease_properties.py -q`.

### Task 2: Property-Conditioned Adapter Path

**Files:**
- Modify: `anaprior/models/dp_msa_adapter.py`
- Test: `tests/test_dp_msa_adapter.py`

**Interfaces:**
- Extends `DPMultiScaleSpatialAdapter.__init__` with `num_disease_properties: int | None = None`
- Extends `DPMultiScaleSpatialAdapter.forward` with keyword-only `disease_properties: torch.Tensor | None = None`
- Produces 5 branch weights for property-conditioned mode.

- [ ] Write tests that passing disease property vectors changes the condition path and returns 5 shared branch weights.
- [ ] Run targeted tests and confirm failure.
- [ ] Implement property projection, five shared branches, and backwards-compatible id embedding fallback.
- [ ] Run `python -m pytest tests/test_dp_msa_adapter.py -q`.

### Task 3: Verification

**Files:**
- Test only.

- [ ] Run `python -m pytest tests/test_disease_properties.py tests/test_dp_msa_adapter.py tests/test_phrase_subtype.py -q`.
- [ ] Run broader relevant tests if runtime is reasonable.
