# Gated DCEM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a disease-conditioned gated learned repair branch while preserving the original raw learned repair logic and outputs.

**Architecture:** Keep predictor inference and raw `learned_selective` unchanged. Add a small disease-conditioned score transform and gate layer that can be invoked as a new repair score mode, then register a new `disease_gated_learned` method alongside existing baselines.

**Tech Stack:** Python, NumPy, pytest, existing AnaPrior evaluation modules.

## Global Constraints

- Do not overwrite or redefine `learned_selective`; it remains the raw learned repair baseline.
- Gate failure must bypass repair by returning zero usable evidence, so existing baseline-copy behavior is preserved.
- First implementation must be deterministic and config-driven, with no MS-CXR metric tuning.
- Tests must be written and observed failing before production code changes.

---

### Task 1: Disease-Conditioned Gate Core

**Files:**
- Create: `anaprior/eval/disease_conditioned_gate.py`
- Create: `tests/test_disease_conditioned_gate.py`

**Interfaces:**
- Produces: `apply_disease_conditioned_gate(category: str, regions: list[str], scores: np.ndarray) -> GatedEvidence`
- Produces: `GatedEvidence(scores: np.ndarray, status: str, reason: str, top_region: str, top_score: float)`

- [ ] **Step 1: Write failing tests**
  - Verify Pneumothorax blocks cardiac/hilar top evidence.
  - Verify Pleural Effusion downweights cardiac/hilar but preserves pleural/lower evidence.
  - Verify unknown categories pass raw scores through.

- [ ] **Step 2: Run tests and observe import/function failure**
  - Run: `pytest tests/test_disease_conditioned_gate.py -q`
  - Expected: FAIL because module/function does not exist.

- [ ] **Step 3: Implement minimal gate core**
  - Add dataclasses and default disease configs for supported findings.
  - Apply blocked regions, allowed region constraints, region weights, and top-region plausibility.

- [ ] **Step 4: Run tests and verify green**
  - Run: `pytest tests/test_disease_conditioned_gate.py -q`
  - Expected: PASS.

### Task 2: Repair Core Integration

**Files:**
- Modify: `anaprior/eval/selective_repair.py`
- Modify: `tests/test_selective_repair_core.py`

**Interfaces:**
- Consumes: `apply_disease_conditioned_gate(...)`
- Produces: `ScoreMode = Literal["learned", "shuffled", "uniform", "disease_gated"]`
- Extends: `RepairCase` with `regions: list[str] | None = None`

- [ ] **Step 1: Write failing tests**
  - Verify `score_mode="disease_gated"` bypasses a Pneumothorax case whose top evidence is cardiac.
  - Verify raw `score_mode="learned"` still repairs that same case.

- [ ] **Step 2: Run tests and observe failure**
  - Run: `pytest tests/test_selective_repair_core.py -q`
  - Expected: FAIL because `disease_gated` is unsupported.

- [ ] **Step 3: Implement integration**
  - Add optional `regions` to `RepairCase`.
  - Route `disease_gated` through `apply_disease_conditioned_gate`.
  - Add gate status counters without removing existing counters.

- [ ] **Step 4: Run tests and verify green**
  - Run: `pytest tests/test_selective_repair_core.py -q`
  - Expected: PASS.

### Task 3: Method Registration and Metrics Comparison

**Files:**
- Modify: `anaprior/eval/eval_mscxr_learned_repair.py`
- Modify: `anaprior/eval/score_mscxr_learned_repair_metrics.py`
- Modify: `tests/test_eval_mscxr_learned_repair.py`
- Modify: `tests/test_score_mscxr_learned_repair_metrics.py`

**Interfaces:**
- Produces method: `disease_gated_learned`
- Produces comparisons: `disease_gated_learned_vs_baseline`, `disease_gated_learned_vs_learned_selective`, `disease_gated_learned_vs_candidate_shuffled`

- [ ] **Step 1: Write failing tests**
  - Verify `build_method_hmaps` includes `disease_gated_learned`.
  - Verify method stats keep raw `learned_selective` untouched.
  - Verify metrics comparisons include gated-vs-raw and gated-vs-baseline.

- [ ] **Step 2: Run tests and observe failure**
  - Run: `pytest tests/test_eval_mscxr_learned_repair.py tests/test_score_mscxr_learned_repair_metrics.py -q`
  - Expected: FAIL because gated method/comparisons are absent.

- [ ] **Step 3: Implement method registration**
  - Add `("disease_gated_learned", "candidate", "disease_gated")` to method specs.
  - Add gated method to default metrics methods and comparison specs.
  - Pass `regions` into `RepairCase`.

- [ ] **Step 4: Run targeted and full tests**
  - Run: `pytest tests/test_disease_conditioned_gate.py tests/test_selective_repair_core.py tests/test_eval_mscxr_learned_repair.py tests/test_score_mscxr_learned_repair_metrics.py -q`
  - Run: `pytest -q`
  - Expected: PASS.
