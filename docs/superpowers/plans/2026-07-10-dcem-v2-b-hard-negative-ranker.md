# DCEM-v2-B Hard-Negative Ranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DCEM-v2-B as a frozen-AFLoc, Chest-ImaGenome-trained hard-negative region ranker with disease-specific pooling, while preserving all DCEM-v1/raw learned controls.

**Architecture:** Stage B keeps the existing BCE region predictor and optionally adds disease-specific margin ranking over positive and hard-negative regions from the same finding. Stage C adds a new `disease_pooled_learned` heatmap method using disease-specific score pooling/region priors, scored beside `learned_selective` and `disease_gated_learned`.

**Tech Stack:** Python 3.9, PyTorch, NumPy, pandas, pytest, existing AnaPrior Stage B/C runners.

## Global Constraints

- Do not change AFLoc backbone or image/text encoders.
- Do not train on MS-CXR boxes or masks.
- Do not remove raw learned, disease-gated, shuffled, or uniform controls.
- Any pass/bypass claim must still come from validation gate, not test inspection.
- Keep DCEM-v2-B opt-in through CLI/env flags so DCEM-v1 remains reproducible.

---

### Task 1: Hard-Negative Rule Config

**Files:**
- Create: `anaprior/train/dcem_v2_hard_negatives.py`
- Test: `tests/test_dcem_v2_hard_negatives.py`

**Interfaces:**
- Produces: `is_hard_negative(category: str, region: str) -> bool`
- Produces: `hard_negative_regions_for(category: str) -> frozenset[str]`
- Produces: `positive_prior_regions_for(category: str) -> frozenset[str]`

- [ ] Write tests for Pneumothorax, Cardiomegaly, Consolidation, Lung Opacity, Edema, Pneumonia, Pleural Effusion.
- [ ] Implement normalized disease/region rules.
- [ ] Run `python -m pytest tests/test_dcem_v2_hard_negatives.py -q`.

### Task 2: Margin Ranking Loss

**Files:**
- Modify: `anaprior/train/train_region_predictor.py`
- Test: `tests/test_train_region_predictor.py`

**Interfaces:**
- Produces: `build_hard_negative_pairs(labels, finding_ids, region_names, finding_vocab, max_pairs_per_finding=4096) -> list[tuple[int, int]]`
- Produces: `hard_negative_ranking_loss(model, features, finding_ids, pairs, margin, batch_size=None) -> torch.Tensor`
- Extends: `TrainConfig` with `rank_loss_weight`, `rank_margin`, `max_rank_pairs_per_finding`

- [ ] Write failing tests for pair construction and positive margin behavior.
- [ ] Implement CPU-safe pair construction from feature-cache `region_names`.
- [ ] Add optional rank loss into the training loop only when `rank_loss_weight > 0`.
- [ ] Save rank config and pair counts into checkpoint/report.
- [ ] Run `python -m pytest tests/test_train_region_predictor.py -q`.

### Task 3: Stage B-v2 Runner Flags

**Files:**
- Modify: `scripts/run_stage_b_8class_predictor_server.sh`
- Test: `tests/test_stage_b_server_script.py`

**Interfaces:**
- Consumes env `DCEM_V2_RANK_LOSS_WEIGHT`, `DCEM_V2_RANK_MARGIN`, `DCEM_V2_MAX_RANK_PAIRS_PER_FINDING`
- Passes `--rank-loss-weight`, `--rank-margin`, `--max-rank-pairs-per-finding`

- [ ] Add a shell-script static test for the new flags.
- [ ] Modify the runner to print and pass the flags.
- [ ] Run `python -m pytest tests/test_stage_b_server_script.py -q`.

### Task 4: Disease-Specific Pooling Branch

**Files:**
- Modify: `anaprior/eval/disease_conditioned_gate.py`
- Modify: `anaprior/eval/selective_repair.py`
- Test: `tests/test_disease_conditioned_gate.py`
- Test: `tests/test_selective_repair_core.py`

**Interfaces:**
- Produces: `apply_disease_specific_pooling(category, regions, scores) -> GatedEvidence`
- Extends `ScoreMode` with `disease_pooled`

- [ ] Write tests that Pneumothorax top-k-like pooling keeps pleural/upper scores and suppresses cardiac/hilar without bypassing.
- [ ] Write tests that Edema diffuse pooling keeps bilateral/global evidence.
- [ ] Implement disease-pooled score transformation separate from existing strict gate.
- [ ] Run targeted tests.

### Task 5: Stage C-v2 Method Output and Metrics

**Files:**
- Modify: `anaprior/eval/eval_mscxr_learned_repair.py`
- Modify: `anaprior/eval/score_mscxr_learned_repair_metrics.py`
- Modify: `scripts/run_stage_c_learned_repair_server.sh`
- Test: `tests/test_eval_mscxr_learned_repair.py`
- Test: `tests/test_score_mscxr_learned_repair_metrics.py`
- Test: `tests/test_stage_c_server_script.py`

**Interfaces:**
- Produces method `disease_pooled_learned`
- Produces comparison `disease_pooled_learned_vs_baseline`
- Validation gate source remains configurable and defaults to `disease_gated_learned` for v1 reproducibility.

- [ ] Add method generation test.
- [ ] Add metric comparison test.
- [ ] Add runner static test for `STAGE_C_METHODS` and `VALIDATION_GATE_SOURCE_METHOD`.
- [ ] Implement method and comparison while preserving defaults.
- [ ] Run targeted tests and full `python -m pytest -q`.
