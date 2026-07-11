# DCEM-v3 Phrase-Anatomy Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build DCEM-v3 as a phrase-anatomy router plus disease-specific pooling method for frozen-AFLoc learned repair.

**Architecture:** Add a deterministic phrase parser and router in a new evaluation module, route it through the existing selective repair score-mode interface, register a new `phrase_anatomy_dcem` Stage C method, and allow a named `validation_gated_dcem_v3` gate selected on validation only. Existing raw learned, disease-gated, disease-pooled, shuffled, and uniform controls remain unchanged.

**Tech Stack:** Python, NumPy, pandas, pytest, existing AnaPrior Stage C evaluation scripts.

## Global Constraints

- Do not change AFLoc backbone, image encoder, or text encoder.
- Do not use MS-CXR boxes, masks, or box-derived oracle profiles for training, tuning, phrase lexicon design, gate threshold selection, or model selection.
- MS-CXR boxes remain final held-out evaluation only after the method and validation gate protocol are frozen.
- Keep DCEM-v2B frozen outputs untouched.
- Keep raw `learned_selective`, `disease_gated_learned`, `disease_pooled_learned`, shuffled, and uniform controls available.
- Use TDD for each behavior change.

---

### Task 1: Phrase-Anatomy Router Core

**Files:**
- Create: `anaprior/eval/phrase_anatomy_router.py`
- Test: `tests/test_phrase_anatomy_router.py`

**Interfaces:**
- Produces: `PhraseAnatomy(laterality: frozenset[str], vertical: frozenset[str], compartments: frozenset[str], tokens: frozenset[str])`
- Produces: `parse_phrase_anatomy(phrase: str) -> PhraseAnatomy`
- Produces: `apply_phrase_anatomy_router(category: str, phrase: str, regions: list[str], scores: np.ndarray) -> GatedEvidence`

- [ ] **Step 1: Write failing parser/router tests**
  - Test left/right/bilateral parsing.
  - Test upper/apical/lower/basal parsing.
  - Test pleural/cardiac/hilar compartment parsing.
  - Test Pneumothorax phrase routing suppresses cardiac/hilar and keeps apical/pleural evidence.
  - Test Pneumonia lower-right phrase upweights right lower lung over cardiac shortcut.
  - Test missing phrase falls back to disease pooling.

- [ ] **Step 2: Run router tests and verify RED**
  - Run: `pytest tests/test_phrase_anatomy_router.py -q`
  - Expected: FAIL because `anaprior.eval.phrase_anatomy_router` is missing.

- [ ] **Step 3: Implement router core**
  - Add frozen anatomical lexicons.
  - Implement parser with generic medical/anatomy vocabulary only.
  - Implement disease plus phrase region weighting using existing `GatedEvidence`.
  - Never inspect or import MS-CXR box data.

- [ ] **Step 4: Run router tests and verify GREEN**
  - Run: `pytest tests/test_phrase_anatomy_router.py -q`
  - Expected: PASS.

### Task 2: Selective Repair Integration

**Files:**
- Modify: `anaprior/eval/selective_repair.py`
- Test: `tests/test_selective_repair_core.py`

**Interfaces:**
- Consumes: `apply_phrase_anatomy_router(category, phrase, regions, scores) -> GatedEvidence`
- Extends: `RepairCase` with `phrase: str = ""`
- Extends: `ScoreMode` with `"phrase_anatomy"`

- [ ] **Step 1: Write failing repair integration tests**
  - Test `score_mode="phrase_anatomy"` uses phrase hints to repair a right-apical Pneumothorax region instead of a cardiac shortcut.
  - Test raw `score_mode="learned"` remains unchanged for the same case.

- [ ] **Step 2: Run selective repair tests and verify RED**
  - Run: `pytest tests/test_selective_repair_core.py::test_phrase_anatomy_mode_routes_pneumothorax_by_phrase_without_changing_raw_learned -q`
  - Expected: FAIL because `phrase_anatomy` is unsupported.

- [ ] **Step 3: Implement repair integration**
  - Add `phrase` to `RepairCase`.
  - Route `phrase_anatomy` through `apply_phrase_anatomy_router`.
  - Add phrase gate pass/bypass stats without changing existing stats keys.

- [ ] **Step 4: Run selective repair tests and verify GREEN**
  - Run: `pytest tests/test_selective_repair_core.py -q`
  - Expected: PASS.

### Task 3: Stage C Heatmap Method Registration

**Files:**
- Modify: `anaprior/eval/eval_mscxr_learned_repair.py`
- Modify: `anaprior/eval/prepare_mscxr_repair_inputs.py`
- Test: `tests/test_eval_mscxr_learned_repair.py`

**Interfaces:**
- Extends: `LearnedRepairInput(..., phrase: str = "")`
- Produces method: `phrase_anatomy_dcem`
- Preserves methods: `baseline`, `learned_selective`, `disease_gated_learned`, `disease_pooled_learned`, `all_class_learned`, `candidate_shuffled`, `candidate_uniform`

- [ ] **Step 1: Write failing method registration tests**
  - Test `build_method_hmaps` returns `phrase_anatomy_dcem`.
  - Test phrase text from `LearnedRepairInput.phrase` is passed into `RepairCase`.
  - Test old prepared input payloads without `phrase` still load by falling back to `finding`.

- [ ] **Step 2: Run Stage C method tests and verify RED**
  - Run: `pytest tests/test_eval_mscxr_learned_repair.py -q`
  - Expected: FAIL because `phrase_anatomy_dcem` is missing.

- [ ] **Step 3: Implement Stage C registration**
  - Add `phrase` to `LearnedRepairInput`.
  - Save/load `phrase` in prepared input NPZ.
  - Set `phrase=label_text` when preparing MS-CXR repair inputs.
  - Add `("phrase_anatomy_dcem", "candidate", "phrase_anatomy")` to method specs.

- [ ] **Step 4: Run Stage C method tests and verify GREEN**
  - Run: `pytest tests/test_eval_mscxr_learned_repair.py -q`
  - Expected: PASS.

### Task 4: Metrics and Validation Gate v3

**Files:**
- Modify: `anaprior/eval/score_mscxr_learned_repair_metrics.py`
- Test: `tests/test_score_mscxr_learned_repair_metrics.py`

**Interfaces:**
- Adds default method: `phrase_anatomy_dcem`
- Adds comparisons:
  - `phrase_anatomy_dcem_vs_baseline`
  - `phrase_anatomy_dcem_vs_learned_selective`
  - `phrase_anatomy_dcem_vs_candidate_shuffled`
  - `validation_gated_dcem_v3_vs_baseline`
  - `validation_gated_dcem_v3_vs_phrase_anatomy_dcem`
  - `validation_gated_dcem_v3_vs_disease_pooled_learned`
  - `validation_gated_dcem_v3_vs_candidate_shuffled`
- Adds CLI argument: `--validation-gate-method-name`, default `validation_gated_dcem`

- [ ] **Step 1: Write failing metrics tests**
  - Test default methods include `phrase_anatomy_dcem`.
  - Test comparisons include phrase anatomy comparisons.
  - Test custom validation gate method name creates `validation_gated_dcem_v3` per-case records.

- [ ] **Step 2: Run metrics tests and verify RED**
  - Run: `pytest tests/test_score_mscxr_learned_repair_metrics.py::test_default_methods_include_gated_dcem_without_removing_raw_learned_comparisons -q`
  - Expected: FAIL because phrase anatomy comparisons are missing.

- [ ] **Step 3: Implement metrics support**
  - Add phrase method/comparisons to defaults.
  - Allow `build_validation_gated_per_case(..., method_name="validation_gated_dcem_v3")`.
  - Add dynamic comparison specs for custom validation gate method names.
  - Write validation gate JSON using the requested method name.

- [ ] **Step 4: Run metrics tests and verify GREEN**
  - Run: `pytest tests/test_score_mscxr_learned_repair_metrics.py -q`
  - Expected: PASS.

### Task 5: Script and Result Pack Support

**Files:**
- Modify: `scripts/run_stage_c_learned_repair_server.sh`
- Modify: `anaprior/eval/report_dcem_v1_paper_results.py`
- Test: `tests/test_stage_c_server_script.py`
- Test: `tests/test_report_dcem_v1_paper_results.py`

**Interfaces:**
- Script env: `VALIDATION_GATE_METHOD_NAME`
- Report supports:
  - `phrase_anatomy_dcem_vs_baseline`
  - `validation_gated_dcem_v3_vs_baseline`
  - `validation_gated_dcem_v3_vs_phrase_anatomy_dcem`

- [ ] **Step 1: Write failing script/report tests**
  - Test Stage C runner passes `--validation-gate-method-name`.
  - Test report tables include phrase-anatomy and v3 validation columns when present.

- [ ] **Step 2: Run script/report tests and verify RED**
  - Run: `pytest tests/test_stage_c_server_script.py tests/test_report_dcem_v1_paper_results.py -q`
  - Expected: FAIL because v3 script/report support is missing.

- [ ] **Step 3: Implement script/report support**
  - Add `VALIDATION_GATE_METHOD_NAME` to Stage C runner.
  - Add phrase/v3 comparisons to report result pack table mapping.
  - Preserve DCEM-v1/DCEM-v2B output behavior through `method_name` and `output_prefix`.

- [ ] **Step 4: Run targeted and full tests**
  - Run: `pytest tests/test_phrase_anatomy_router.py tests/test_selective_repair_core.py tests/test_eval_mscxr_learned_repair.py tests/test_score_mscxr_learned_repair_metrics.py tests/test_stage_c_server_script.py tests/test_report_dcem_v1_paper_results.py -q`
  - Run: `pytest -q`
  - Expected: all tests PASS.
