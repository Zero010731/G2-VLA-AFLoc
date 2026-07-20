# AFLoc-MRSG Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize bounded residuals, enforce correction-level geometric correspondence, and add same-image phrase-swap correction contrast without disease rules or spatial annotations.

**Architecture:** Cap normalized residual logits before bounded correction, then train three sequential grounding stages initialized from the preceding checkpoint. Phase 2A activates residual stability, 2B adds correction equivariance, and 2C adds negative-phrase correction contrast. AFLoc remains frozen and official-anchor composition remains mandatory.

**Tech Stack:** Python 3.9, PyTorch, NumPy, SciPy, pytest, Bash.

## Global Constraints

- No MS-CXR boxes or masks in training.
- No DCEM, anatomy annotations, region predictor, disease ID, or per-disease gate.
- No EMA in Phase 2.
- Phase 2 selection uses validation only; test evaluation is not part of intermediate stages.
- Existing Phase 1 checkpoint is the Phase 2A initializer, not a fallback heatmap.

---

### Task 1: Residual Stabilization

**Files:**
- Modify: `anaprior/models/afloc_mrsg/contracts.py`
- Modify: `anaprior/models/afloc_mrsg/dense_decoder.py`
- Modify: `anaprior/models/afloc_mrsg/losses.py`
- Modify corresponding tests.

- [ ] Test normalized/capped residual logits and finite gradients.
- [ ] Add pre-head GroupNorm and configurable residual logit cap.
- [ ] Add residual energy and saturation loss with diagnostics.

### Task 2: Correction-Level Cross-View

**Files:**
- Modify: `anaprior/train/train_afloc_mrsg.py`
- Modify: `anaprior/models/afloc_mrsg/losses.py`
- Modify corresponding tests.

- [ ] Test geometry-aligned correction consistency.
- [ ] Compute the extra equivariant forward only when its weight is active.
- [ ] Compare transformed signed bounded corrections, not final heatmaps.

### Task 3: Phrase-Swap Correction Contrast

**Files:**
- Modify: `anaprior/models/afloc_mrsg/losses.py`
- Modify: `anaprior/train/train_afloc_mrsg.py`
- Modify corresponding tests.

- [ ] Test that identical positive/negative corrections are penalized and distinct corrections reduce loss.
- [ ] Reuse negative-phrase forwards and aggregate only valid phrase pairs.
- [ ] Keep negative padded rows out of the loss.

### Task 4: Sequential Initialization and Diagnostics

**Files:**
- Modify: `anaprior/train/train_afloc_mrsg.py`
- Modify: `anaprior/models/afloc_mrsg/diagnostics.py`
- Modify corresponding tests.

- [ ] Add explicit `--initial-checkpoint` distinct from same-stage resume.
- [ ] Persist Phase 2 loss weights and initializer provenance.
- [ ] Gate nonfinite values, correction-bound violation, anchor-rank loss, and residual saturation.

### Task 5: Phase 2 Server Runner

**Files:**
- Create: `scripts/run_afloc_mrsg_phase2_server.sh`
- Create: `tests/test_afloc_mrsg_phase2_server_script.py`
- Modify: `docs/ANAPRIOR_RESEARCH_CONTEXT_2026-07-19.md`

- [ ] Run 2A, 2B, and 2C sequentially in separate output directories.
- [ ] Support explicit same-stage resume for every stage.
- [ ] Export validation diagnostics without validation gate or test-based selection.
- [ ] Run focused and complete test suites and push the branch.
