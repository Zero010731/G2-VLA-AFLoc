# AFLoc Anchor-Preserving MRSG Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the rejected absolute heatmap decoder with an unavoidable official-AFLoc anchor plus bounded learned logit residual, then run a no-EMA Phase 1 smoke experiment.

**Architecture:** Build each phrase's official AFLoc anchor from frozen `iel` and `teg`, normalize it only for stable logit composition, and pass it explicitly into every positive and negative MRSG forward. The decoder predicts only a bounded residual; zero initialization recovers the anchor and no query path can replace it. Phase 1 trains grounding without cross-view or EMA and records correction diagnostics before raw MS-CXR comparison.

**Tech Stack:** Python 3.9, PyTorch, NumPy, SciPy, pytest, Bash.

## Global Constraints

- Freeze AFLoc completely.
- No boxes, masks, anatomy annotations, region predictor, DCEM, disease IDs, or validation gate.
- `afloc_official_anchor` is the only forward base.
- Phase 1 excludes EMA and cross-view consistency.
- Old absolute-decoder checkpoints are incompatible and must not be loaded.

---

### Task 1: Batched Official Anchor

**Files:**
- Modify: `anaprior/features/afloc_official_heatmap.py`
- Modify: `tests/test_afloc_official_heatmap.py`

- [ ] Test batched `iel/teg` construction, output shape, detachment, finite normalization, and rank preservation.
- [ ] Implement `compute_official_afloc_anchor_batch` using the verified official formula and target-grid resize.
- [ ] Verify focused tests.

### Task 2: Bounded Residual Forward

**Files:**
- Modify: `anaprior/models/afloc_mrsg/contracts.py`
- Modify: `anaprior/models/afloc_mrsg/dense_decoder.py`
- Modify: `anaprior/models/afloc_mrsg/model.py`
- Modify: `tests/test_mrsg_grounding_decoder.py`
- Modify: `tests/test_afloc_mrsg_model.py`

- [ ] Test that official anchor is mandatory and zero residual recovers it.
- [ ] Test the logit correction cannot exceed the configured bound and gradients reach all shared queries.
- [ ] Replace `StandaloneDenseDecoder` with `AnchorBoundedResidualDecoder` and zero-initialize its residual head.
- [ ] Expose detached anchor plus differentiable residual/correction tensors in `MRSGOutput`.

### Task 3: Loss and Diagnostics

**Files:**
- Modify: `anaprior/models/afloc_mrsg/losses.py`
- Modify: `anaprior/models/afloc_mrsg/diagnostics.py`
- Modify: `tests/test_mrsg_losses.py`
- Modify: `tests/test_mrsg_diagnostics.py`

- [ ] Test anchor preservation and residual magnitude diagnostics.
- [ ] Remove the old final-to-routed-query alignment that can overwrite the anchor.
- [ ] Add residual mean, absolute mean, maximum, correction amplitude, final-anchor MAE, and final-anchor correlation.

### Task 4: Training and Evaluation Plumbing

**Files:**
- Modify: `anaprior/train/train_afloc_mrsg.py`
- Modify: `anaprior/eval/eval_mscxr_afloc_mrsg.py`
- Modify: `tests/test_train_afloc_mrsg.py`
- Modify: `tests/test_eval_mscxr_afloc_mrsg.py`

- [ ] Test that every positive, negative, and evaluation forward receives its own official phrase anchor.
- [ ] Compute official anchors before model calls and remove the rejected handcrafted anchor from training.
- [ ] Restrict cross-view and EMA to later phases; Phase 1 grounding uses neither.
- [ ] Persist architecture identity and residual bound in checkpoints/reports.

### Task 5: Phase 1 Server Script

**Files:**
- Create: `scripts/run_afloc_anchor_preserving_phase1_server.sh`
- Create: `tests/test_afloc_anchor_preserving_phase1_server_script.py`
- Modify: `docs/ANAPRIOR_RESEARCH_CONTEXT_2026-07-19.md`

- [ ] Test the script uses a fresh output root, verified Phase 0b parity, grounding-only training, no teacher, and raw scoring.
- [ ] Implement preflight, train, diagnostics, heatmap export, and AFLoc comparison stages with resume support.
- [ ] Run focused tests, full tests, and `git diff --check`.
