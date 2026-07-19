# AFLoc Official Anchor Phase 0b Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the repository's official AFLoc baseline heatmap exactly and prove pixel-level parity before using it as MRSG's forward anchor.

**Architecture:** Keep the rejected multi-scale token-max anchor unchanged as an ablation. Add a separate official heatmap path implementing final local image embedding times global report embedding, Gaussian smoothing with sigma 1.5, and bilinear resize. Export raw heatmaps and compare them directly with the saved AFLoc baseline using legacy image-plus-phrase keys.

**Tech Stack:** Python 3.9, PyTorch, NumPy, SciPy, pytest, Bash.

## Global Constraints

- AFLoc remains frozen.
- No boxes, masks, region annotations, DCEM maps, region predictor, or disease-specific rules enter inference.
- MS-CXR annotations are used only to enumerate diagnostic cases, never as model inputs.
- Method name is `afloc_official_anchor`; `afloc_anchor` remains the rejected handcrafted ablation.
- No MRSG training starts until parity passes.

---

### Task 1: Exact Official AFLoc Heatmap

**Files:**
- Create: `anaprior/features/afloc_official_heatmap.py`
- Create: `tests/test_afloc_official_heatmap.py`

**Interfaces:**
- Produces: `compute_official_afloc_heatmap(local_embeddings, report_embeddings, output_size=(224, 224), sigma=1.5) -> torch.Tensor`.

- [ ] Write a failing test comparing the new function against `ImageTextInferenceEngine._get_similarity_map_from_embeddings` and `convert_similarity_to_image_size` on deterministic embeddings.
- [ ] Run `python -m pytest tests/test_afloc_official_heatmap.py -q` and verify failure because the module is absent.
- [ ] Implement shape validation, exact dot product, SciPy Gaussian smoothing, and bilinear resizing without min-max normalization.
- [ ] Re-run the focused test and verify it passes.

### Task 2: Official Export and Pixel Parity

**Files:**
- Create: `anaprior/eval/eval_mscxr_afloc_official_anchor.py`
- Create: `anaprior/eval/report_afloc_official_anchor_parity.py`
- Create: `tests/test_eval_mscxr_afloc_official_anchor.py`
- Create: `tests/test_report_afloc_official_anchor_parity.py`
- Modify: `tests/test_python39_compat.py`

**Interfaces:**
- Produces: CLI `python -m anaprior.eval.eval_mscxr_afloc_official_anchor` and `official_anchor_eval/afloc_official_anchor/hmaps.npy`.
- Produces: CLI `python -m anaprior.eval.report_afloc_official_anchor_parity` and `official_anchor_parity.json`.

- [ ] Write failing exporter tests proving that only image path and report phrase reach inference and that raw values are preserved.
- [ ] Implement frozen AFLoc loading with local BioClinicalBERT, checkpoint-configured preprocessing, exact final-local/global-report heatmap generation, canonical IDs, and legacy lookup keys.
- [ ] Write failing parity tests for exact match, shape mismatch, missing keys, and numerical mismatch.
- [ ] Implement direct key matching and aggregate `coverage`, `mean_pearson_r`, `mean_mae`, and `max_abs_error`.
- [ ] Require `coverage >= 0.99`, `mean_pearson_r >= 0.999`, and `mean_mae <= 1e-5` for `parity_passed=true`.
- [ ] Run all four focused test files and verify they pass.

### Task 3: One-Command Server Run and Research Record

**Files:**
- Create: `scripts/run_afloc_official_anchor_phase0b_server.sh`
- Create: `tests/test_afloc_official_anchor_phase0b_server_script.py`
- Modify: `docs/ANAPRIOR_RESEARCH_CONTEXT_2026-07-19.md`

**Interfaces:**
- Produces a resumable two-stage server run: official heatmap export, then parity report.

- [ ] Write a failing script contract test covering explicit paths, offline BERT settings, method name, and stage resume.
- [ ] Implement preflight plus `START_STAGE=0..2` execution without requiring MIMIC training CSV or MRSG checkpoints.
- [ ] Record the formal Phase 0 failure numbers and state that official baseline parity replaces handcrafted-anchor refinement as the next decision gate.
- [ ] Run `bash -n scripts/run_afloc_official_anchor_phase0b_server.sh` where Bash is available and run the script contract test.
- [ ] Run the complete Python test suite and inspect `git diff --check`.
