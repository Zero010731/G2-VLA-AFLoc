# AFLoc Anchor Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the frozen multi-scale AFLoc phrase-patch anchor through the exact MS-CXR localization pipeline, score it against AFLoc and DCEM-v3, and produce a fixed readiness decision before any Anchor-Preserving MRSG training.

**Architecture:** Keep AFLoc completely frozen and make `compute_afloc_phrase_anchor` the single canonical anchor implementation. A dedicated raw-data evaluator will emit `afloc_anchor/hmaps.npy` with the same case IDs, 224x224 normalization, and metadata contract as AFLoc-MRSG. A Phase 0 report will validate coverage and bootstrap metrics, then a standalone server script will run export, scoring, and reporting without loading an MRSG checkpoint.

**Tech Stack:** Python 3.9, PyTorch, NumPy, pandas, pycocotools-backed MS-CXR loader, pytest, Bash.

## Global Constraints

- Freeze all AFLoc parameters; Phase 0 has no optimizer and no trainable model.
- Do not consume MS-CXR boxes or masks while constructing heatmaps. The existing scorer may use them only after heatmaps are frozen.
- Do not consume DCEM, region maps, region scores, oracle maps, disease IDs, or validation gates while constructing the anchor.
- Use the same `path + label_text`, stable case-ID, resize, normalization, and margin contracts as the existing raw MRSG evaluator and scorer.
- Use `afloc_anchor` as the method name and a new output root; never overwrite a prior test-evaluated experiment.
- Record the AFLoc checkpoint hash, source git commit, explicit local BERT path, dataset paths, method count, case coverage, and output hashes.
- Do not implement bounded residual training, query changes, or EMA in this plan.

---

## File Structure

| File | Responsibility |
|---|---|
| `anaprior/models/afloc_mrsg/anchor.py` | canonical batch anchor and confidence computation |
| `anaprior/eval/diagnose_afloc_anchor_quality.py` | diagnostic GT overlap, delegated to canonical anchor |
| `anaprior/eval/eval_mscxr_afloc_anchor.py` | raw anchor-only heatmap export |
| `anaprior/eval/report_afloc_anchor_phase0.py` | coverage and metric readiness report |
| `scripts/run_afloc_anchor_phase0_server.sh` | one-command server export, scoring, and reporting |
| `tests/test_mrsg_anchor.py` | canonical anchor tests |
| `tests/test_diagnose_afloc_anchor_quality.py` | diagnostic/canonical parity test |
| `tests/test_eval_mscxr_afloc_anchor.py` | raw evaluator contract tests |
| `tests/test_report_afloc_anchor_phase0.py` | readiness decision tests |
| `tests/test_afloc_anchor_phase0_server_script.py` | server script contract tests |

---

### Task 1: Make the Training and Diagnostic Anchor Identical

**Files:**
- Modify: `anaprior/eval/diagnose_afloc_anchor_quality.py:24-69`
- Modify: `tests/test_diagnose_afloc_anchor_quality.py`
- Test: `tests/test_mrsg_anchor.py`

**Interfaces:**
- Consumes: `compute_afloc_phrase_anchor(image_features, phrase_features)` from `anaprior.models.afloc_mrsg.anchor`.
- Produces: diagnostic `phrase_patch_anchor(...) -> tuple[Tensor, dict[str, Tensor]]` that delegates to the canonical implementation and preserves the existing single-case API.

- [ ] **Step 1: Write the canonical-delegation test**

Monkeypatch the canonical function and prove the diagnostic wrapper invokes it:

```python
def test_diagnostic_anchor_delegates_to_canonical(monkeypatch) -> None:
    from anaprior.eval import diagnose_afloc_anchor_quality as module

    called = {"value": False}

    def fake_compute(image_batch, phrase_batch):
        called["value"] = True
        anchor = torch.ones(1, 1, 2, 2)
        confidence = torch.full_like(anchor, 0.75)
        scales = {name: anchor.clone() for name in ("l2", "l", "lf")}
        return anchor, confidence, scales

    monkeypatch.setattr(module, "compute_afloc_phrase_anchor", fake_compute)
    image = {
        "l2": torch.rand(1, 2, 2, 2),
        "l": torch.rand(1, 2, 1, 1),
        "lf": torch.rand(1, 2, 1, 1),
    }
    anchor, scales = module.phrase_patch_anchor(
        image,
        torch.rand(1, 2, 2),
        torch.tensor([[True, True]]),
    )
    assert called["value"] is True
    assert torch.equal(anchor, torch.ones(2, 2))
    assert set(scales) == {"l2", "l", "lf"}
```

- [ ] **Step 2: Run the parity test and verify it fails before delegation**

Run:

```bash
python -m pytest tests/test_diagnose_afloc_anchor_quality.py::test_diagnostic_anchor_matches_canonical_batch_anchor -q
```

Expected: FAIL because the diagnostic module does not yet expose or call
`compute_afloc_phrase_anchor`.

- [ ] **Step 3: Delegate the diagnostic wrapper to the canonical function**

Construct the two contract dataclasses from the existing tensors, call the canonical function, and return the single-case slices. Remove the duplicate cosine/resize implementation while retaining `normalize_map` for GT metric utilities.

```python
anchor, _, scales = compute_afloc_phrase_anchor(image_batch, phrase_batch)
return anchor[0, 0], {name: value[0, 0] for name, value in scales.items()}
```

- [ ] **Step 4: Run focused anchor tests**

```bash
python -m pytest tests/test_mrsg_anchor.py tests/test_diagnose_afloc_anchor_quality.py -q
```

Expected: all tests PASS, including CUDA mask compatibility when CUDA is available.

- [ ] **Step 5: Commit canonicalization**

```bash
git add anaprior/eval/diagnose_afloc_anchor_quality.py tests/test_diagnose_afloc_anchor_quality.py
git commit -m "refactor: use canonical AFLoc anchor in diagnostics"
```

---

### Task 2: Add the Raw Anchor-Only Evaluator

**Files:**
- Create: `anaprior/eval/eval_mscxr_afloc_anchor.py`
- Create: `tests/test_eval_mscxr_afloc_anchor.py`
- Reuse: `anaprior/eval/eval_mscxr_afloc_mrsg.py`

**Interfaces:**
- Consumes: `normalize_eval_rows`, `normalize_output_heatmap`, `load_dataset_rows`, and `stable_case_id` from the existing raw evaluator; `FrozenAFLocMRSGEncoder`; canonical anchor computation.
- Produces: `AnchorEvalResult`, `default_encode_anchor_case`, `build_mscxr_afloc_anchor_hmaps`, `save_mscxr_afloc_anchor_outputs`, and CLI `python -m anaprior.eval.eval_mscxr_afloc_anchor`.

- [ ] **Step 1: Write tests for input isolation, stable keys, and output shape**

The fake encoder callback receives only normalized non-spatial fields:

```python
def test_anchor_eval_uses_only_image_phrase_and_category(tmp_path: Path) -> None:
    seen = []
    result = build_mscxr_afloc_anchor_hmaps(
        data_rows=[{
            "path": str(tmp_path / "case.jpg"),
            "label_text": "right basilar opacity",
            "category": "Lung Opacity",
            "gtmasks": np.ones((4, 4)),
            "boxes": [[0, 0, 2, 2]],
        }],
        dataset="MS_CXR",
        afloc_checkpoint=tmp_path / "afloc.ckpt",
        encode_case=lambda row, runtime, device: (
            seen.append(dict(row)) or {
                "hmap": np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32),
                "confidence": np.ones((2, 2), dtype=np.float32),
            }
        ),
    )
    assert set(seen[0]) == {
        "case_id", "category", "dataset", "duplicate_index",
        "hmap_key", "label_text", "path",
    }
    assert next(iter(result.hmaps.values()))["hmap"].shape == (224, 224)
    assert result.summary["uses_spatial_annotations"] is False
```

Also test duplicate rows, constant maps, missing local BERT/AFLoc paths, and explicit MS-CXR path forwarding.

- [ ] **Step 2: Run the evaluator tests and verify module absence**

```bash
python -m pytest tests/test_eval_mscxr_afloc_anchor.py -q
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement runtime and one-case encoding**

The real callback must:

```python
image = encoder.afloc.process_img([row["path"]], device, flag=0)
image_features = encoder.encode_images(image)
phrase_features = encoder.encode_phrases(
    [row["label_text"]],
    [row["category"] or row["label_text"]],
    device=device,
)
anchor, confidence, scale_maps = compute_afloc_phrase_anchor(
    image_features,
    phrase_features,
)
return {
    "hmap": anchor[0, 0].detach().cpu().numpy(),
    "confidence": confidence[0, 0].detach().cpu().numpy(),
    "scale_agreement": float(confidence.mean().detach().cpu()),
}
```

Load `AFLOC_BERT_TYPE` from the required `--bert-type` argument and force local files when `--hf-local-files-only` is set. Do not accept any MRSG checkpoint.

- [ ] **Step 4: Implement result construction and persistence**

Save:

```text
<outdir>/afloc_anchor/hmaps.npy
<outdir>/anchor_case_diagnostics.json
<outdir>/anchor_eval_summary.json
```

Each heatmap payload must contain `case_id`, `path`, `label_text`, `category`, and a 224x224 finite normalized `hmap`. Summary fields must include `num_cases`, `num_hmaps`, duplicate count, zero-variance count, method name, AFLoc checkpoint path/SHA256, git commit, and all prohibited-input flags set to false.

- [ ] **Step 5: Run evaluator and compatibility tests**

```bash
python -m pytest tests/test_eval_mscxr_afloc_anchor.py tests/test_eval_mscxr_afloc_mrsg.py tests/test_python39_compat.py -q
```

Expected: all tests PASS under Python 3.9 syntax rules.

- [ ] **Step 6: Commit the evaluator**

```bash
git add anaprior/eval/eval_mscxr_afloc_anchor.py tests/test_eval_mscxr_afloc_anchor.py tests/test_python39_compat.py
git commit -m "feat: export frozen AFLoc anchor heatmaps"
```

---

### Task 3: Add the Phase 0 Readiness Report

**Files:**
- Create: `anaprior/eval/report_afloc_anchor_phase0.py`
- Create: `tests/test_report_afloc_anchor_phase0.py`

**Interfaces:**
- Consumes: `anchor_eval_summary.json` and `learned_repair_metrics_summary.json` produced by the existing scorer for methods `baseline,phrase_anatomy_dcem,afloc_anchor`.
- Produces: `anchor_phase0_decision.json`, `anchor_phase0_report.md`, and process exit code 0 for a complete report even when scientific readiness is false.

- [ ] **Step 1: Write report tests for ready, rejected, and incomplete evidence**

Use three fixtures. The ready fixture must satisfy:

```python
assert decision == {
    "status": "anchor_ready_for_bounded_refinement",
    "coverage_complete": True,
    "macro_cnr_delta_vs_baseline": 0.01,
    "macro_cnr_ci_low_vs_baseline": -0.004,
}
```

The fixed readiness rule is:

```text
coverage complete
AND zero_variance_count == 0
AND afloc_anchor_vs_baseline macro_all CNR CI low >= -0.005
AND afloc_anchor absolute pooled CNR > 0
```

If metrics or comparison rows are missing, emit `incomplete_evidence`. If the
rule fails, emit `anchor_not_ready_improve_construction`; never silently fall
back to DCEM.

- [ ] **Step 2: Run tests and verify module absence**

```bash
python -m pytest tests/test_report_afloc_anchor_phase0.py -q
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement structured parsing and fixed decisions**

Use pandas/JSON field access, never grep or ad hoc report-text parsing. Include:

- raw and common-method case counts;
- absolute pooled/macro CNR for all three methods;
- paired pooled/macro CNR, IoU, and Dice deltas;
- eight-class anchor-vs-baseline rows;
- anchor confidence and heatmap-distribution summaries;
- explicit reasons for readiness failure.

- [ ] **Step 4: Run report and scorer regression tests**

```bash
python -m pytest tests/test_report_afloc_anchor_phase0.py tests/test_score_mscxr_learned_repair_metrics.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit the report**

```bash
git add anaprior/eval/report_afloc_anchor_phase0.py tests/test_report_afloc_anchor_phase0.py
git commit -m "feat: report AFLoc anchor phase zero readiness"
```

---

### Task 4: Add a Reproducible One-Command Server Runner

**Files:**
- Create: `scripts/run_afloc_anchor_phase0_server.sh`
- Create: `tests/test_afloc_anchor_phase0_server_script.py`
- Modify: `docs/ANAPRIOR_RESEARCH_CONTEXT_2026-07-19.md`

**Interfaces:**
- Consumes required environment variables `AFLOC_CHECKPOINT`, `AFLOC_BERT_TYPE`, `LOCALIZATION_MS_CXR_JSON`, `LOCALIZATION_MIMIC_IMG_DIR`, and `REFERENCE_HMAPS_ROOT`.
- Produces a fresh `OUTROOT` with anchor heatmaps, copied frozen baselines, metric outputs, report, and immutable manifest.

- [ ] **Step 1: Write static script-contract tests**

Require:

```python
assert "set -euo pipefail" in script
assert "eval_mscxr_afloc_anchor" in script
assert "score_mscxr_learned_repair_metrics" in script
assert "report_afloc_anchor_phase0" in script
assert "baseline,phrase_anatomy_dcem,afloc_anchor" in script
assert "validation-gate" not in script
assert "train_afloc_mrsg" not in script
```

Also require preflight file checks, CRLF-safe Bash syntax, progress messages,
fresh-output protection, explicit local BERT, and actual git commit capture.

- [ ] **Step 2: Run script tests and verify failure**

```bash
python -m pytest tests/test_afloc_anchor_phase0_server_script.py -q
```

Expected: FAIL because the script is absent.

- [ ] **Step 3: Implement the three-stage runner**

The runner stages are fixed:

```text
[0/3] preflight and immutable manifest
[1/3] export afloc_anchor heatmaps
[2/3] score baseline, phrase_anatomy_dcem, afloc_anchor
[3/3] write readiness report and output hashes
```

Use defaults matching the verified server:

```bash
AFLOC_CHECKPOINT=/mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt
AFLOC_BERT_TYPE=/mnt/zhangran/Bio_ClinicalBERT
LOCALIZATION_MS_CXR_JSON=/mnt/zhangran/ms-cxr_1.1.0/MS_CXR_Local_Alignment_v1.1.0.json
LOCALIZATION_MIMIC_IMG_DIR=/mnt/mimic-cxr/jpg
METHOD_NAME=afloc_anchor
SCORE_METHODS=baseline,phrase_anatomy_dcem,afloc_anchor
```

Support `PREFLIGHT_ONLY=1`, `DRY_RUN=1`, `START_STAGE=0..3`, and explicit
`ALLOW_DIRTY_OUTROOT=1`. Do not require `MIMIC_CSV` because Phase 0 does not
train on MIMIC.

- [ ] **Step 4: Add the exact server command to the research ledger**

Document a fresh output name such as:

```bash
OUTROOT=/mnt3/zhangran/anaprior_outputs/anaprior_stage_k_afloc_anchor_phase0_run1 \
CUDA_VISIBLE_DEVICES=1 DEVICE=cuda \
bash scripts/run_afloc_anchor_phase0_server.sh
```

- [ ] **Step 5: Run script and documentation checks**

```bash
bash -n scripts/run_afloc_anchor_phase0_server.sh
python -m pytest tests/test_afloc_anchor_phase0_server_script.py tests/test_python39_compat.py -q
rg -n "anchor_phase0|afloc_anchor" docs/ANAPRIOR_RESEARCH_CONTEXT_2026-07-19.md
```

Expected: Bash syntax succeeds, tests PASS, and the documented command exists.

- [ ] **Step 6: Commit runner and protocol**

```bash
git add scripts/run_afloc_anchor_phase0_server.sh tests/test_afloc_anchor_phase0_server_script.py docs/ANAPRIOR_RESEARCH_CONTEXT_2026-07-19.md
git commit -m "feat: add AFLoc anchor phase zero runner"
```

---

### Task 5: Final Review and Execution Handoff

**Files:**
- Review all files from Tasks 1-4.

**Interfaces:**
- Consumes: complete Phase 0 implementation.
- Produces: a tested branch and exact server execution handoff; no model training.

- [ ] **Step 1: Verify prohibited dependencies are absent**

```bash
rg -n "region_scores|region_maps|oracle|base_hmap|dcem.*input|validation_gate|bbox|gtmask" \
  anaprior/eval/eval_mscxr_afloc_anchor.py \
  scripts/run_afloc_anchor_phase0_server.sh
```

Expected: no construction-time spatial supervision or repair input. References
to DCEM are allowed only in scorer method names and baseline-copy paths.

- [ ] **Step 2: Run the complete test suite**

```bash
python -m pytest -q
```

Expected: all tests PASS; CUDA-only tests may be skipped on CPU hosts.

- [ ] **Step 3: Run dry-run and preflight smoke tests**

```bash
DRY_RUN=1 bash scripts/run_afloc_anchor_phase0_server.sh
PREFLIGHT_ONLY=1 bash scripts/run_afloc_anchor_phase0_server.sh
```

Expected: dry-run prints all three commands; preflight validates paths and exits
without loading AFLoc or creating heatmaps.

- [ ] **Step 4: Inspect the final diff and commit any review corrections**

```bash
git diff --check
git status --short
git log --oneline -5
```

Expected: no whitespace errors, only intended files changed, and one focused
commit per completed task.

- [ ] **Step 5: Push and run only Phase 0 on the server**

Do not begin Anchor-Preserving MRSG implementation until
`anchor_phase0_decision.json` has been reviewed. If status is
`anchor_ready_for_bounded_refinement`, write a separate implementation plan for
the bounded forward. Otherwise improve anchor construction first.
