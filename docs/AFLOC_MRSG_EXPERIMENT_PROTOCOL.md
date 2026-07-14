# AFLoc-MRSG Server Experiment Protocol

This protocol describes the only supported end-to-end server execution path for standalone AFLoc-MRSG experiments. It is derived from `scripts/run_afloc_mrsg_full_server.sh` and the AFLoc-MRSG box-free design spec.

## Core Rules

1. Freeze the complete AFLoc backbone. Training must report `afloc_trainable_parameters == 0`.
2. Use only box-free image-report training data. Do not train, tune, gate, or select checkpoints with MS-CXR boxes, masks, oracle maps, region maps, region scores, or DCEM outputs.
3. Exclude all MS-CXR identities from the MIMIC cache before phrase mining and training.
4. Use exactly four top-level loss weights: `w_ground`, `w_teacher`, `w_mask`, `w_query`.
5. Use explicit phase gates only for Phases A, B, and C. Do not add a hidden validation gate to raw AFLoc-MRSG evaluation outputs.
6. Run Stage 7 once, only after Stage 6 has frozen the experiment manifest.

## Required Inputs

Server-side environment variables or files:

- `MIMIC_CSV`: leakage-screened MIMIC-CXR CSV source.
- `MSCXR_EXCLUSION_JSON`: MS-CXR exclusion manifest.
- `AFLOC_CHECKPOINT`: frozen AFLoc checkpoint.
- `MIMIC_IMAGE_ROOT`: MIMIC image directory.
- `DESCRIPTIONS_JSON`: disease description JSON.
- `REFERENCE_HMAPS_ROOT`: baseline and DCEM reference heatmaps for Stage 8 scoring.
- `LOCALIZATION_MS_CXR_JSON`: raw MS-CXR evaluation source.
- `LOCALIZATION_MIMIC_IMG_DIR`: raw MS-CXR image root.
- `CHEXLOCALIZE_TEST_JSON`: optional CheXlocalize test metadata, required only when `RUN_CHEXLOCALIZE=1`.
- `CHEXLOCALIZE_TEST_IMG_DIR`: optional CheXlocalize image root, required only when `RUN_CHEXLOCALIZE=1`.

Important runtime knobs:

- `OUTROOT`: experiment root.
- `RUN_NAME`: names the experiment subdirectory.
- `START_STAGE`: integer `0..10`.
- `PREFLIGHT_ONLY=1`: validate inputs and stop.
- `DRY_RUN=1`: print commands only.
- `RUN_CHEXLOCALIZE=0`: default; skip the optional external CheXlocalize evaluation. Set to `1` only when its data is available.
- `ALLOW_DIRTY_OUTROOT=1`: opt into rerunning with an existing non-empty `OUTROOT`.
- `PHASE_A_RESUME_CHECKPOINT`, `PHASE_B_RESUME_CHECKPOINT`, `PHASE_C_RESUME_CHECKPOINT`: explicit same-phase resume only.

## Prohibited Inputs

Do not pass or derive any of the following into raw AFLoc-MRSG evaluation:

- prepared-input NPZs
- base heatmaps
- region maps
- region scores
- validation gates or validation-gate source/fallback method names
- lambda overrides

The evaluator rejects:

- `--prepared-inputs-npz`
- `--base-hmaps-npy`
- `--region-maps-npy`
- `--region-score-csv`
- `--validation-gate`
- `--validation-gate-source-method`
- `--validation-gate-fallback-method`
- `--validation-gate-method-name`
- `--lambda-override`

## Stage Commands

`START_STAGE=0` executes the full pipeline. Later starts require the earlier frozen artifacts to already exist.

### Stage 0: Cache + Exclusion Audit

Command:

```bash
python -m anaprior.train.build_mrsg_image_report_cache \
  --mimic-csv "$MIMIC_CSV" \
  --mscxr-json "$MSCXR_EXCLUSION_JSON" \
  --descriptions-json "$DESCRIPTIONS_JSON" \
  --outdir "$CACHE_ROOT" \
  --valid-fraction "$VALID_FRACTION" \
  --seed "$SEED"
```

Required checks:

- `uses_spatial_annotations == false`
- `uses_dcem == false`
- protocol `sanity.mscxr_overlap == 0`
- protocol `sanity.train_valid_subject_overlap == 0`

Outputs:

- `cache/mrsg_image_report_cache_report.json`
- `cache/train_mrsg.jsonl`
- `cache/valid_mrsg.jsonl`
- `cache/mrsg_protocol_manifest.json`

### Stage 1: Phase A Locality Warm-Up

Command:

```bash
python -m anaprior.train.train_afloc_mrsg \
  --phase locality \
  --train-manifest "$TRAIN_MANIFEST" \
  --valid-manifest "$VALID_MANIFEST" \
  --outdir "$PHASE_A_ROOT" \
  --afloc-checkpoint "$AFLOC_CHECKPOINT" \
  --protocol-manifest "$PROTOCOL_MANIFEST" \
  --descriptions-json "$DESCRIPTIONS_JSON" \
  --image-root "$MIMIC_IMAGE_ROOT" \
  --feature-dim "$FEATURE_DIM" \
  --num-heads "$NUM_HEADS" \
  --focal-slots "$FOCAL_SLOTS" \
  --topk-fraction "$TOPK_FRACTION" \
  --route-temperature "$ROUTE_TEMPERATURE" \
  --epochs "$PHASE_A_EPOCHS" \
  --batch-size "$PHASE_A_BATCH_SIZE" \
  --learning-rate "$PHASE_A_LEARNING_RATE" \
  --w-ground "$W_GROUND" \
  --w-teacher "$W_TEACHER" \
  --w-mask "$W_MASK" \
  --w-query "$W_QUERY" \
  --seed "$SEED" \
  --device "$DEVICE"
```

Optional:

- append `--text-dim "$TEXT_DIM"` if set
- append `--resume-checkpoint "$PHASE_A_RESUME_CHECKPOINT"` only if explicitly provided

### Stage 2: Gate A Check

Gate requirements:

- `train_report.json.phase == "locality"`
- `phase_gate.passed == true`
- selected checkpoint must be `phase_a/mrsg_phase_a.pt`

### Stage 3: Phase B Sparse Grounding

Same command form as Phase A, with:

- `--phase grounding`
- `--outdir "$PHASE_B_ROOT"`
- `--previous-checkpoint "$PHASE_A_BEST"`
- Phase B epochs, batch size, and learning rate
- optional explicit `--resume-checkpoint "$PHASE_B_RESUME_CHECKPOINT"`

### Stage 4: Gate B Check

Gate requirements:

- `train_report.json.phase == "grounding"`
- `phase_gate.passed == true`
- selected checkpoint must be `phase_b/mrsg_phase_b.pt`

### Stage 5: Phase C Dual Consistency

Same command form as Phase B, with:

- `--phase consistency`
- `--outdir "$PHASE_C_ROOT"`
- `--previous-checkpoint "$PHASE_B_BEST"`
- `--teacher-decay "$PHASE_C_TEACHER_DECAY"`
- Phase C epochs, batch size, and learning rate
- optional explicit `--resume-checkpoint "$PHASE_C_RESUME_CHECKPOINT"`

### Stage 6: Gate C + Freeze Manifest

Gate requirements:

- `train_report.json.phase == "consistency"`
- `phase_gate.passed == true`
- selected checkpoint must be `phase_c/mrsg_phase_c.pt`

Freeze step:

- write `frozen_experiment_manifest.json`
- capture immutable training hashes, phase artifacts, AFLoc checkpoint hash, model config, and four top-level loss weights

### Stage 7: Raw MS-CXR Heatmaps

Command:

```bash
python -m anaprior.eval.eval_mscxr_afloc_mrsg \
  --dataset MS_CXR \
  --split test \
  --afloc-checkpoint "$AFLOC_CHECKPOINT" \
  --checkpoint "$PHASE_C_BEST" \
  --outdir "$MSCXR_EVAL_ROOT" \
  --device "$DEVICE" \
  --method-name "$METHOD_NAME"
```

Optional:

- append `--max-cases "$MAX_CASES"` if set

One-shot rule:

- Stage 7 is guarded by `frozen_experiment_manifest.json`
- if `test_evaluated == true`, rerunning Stage 7 must fail

### Stage 8: Raw Scoring vs AFLoc and DCEM

Commands:

1. copy Stage 7 heatmaps into `score_hmaps/$METHOD_NAME/hmaps.npy`
2. copy baseline/DCEM reference heatmaps from `REFERENCE_HMAPS_ROOT`
3. run:

```bash
python -m anaprior.eval.score_mscxr_learned_repair_metrics \
  --hmaps-root "$SCORE_HMAP_ROOT" \
  --outdir "$METRIC_ROOT" \
  --methods "$SCORE_METHODS" \
  --candidate-categories "$FINDINGS" \
  --dataset "$SCORE_DATASET" \
  --val-fraction "$SCORE_VAL_FRACTION" \
  --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
  --seed "$SEED" \
  --candidate-effect-floor "$CANDIDATE_EFFECT_FLOOR" \
  --macro-all-harm-floor "$MACRO_ALL_HARM_FLOOR" \
  --margin
```

### Stage 9: Frozen CheXlocalize External Evaluation

This stage is optional and disabled by default. It runs only when `RUN_CHEXLOCALIZE=1`; otherwise Stage 9 and its bundle artifact are skipped.

Command:

```bash
python -m anaprior.eval.eval_mscxr_afloc_mrsg \
  --dataset CHEXLOCALIZE \
  --split test \
  --afloc-checkpoint "$AFLOC_CHECKPOINT" \
  --checkpoint "$PHASE_C_BEST" \
  --outdir "$CHEXLOCALIZE_EVAL_ROOT" \
  --device "$DEVICE" \
  --method-name "$METHOD_NAME"
```

### Stage 10: Report Bundle

Command:

```bash
python -m anaprior.eval.report_stage_c_results \
  --metrics-dir "$METRIC_ROOT" \
  --output-md "$REPORT_MD"
```

This also writes a bundle manifest over the frozen manifest, cache report, phase reports, evaluation summaries, and final markdown report.

## Phase Gates

Only three explicit gates are allowed:

- Gate A after locality training
- Gate B after grounding training
- Gate C after consistency training

Each gate reads the corresponding `train_report.json` and checks:

- the expected phase name
- `phase_gate.passed == true`
- the expected selected best checkpoint path exists

No additional raw-evaluation validation gate is permitted after Phase C.

## Resume Semantics

Resume is explicit-only.

- Same-phase reruns do not auto-discover latest checkpoints.
- To resume a phase, pass exactly one of:
  - `PHASE_A_RESUME_CHECKPOINT`
  - `PHASE_B_RESUME_CHECKPOINT`
  - `PHASE_C_RESUME_CHECKPOINT`
- The resume checkpoint must already exist.
- Cross-phase resume is not allowed; Phase B always starts from passed Phase A best, and Phase C always starts from passed Phase B best.

## Sync Instructions

Before Stage 7:

1. Sync `cache/`, `phase_a/`, `phase_b/`, and `phase_c/` outputs to stable storage.
2. Verify `frozen_experiment_manifest.json` matches the intended training state.
3. Do not modify:
   - AFLoc checkpoint
   - descriptions JSON
   - model config
   - four top-level loss weights
   - phase reports or selected best checkpoints

After Stages 7-10:

1. Sync evaluation outputs and bundle artifacts back alongside `OUTROOT`.
2. Preserve `frozen_experiment_manifest.json` with updated `output_hashes`.
3. Treat `phase_c/mrsg_phase_c.pt` as the only checkpoint used for raw held-out evaluation.

## Expected Output Paths

Under `OUTROOT`:

- `cache/mrsg_image_report_cache_report.json`
- `cache/train_mrsg.jsonl`
- `cache/valid_mrsg.jsonl`
- `cache/mrsg_protocol_manifest.json`
- `phase_a/train_report.json`
- `phase_a/mrsg_phase_a.pt`
- `phase_a/mrsg_phase_a_latest.pt`
- `phase_b/train_report.json`
- `phase_b/mrsg_phase_b.pt`
- `phase_b/mrsg_phase_b_latest.pt`
- `phase_c/train_report.json`
- `phase_c/mrsg_phase_c.pt`
- `phase_c/mrsg_phase_c_latest.pt`
- `frozen_experiment_manifest.json`
- `mscxr_eval/afloc_mrsg/hmaps.npy`
- `mscxr_eval/mrsg_eval_summary.json`
- `mscxr_eval/mrsg_case_diagnostics.json`
- `score_hmaps/afloc_mrsg/hmaps.npy`
- `learned_repair_metrics/learned_repair_metrics_summary.json`
- `learned_repair_metrics/learned_repair_decision.json`
- `chexlocalize_eval/afloc_mrsg/hmaps.npy`
- `chexlocalize_eval/mrsg_eval_summary.json`
- `chexlocalize_eval/mrsg_case_diagnostics.json`
- `bundle/afloc_mrsg_result_report.md`
- `bundle/bundle_manifest.json`

## Evaluation Contract Summary

- Training uses AFLoc-compatible deterministic preprocessing plus geometry-tracked augmentations.
- Raw evaluation uses the loaded AFLoc checkpoint preprocessing contract and does not consume repair-pipeline inputs.
- The server protocol selects checkpoints only through Phase A/B/C gates and then performs one frozen held-out Stage 7 MS-CXR evaluation.
