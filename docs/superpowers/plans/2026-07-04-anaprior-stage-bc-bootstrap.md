# AnaPrior-Loc Stage B/C Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a clean AFLoc-derived workspace for the learned AnaPrior-Loc pipeline and implement the first patient-level leakage guardrail.

**Architecture:** Keep AFLoc as the backbone. Add a separate `anaprior` package for data preparation, feature pooling, learned region abnormality prediction, and learned repair evaluation.

**Tech Stack:** Python standard library for split generation; PyTorch will be used later for feature pooling and the predictor.

## Global Constraints

- Do not continue adding learned-method code to `C:\Users\joker\Desktop\G2-VLA-AFLoc`.
- Treat `G2-VLA-AFLoc` as Stage A oracle evidence.
- Exclude every MS-CXR patient from Chest ImaGenome train/held-out rows before training any learned predictor.
- Do not tune learned repair hyperparameters on MS-CXR.

---

### Task 1: Clean Workspace Bootstrap

**Files:**
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\`
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\README_NEXT_STAGE.md`
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\configs\anaprior_learned.yaml`

**Interfaces:**
- Consumes: clean AFLoc clone.
- Produces: isolated package layout for Stage B/C work.

- [x] Clone `C:\Users\joker\Desktop\AFLoc` to `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc`.
- [x] Create `anaprior` package folders.
- [x] Add Stage B/C README and config.

### Task 2: Patient-Level Leakage Split

**Files:**
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\data\build_patient_splits.py`

**Interfaces:**
- Consumes: MS-CXR COCO JSON and Chest ImaGenome silver split CSVs.
- Produces: `mscxr_patients.txt`, `imagenome_train_clean.csv`, `imagenome_valid_clean.csv`, `imagenome_test_clean.csv`, and `leakage_report.json`.

- [x] Extract MS-CXR patient IDs from image paths.
- [x] Filter every Chest ImaGenome split by excluding those patients.
- [x] Write a JSON leakage report with train/valid/test overlap counts.
- [x] Run the script on local data and inspect `leakage_report.json`.

Smoke result:

```text
MS-CXR patients = 851
raw train overlap = 579 patients
raw valid overlap = 86 patients
raw test overlap = 165 patients
clean train/valid/test overlap = 0
```

### Task 3: Region-Finding Training Table

**Files:**
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\data\build_region_finding_table.py`

**Interfaces:**
- Consumes: clean Chest ImaGenome split CSVs and scene graph JSONs.
- Produces: a candidate-class region-finding table for Pneumothorax and Pleural Effusion.

- [x] Parse Chest ImaGenome scene graphs for anatomical regions and attributes.
- [x] Emit one row per image/region/finding label.
- [x] Report positive prevalence per finding and region.

Implementation note:

```text
The first training target uses --label-policy explicit:
keep explicit yes/no anatomicalfinding labels;
drop unmentioned negatives.
```

Full local run:

```text
Train rows = 728331
Train Pneumothorax positives = 15749 / 235472
Train Pleural Effusion positives = 174911 / 492859

Valid rows = 104867
Valid Pneumothorax positives = 2094 / 33821
Valid Pleural Effusion positives = 25257 / 71046
```

### Task 4: Region Pooling and Predictor Core

**Files:**
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\features\region_pooling.py`
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\models\region_abnormality_predictor.py`
- Test: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\tests\test_region_pooling.py`
- Test: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\tests\test_region_abnormality_predictor.py`

**Interfaces:**
- Consumes: AFLoc local feature maps with shape `[B,C,H,W]` or `[C,H,W]`.
- Produces: pooled region features and binary region abnormality logits.

- [x] Implement `RegionBox` and `pool_region_features`.
- [x] Support AFLoc-like local feature maps such as `[B,768,19,19]`.
- [x] Implement `RegionAbnormalityPredictor`.
- [x] Add masked BCE loss for invalid regions.
- [x] Verify with unit tests.

Next task:

```text
Build a feature extraction/cache script that runs AFLoc over clean Chest ImaGenome rows,
pools region features, and writes train/valid tensors for predictor training.
```

### Task 5: AFLoc Region Feature Cache

**Files:**
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\features\extract_region_features.py`
- Test: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\tests\test_extract_region_features.py`

**Interfaces:**
- Consumes: `region_finding_train.csv` / `region_finding_valid.csv`.
- Consumes: AFLoc checkpoint and MIMIC-CXR JPG root at CLI runtime.
- Produces: `.pt` cache with region features, labels, finding ids, valid mask, cell counts, finding vocab, and metadata.

- [x] Implement testable cache builder with injectable feature extractor.
- [x] Deduplicate work so each DICOM is encoded once even if it has many region-finding rows.
- [x] Add real AFLoc CLI extractor behind `--ckpt`.
- [x] Verify cache format with tests.

Server smoke command:

```bash
python -m anaprior.features.extract_region_features \
  --region-table-csv outputs/anaprior_stage_b_region_table_explicit/region_finding_train.csv \
  --output-path outputs/anaprior_stage_b_feature_cache/train_smoke_img_emb_l.pt \
  --ckpt /mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt \
  --image-root /mnt/mimic-cxr/jpg \
  --feature-level img_emb_l \
  --device cuda \
  --max-rows 1000
```

### Task 6: Region Abnormality Predictor Training and Held-Out Gate

**Files:**
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\train\train_region_predictor.py`
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\eval\eval_predictor_heldout.py`
- Test: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\tests\test_train_region_predictor.py`
- Test: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\tests\test_eval_predictor_heldout.py`

**Interfaces:**
- Consumes: train/valid `.pt` feature caches from Task 5.
- Produces: `region_predictor.pt`, `train_report.json`, and held-out metrics JSON/CSV.

- [x] Implement pure binary AUROC, average precision, and best-F1 helpers.
- [x] Evaluate metrics overall and per finding.
- [x] Train the predictor from cached region features and labels.
- [x] Save checkpoint with model config and finding vocabulary.
- [x] Write validation metrics after training.
- [x] Verify with unit tests.

Server training command:

```bash
python -m anaprior.train.train_region_predictor \
  --train-cache outputs/anaprior_stage_b_feature_cache/train_img_emb_l.pt \
  --valid-cache outputs/anaprior_stage_b_feature_cache/valid_img_emb_l.pt \
  --outdir outputs/anaprior_stage_b_predictor_img_emb_l \
  --epochs 20 \
  --batch-size 4096 \
  --learning-rate 0.001 \
  --device cuda
```

Gate meaning:

```text
Before MS-CXR learned repair, prove the predictor can solve its own
Chest ImaGenome held-out region abnormality task without patient leakage.
```

### Task 7: Stage C Learned Repair Core

**Files:**
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\eval\prepare_mscxr_repair_inputs.py`
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\eval\selective_repair.py`
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\eval\predict_region_scores.py`
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\eval\eval_mscxr_learned_repair.py`
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\eval\score_mscxr_learned_repair_metrics.py`
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\eval\stage_c_preflight.py`
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\eval\report_stage_c_results.py`
- Test: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\tests\test_prepare_mscxr_repair_inputs.py`
- Test: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\tests\test_selective_repair_core.py`
- Test: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\tests\test_predict_region_scores.py`
- Test: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\tests\test_eval_mscxr_learned_repair.py`
- Test: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\tests\test_score_mscxr_learned_repair_metrics.py`
- Test: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\tests\test_stage_c_preflight.py`
- Test: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\tests\test_report_stage_c_results.py`
- Script: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\scripts\run_stage_c_learned_repair_server.sh`
- Test: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\tests\test_stage_c_server_script.py`

**Interfaces:**
- Consumes: trained Region Abnormality Predictor checkpoint and `.pt` feature cache.
- Consumes: baseline AFLoc MS-CXR heatmaps and Chest ImaGenome scene graphs.
- Produces: prepared MS-CXR repair inputs, score-request CSV, learned region score CSV, and method heatmaps.

- [x] Implement `repair_map_from_region_scores`.
- [x] Implement `blend_heatmap`.
- [x] Implement candidate-only, all-class, shuffled, and uniform score modes.
- [x] Prepare MS-CXR repair input bundle from baseline heatmaps and scene graphs.
- [x] Generate MS-CXR score-request CSV for feature extraction.
- [x] Export predictor logits/probabilities per DICOM/region/finding.
- [x] Build baseline, learned selective, all-class learned, candidate shuffled, and candidate uniform heatmaps in one pass.
- [x] Score all method heatmaps with MS-CXR IoU/CNR/Dice and paired bootstrap summaries.
- [x] Add preflight checks for required checkpoints, data roots, baseline heatmaps, and output writability.
- [x] Generate a Chinese Markdown result report from Stage C decision and bootstrap tables.
- [x] Add a server runner script that executes the whole Stage C chain with smoke/full modes.
- [x] Verify with unit tests.

Recommended server entry:

```bash
ANAPRIOR_RUN_SMOKE=1 bash scripts/run_stage_c_learned_repair_server.sh
ANAPRIOR_RUN_SMOKE=0 bash scripts/run_stage_c_learned_repair_server.sh
```

Prepare MS-CXR repair inputs and score requests:

```bash
python -m anaprior.eval.prepare_mscxr_repair_inputs \
  --base-hmaps-npy /mnt/zhangran/afloc_outputs/anaprior_fixed_gate1_reliable_adaptive/baseline/hmaps.npy \
  --prior-table /home/zhangran/zr/G2-VLA-AFLoc/anaprior_assets/prior_table.json \
  --chest-imagenome-root /mnt/chest-imagenome_1.0.0 \
  --output-npz outputs/anaprior_stage_c_mscxr/mscxr_repair_inputs.npz \
  --score-request-csv outputs/anaprior_stage_c_mscxr/mscxr_score_request.csv \
  --findings "Pneumothorax,Pleural Effusion"
```

Score export command:

```bash
python -m anaprior.eval.predict_region_scores \
  --checkpoint outputs/anaprior_stage_b_predictor_img_emb_l/region_predictor.pt \
  --cache outputs/anaprior_stage_c_mscxr_feature_cache/mscxr_img_emb_l.pt \
  --output-csv outputs/anaprior_stage_c_learned_scores/mscxr_region_scores.csv \
  --device cuda
```

Build pre-registered method heatmaps:

```bash
python -m anaprior.eval.eval_mscxr_learned_repair \
  --prepared-inputs-npz outputs/anaprior_stage_c_mscxr/mscxr_repair_inputs.npz \
  --region-score-csv outputs/anaprior_stage_c_learned_scores/mscxr_region_scores.csv \
  --outdir outputs/anaprior_stage_c_learned_repair_hmaps \
  --candidate-categories "Pneumothorax,Pleural Effusion" \
  --alpha 0.3 \
  --seed 0
```

Score the generated method heatmaps:

```bash
python -m anaprior.eval.score_mscxr_learned_repair_metrics \
  --hmaps-root outputs/anaprior_stage_c_learned_repair_hmaps \
  --outdir outputs/anaprior_stage_c_learned_repair_metrics \
  --methods baseline,learned_selective,all_class_learned,candidate_shuffled,candidate_uniform \
  --candidate-categories "Pneumothorax,Pleural Effusion" \
  --bootstrap-replicates 1000 \
  --seed 0 \
  --margin
```

Decision rule:

```text
Keep learned selective repair only if:
1. learned_selective beats baseline on test macro_candidate CNR;
2. learned_selective beats candidate_shuffled on test macro_candidate CNR;
3. macro_all CNR is not harmed beyond the pre-registered floor.
```

### Task 8: Server Transfer Source Bundle

**Files:**
- Create: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\anaprior\tools\create_stage_c_source_bundle.py`
- Test: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\tests\test_create_stage_c_source_bundle.py`
- Update: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\README_NEXT_STAGE.md`
- Update: `C:\Users\joker\Desktop\AnaPrior-Loc-AFLoc\docs\ANAPRIOR_LOC_中文落地说明.md`

**Interfaces:**
- Consumes: clean `AnaPrior-Loc-AFLoc` source tree.
- Produces: `outputs/anaprior_stage_c_source_bundle.zip`.

- [x] Bundle only next-stage source files, tests, docs, and the Stage C runner.
- [x] Exclude `outputs/`, `.git/`, `.pytest_cache/`, `__pycache__/`, and bytecode.
- [x] Embed `ANAPRIOR_STAGE_C_BUNDLE_MANIFEST.json` inside the zip for auditability.
- [x] Document `scp` / `unzip` transfer flow for the server.
- [x] Verify with unit tests and a real bundle generation run.

Local bundle command:

```powershell
python -m anaprior.tools.create_stage_c_source_bundle `
  --output outputs\anaprior_stage_c_source_bundle.zip
```

Server transfer sketch:

```powershell
scp outputs\anaprior_stage_c_source_bundle.zip zhangran@SERVER:/home/zhangran/zr/
```

```bash
cd /home/zhangran/zr/G2-VLA-AFLoc
unzip -o /home/zhangran/zr/anaprior_stage_c_source_bundle.zip
ANAPRIOR_RUN_SMOKE=1 bash scripts/run_stage_c_learned_repair_server.sh
```
