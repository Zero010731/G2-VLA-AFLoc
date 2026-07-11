# AnaPrior-Loc Stage B/C

中文落地说明见：

```text
docs/ANAPRIOR_LOC_中文落地说明.md
```

交付审计见：

```text
docs/ANAPRIOR_STAGE_C_DELIVERY_AUDIT.md
```

当前上下文见：

```text
docs/ANAPRIOR_CURRENT_CONTEXT.md
```

This repository is a clean AFLoc-derived workspace for the learned AnaPrior-Loc line.
The previous `G2-VLA-AFLoc` directory should be treated as the Stage A oracle and
ablation evidence workspace. This repository is for the learned method only.

## Source Bundle For Server Transfer

Use this command on Windows before moving the next-stage code to the Linux server:

```powershell
python -m anaprior.tools.create_stage_c_source_bundle `
  --output outputs\anaprior_stage_c_source_bundle.zip
```

The zip intentionally includes only the Stage B/C source files, tests, docs, and
runner script. It excludes `outputs/`, `.git/`, `.pytest_cache/`, `__pycache__/`,
and Python bytecode.

Example transfer:

```powershell
scp outputs\anaprior_stage_c_source_bundle.zip zhangran@SERVER:/home/zhangran/zr/
```

On the server:

```bash
cd /home/zhangran/zr/G2-VLA-AFLoc
unzip -o /home/zhangran/zr/anaprior_stage_c_source_bundle.zip
ANAPRIOR_RUN_SMOKE=1 bash scripts/run_stage_c_learned_repair_server.sh
```

## Main Line

The new method is:

```text
Selective Pathology-Conditioned Region Repair
```

The method keeps AFLoc as the localization backbone and adds a learned repair head:

```text
AFLoc local features
  -> region pooling
  -> Region Abnormality Predictor
  -> Class Reliability Gate
  -> Selective Repair Fusion
  -> final heatmap
```

## Current Boundary

Stage A has shown that oracle Chest ImaGenome region evidence can strongly repair
AFLoc for:

- Pneumothorax
- Pleural Effusion

The oracle result is idea validation and an upper-bound signal. It is not the final
publishable learned method because it reads region abnormality evidence directly.

## Stage B: Data Split and Predictor

The first hard requirement is patient-level leakage control:

```text
Chest ImaGenome train/valid rows must exclude every MS-CXR patient.
```

First command on Windows:

```powershell
python -m anaprior.data.build_patient_splits `
  --mscxr-json C:\Users\joker\Desktop\G2-VLA-AFLoc\data\ms-cxr\1.1.0\MS_CXR_Local_Alignment_v1.1.0_radgraph_phrase.json `
  --chest-imagenome-root C:\baidunetdiskdownload\chest-imagenome_1.0.0 `
  --outdir outputs\anaprior_stage_b_splits
```

Expected outputs:

- `outputs/anaprior_stage_b_splits/mscxr_patients.txt`
- `outputs/anaprior_stage_b_splits/imagenome_train_clean.csv`
- `outputs/anaprior_stage_b_splits/imagenome_valid_clean.csv`
- `outputs/anaprior_stage_b_splits/imagenome_test_clean.csv`
- `outputs/anaprior_stage_b_splits/leakage_report.json`

Stage B cannot continue unless the report shows:

```text
train_mscxr_overlap = 0
valid_mscxr_overlap = 0
```

Local smoke run on 2026-07-04:

```text
MS-CXR patients: 851

Raw Chest ImaGenome overlap:
train patients overlapping MS-CXR: 579
valid patients overlapping MS-CXR: 86
test patients overlapping MS-CXR: 165

Clean split sanity:
train_mscxr_overlap = 0
valid_mscxr_overlap = 0
test_mscxr_overlap = 0
train_valid_overlap = 0
train_test_overlap = 0
valid_test_overlap = 0
```

This confirms the leakage risk is real and the clean split step is required
before any Region Abnormality Predictor training.

## Stage B Region-Finding Table

The first learned predictor target uses conservative explicit labels only:

```text
label_policy = explicit
keep: anatomicalfinding|yes|finding
keep: anatomicalfinding|no|finding
drop: unmentioned negatives
```

Command:

```powershell
python -m anaprior.data.build_region_finding_table `
  --split-csv outputs\anaprior_stage_b_splits\imagenome_train_clean.csv `
  --chest-imagenome-root C:\baidunetdiskdownload\chest-imagenome_1.0.0 `
  --outdir outputs\anaprior_stage_b_region_table_explicit `
  --split-name train `
  --label-policy explicit

python -m anaprior.data.build_region_finding_table `
  --split-csv outputs\anaprior_stage_b_splits\imagenome_valid_clean.csv `
  --chest-imagenome-root C:\baidunetdiskdownload\chest-imagenome_1.0.0 `
  --outdir outputs\anaprior_stage_b_region_table_explicit `
  --split-name valid `
  --label-policy explicit
```

Local full run on 2026-07-04:

```text
Train:
input images: 156936
scene graphs found/missing: 156919 / 17
rows: 728331
Pneumothorax: 15749 positive / 235472 rows, prevalence 0.0669
Pleural Effusion: 174911 positive / 492859 rows, prevalence 0.3549

Valid:
input images: 22674
scene graphs found/missing: 22672 / 2
rows: 104867
Pneumothorax: 2094 positive / 33821 rows, prevalence 0.0619
Pleural Effusion: 25257 positive / 71046 rows, prevalence 0.3555
```

Outputs:

- `outputs/anaprior_stage_b_region_table_explicit/region_finding_train.csv`
- `outputs/anaprior_stage_b_region_table_explicit/region_finding_valid.csv`
- `outputs/anaprior_stage_b_region_table_explicit/region_finding_report_train.json`
- `outputs/anaprior_stage_b_region_table_explicit/region_finding_report_valid.json`

## Stage B Learned Modules

The first code-level learned components are now separated from AFLoc:

```text
anaprior.features.region_pooling
anaprior.models.region_abnormality_predictor
```

Intended data flow:

```text
AFLoc image_encoder_forward(img)
  -> img_emb_l  [B, 768, 19, 19]
  -> img_emb_l2 [B, 768, 38, 38]
  -> pool_region_features(local_features, Chest ImaGenome boxes)
  -> RegionAbnormalityPredictor(region_features, finding_id)
  -> region abnormality logits
```

Region pooling behavior:

- accepts `[C,H,W]` or `[B,C,H,W]` local features;
- maps Chest ImaGenome 224-space boxes to the feature grid by cell centers;
- returns pooled region features, valid region mask, cell counts, and region names.

Predictor behavior:

- consumes pooled region features and a finding id;
- outputs one binary logit per region;
- supports masked BCE loss for invalid or empty regions.

Current module tests:

```powershell
pytest tests\test_region_pooling.py tests\test_region_abnormality_predictor.py -q
```

Next missing piece before training:

```text
extract/cache AFLoc local features for rows in region_finding_train.csv and region_finding_valid.csv
```

Feature cache module:

```text
anaprior.features.extract_region_features
```

It reads `region_finding_*.csv`, runs AFLoc once per DICOM, pools all requested
regions for that DICOM, and writes a `.pt` payload containing:

- `region_features`
- `labels`
- `finding_ids`
- `valid_mask`
- `cell_counts`
- `finding_vocab`
- `metadata`

Server command template:

```bash
python -m anaprior.features.extract_region_features \
  --region-table-csv outputs/anaprior_stage_b_region_table_explicit/region_finding_train.csv \
  --output-path outputs/anaprior_stage_b_feature_cache/train_img_emb_l.pt \
  --ckpt /mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt \
  --image-root /mnt/mimic-cxr/jpg \
  --feature-level img_emb_l \
  --device cuda

python -m anaprior.features.extract_region_features \
  --region-table-csv outputs/anaprior_stage_b_region_table_explicit/region_finding_valid.csv \
  --output-path outputs/anaprior_stage_b_feature_cache/valid_img_emb_l.pt \
  --ckpt /mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt \
  --image-root /mnt/mimic-cxr/jpg \
  --feature-level img_emb_l \
  --device cuda
```

Use `--max-rows 1000` first for a smoke run.

## Stage B Predictor Training Gate

After train/valid feature caches exist, train the lightweight Region Abnormality
Predictor on Chest ImaGenome only:

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

The training script writes:

- `region_predictor.pt`
- `train_report.json`
- `valid_metrics/predictor_heldout_metrics.json`
- `valid_metrics/predictor_heldout_metrics.csv`

You can also rerun held-out evaluation explicitly:

```bash
python -m anaprior.eval.eval_predictor_heldout \
  --checkpoint outputs/anaprior_stage_b_predictor_img_emb_l/region_predictor.pt \
  --cache outputs/anaprior_stage_b_feature_cache/valid_img_emb_l.pt \
  --outdir outputs/anaprior_stage_b_predictor_img_emb_l/valid_metrics \
  --device cuda
```

This gate answers a narrow question:

```text
Can a small head predict Chest ImaGenome region abnormality
from frozen AFLoc local features, without any MS-CXR patient leakage?
```

Only if Pneumothorax and Pleural Effusion show useful held-out AUROC / AP / F1
should Stage C spend one MS-CXR look on learned repair.

## Stage C: Learned Repair

Only after the Region Abnormality Predictor passes its own Chest ImaGenome held-out
task gate should we run MS-CXR learned repair.

Recommended server entry:

```bash
# Smoke run: validates paths and the full command chain on a small subset.
ANAPRIOR_RUN_SMOKE=1 bash scripts/run_stage_c_learned_repair_server.sh

# Full run: uses all MS-CXR cases and full score-request rows.
ANAPRIOR_RUN_SMOKE=0 bash scripts/run_stage_c_learned_repair_server.sh
```

Important environment overrides:

```bash
CKPT=/mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt
MIMIC_IMAGE_ROOT=/mnt/mimic-cxr/jpg
CHEST_IMAGENOME_ROOT=/mnt/chest-imagenome_1.0.0
PRIOR_TABLE=/home/zhangran/zr/G2-VLA-AFLoc/anaprior_assets/prior_table.json
BASE_HMAPS_NPY=/mnt/zhangran/afloc_outputs/anaprior_fixed_gate1_reliable_adaptive/baseline/hmaps.npy
PREDICTOR_CKPT=outputs/anaprior_stage_b_predictor_img_emb_l/region_predictor.pt
AFLOC_BERT_TYPE=/mnt/zhangran/hf_models/Bio_ClinicalBERT
AFLOC_HF_LOCAL_FILES_ONLY=1
OUTROOT=outputs/anaprior_stage_c
STAGE_C_ALPHA=0.3
```

Stage C now has two reusable code-level pieces:

```text
anaprior.eval.prepare_mscxr_repair_inputs
anaprior.eval.predict_region_scores
anaprior.eval.selective_repair
anaprior.eval.eval_mscxr_learned_repair
```

`prepare_mscxr_repair_inputs` creates both the repair input bundle and the
feature-extraction request table for MS-CXR:

```bash
python -m anaprior.eval.prepare_mscxr_repair_inputs \
  --base-hmaps-npy /mnt/zhangran/afloc_outputs/anaprior_fixed_gate1_reliable_adaptive/baseline/hmaps.npy \
  --prior-table /home/zhangran/zr/G2-VLA-AFLoc/anaprior_assets/prior_table.json \
  --chest-imagenome-root /mnt/chest-imagenome_1.0.0 \
  --output-npz outputs/anaprior_stage_c_mscxr/mscxr_repair_inputs.npz \
  --score-request-csv outputs/anaprior_stage_c_mscxr/mscxr_score_request.csv \
  --findings "Pneumothorax,Pleural Effusion"
```

Then extract AFLoc region features for the score-request rows:

```bash
python -m anaprior.features.extract_region_features \
  --region-table-csv outputs/anaprior_stage_c_mscxr/mscxr_score_request.csv \
  --output-path outputs/anaprior_stage_c_mscxr_feature_cache/mscxr_img_emb_l.pt \
  --ckpt /mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt \
  --image-root /mnt/mimic-cxr/jpg \
  --feature-level img_emb_l \
  --device cuda
```

`predict_region_scores` exports predictor outputs from a feature cache:

```bash
python -m anaprior.eval.predict_region_scores \
  --checkpoint outputs/anaprior_stage_b_predictor_img_emb_l/region_predictor.pt \
  --cache outputs/anaprior_stage_c_mscxr_feature_cache/mscxr_img_emb_l.pt \
  --output-csv outputs/anaprior_stage_c_learned_scores/mscxr_region_scores.csv \
  --device cuda
```

Build all pre-registered learned repair heatmaps in one pass:

```bash
python -m anaprior.eval.eval_mscxr_learned_repair \
  --prepared-inputs-npz outputs/anaprior_stage_c_mscxr/mscxr_repair_inputs.npz \
  --region-score-csv outputs/anaprior_stage_c_learned_scores/mscxr_region_scores.csv \
  --outdir outputs/anaprior_stage_c_learned_repair_hmaps \
  --candidate-categories "Pneumothorax,Pleural Effusion" \
  --alpha 0.3 \
  --seed 0
```

Finally score all method heatmaps with the MS-CXR IoU / CNR / Dice evaluator:

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

`selective_repair` is the pure fusion core:

```text
region score CSV / arrays
  -> repair_map_from_region_scores
  -> blend_heatmap(AFLoc heatmap, learned repair map, alpha)
```

It supports the required ablation modes:

- `repair_scope="candidate"`: touch only Pneumothorax / Pleural Effusion.
- `repair_scope="all"`: all-class learned repair ablation.
- `score_mode="learned"`: use predictor scores.
- `score_mode="shuffled"`: preserve score mass but move it to wrong regions.
- `score_mode="uniform"`: remove region-specific abnormality evidence.

Required learned repair comparisons:

- AFLoc baseline
- oracle selective repair
- learned selective repair
- all-class learned repair
- candidate shuffled learned repair
- candidate uniform / no-region-gate repair

All hyperparameters must be frozen before the MS-CXR learned evaluation.

The final Stage C outputs are:

```text
outputs/anaprior_stage_c/preflight_report.json
outputs/anaprior_stage_c_learned_repair_metrics/all_per_case_metrics.csv
outputs/anaprior_stage_c_learned_repair_metrics/delta_table.csv
outputs/anaprior_stage_c_learned_repair_metrics/bootstrap_summary.csv
outputs/anaprior_stage_c_learned_repair_metrics/per_class_bootstrap_summary.csv
outputs/anaprior_stage_c_learned_repair_metrics/learned_repair_decision.json
outputs/anaprior_stage_c_learned_repair_metrics/stage_c_result_report.md
```

The decision requires candidate gain over baseline, specificity over shuffled,
and no macro-all harm.
