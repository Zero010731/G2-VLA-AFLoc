#!/usr/bin/env bash
set -euo pipefail

# Stage B true 8-class region abnormality predictor runner.
#
# Smoke run:
#   ANAPRIOR_RUN_SMOKE=1 bash scripts/run_stage_b_8class_predictor_server.sh
#
# Full run:
#   ANAPRIOR_RUN_SMOKE=0 bash scripts/run_stage_b_8class_predictor_server.sh

export AFLOC_TMPDIR="${AFLOC_TMPDIR:-/mnt3/zhangran/tmp}"
export TMPDIR="${AFLOC_TMPDIR}"
export TEMP="${AFLOC_TMPDIR}"
export TMP="${AFLOC_TMPDIR}"
export AFLOC_HF_LOCAL_FILES_ONLY="${AFLOC_HF_LOCAL_FILES_ONLY:-1}"
mkdir -p "${AFLOC_TMPDIR}"
if [[ -z "${AFLOC_BERT_TYPE:-}" && -d "/mnt/zhangran/Bio_ClinicalBERT" ]]; then
  export AFLOC_BERT_TYPE="/mnt/zhangran/Bio_ClinicalBERT"
fi

ANAPRIOR_RUN_SMOKE="${ANAPRIOR_RUN_SMOKE:-1}"
ANAPRIOR_OUTPUT_BASE="${ANAPRIOR_OUTPUT_BASE:-/mnt3/zhangran/anaprior_outputs}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
FEATURE_LEVEL="${FEATURE_LEVEL:-img_emb_l}"
LABEL_POLICY="${LABEL_POLICY:-explicit}"
SEED="${SEED:-13}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
FINDING_EMBEDDING_DIM="${FINDING_EMBEDDING_DIM:-64}"
DROPOUT="${DROPOUT:-0.0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
FEATURE_DTYPE="${FEATURE_DTYPE:-float16}"
METADATA_MODE="${METADATA_MODE:-none}"
DCEM_V2_RANK_LOSS_WEIGHT="${DCEM_V2_RANK_LOSS_WEIGHT:-0.0}"
DCEM_V2_RANK_MARGIN="${DCEM_V2_RANK_MARGIN:-0.2}"
DCEM_V2_MAX_RANK_PAIRS_PER_FINDING="${DCEM_V2_MAX_RANK_PAIRS_PER_FINDING:-4096}"

EIGHT_FINDINGS="${EIGHT_FINDINGS:-Atelectasis,Cardiomegaly,Consolidation,Edema,Lung Opacity,Pleural Effusion,Pneumonia,Pneumothorax}"

CKPT="${CKPT:-/mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt}"
MIMIC_IMAGE_ROOT="${MIMIC_IMAGE_ROOT:-/mnt/mimic-cxr/jpg}"
CHEST_IMAGENOME_ROOT="${CHEST_IMAGENOME_ROOT:-/mnt/chest-imagenome_1.0.0}"
SPLIT_DIR="${SPLIT_DIR:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_b_splits}"
TRAIN_SPLIT_CSV="${TRAIN_SPLIT_CSV:-${SPLIT_DIR}/imagenome_train_clean.csv}"
VALID_SPLIT_CSV="${VALID_SPLIT_CSV:-${SPLIT_DIR}/imagenome_valid_clean.csv}"

OUTROOT="${OUTROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_b_8class}"
REGION_TABLE_DIR="${REGION_TABLE_DIR:-${OUTROOT}/region_table_${LABEL_POLICY}}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${OUTROOT}/feature_cache_${FEATURE_LEVEL}}"
PREDICTOR_OUTDIR="${PREDICTOR_OUTDIR:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_b_predictor_8class_${FEATURE_LEVEL}}"

TRAIN_REGION_TABLE_CSV="${TRAIN_REGION_TABLE_CSV:-${REGION_TABLE_DIR}/region_finding_train.csv}"
VALID_REGION_TABLE_CSV="${VALID_REGION_TABLE_CSV:-${REGION_TABLE_DIR}/region_finding_valid.csv}"
TRAIN_REGION_REPORT_JSON="${TRAIN_REGION_REPORT_JSON:-${REGION_TABLE_DIR}/region_finding_report_train.json}"
VALID_REGION_REPORT_JSON="${VALID_REGION_REPORT_JSON:-${REGION_TABLE_DIR}/region_finding_report_valid.json}"
TRAIN_FEATURE_CACHE="${TRAIN_FEATURE_CACHE:-${FEATURE_CACHE_DIR}/train_${FEATURE_LEVEL}.pt}"
VALID_FEATURE_CACHE="${VALID_FEATURE_CACHE:-${FEATURE_CACHE_DIR}/valid_${FEATURE_LEVEL}.pt}"
PREDICTOR_CKPT="${PREDICTOR_CKPT:-${PREDICTOR_OUTDIR}/region_predictor.pt}"
export ANAPRIOR_RUN_SMOKE EIGHT_FINDINGS PREDICTOR_CKPT TRAIN_REGION_REPORT_JSON VALID_REGION_REPORT_JSON

TABLE_MAX_ROWS_TRAIN_ARGS=()
TABLE_MAX_ROWS_VALID_ARGS=()
FEATURE_MAX_ROWS_TRAIN_ARGS=()
FEATURE_MAX_ROWS_VALID_ARGS=()
if [[ "${ANAPRIOR_RUN_SMOKE}" == "1" ]]; then
  TABLE_MAX_ROWS_TRAIN="${TABLE_MAX_ROWS_TRAIN:-200}"
  TABLE_MAX_ROWS_VALID="${TABLE_MAX_ROWS_VALID:-80}"
  FEATURE_MAX_ROWS_TRAIN="${FEATURE_MAX_ROWS_TRAIN:-1000}"
  FEATURE_MAX_ROWS_VALID="${FEATURE_MAX_ROWS_VALID:-400}"
  EPOCHS="${SMOKE_EPOCHS:-2}"
  TABLE_MAX_ROWS_TRAIN_ARGS=(--max-rows "${TABLE_MAX_ROWS_TRAIN}")
  TABLE_MAX_ROWS_VALID_ARGS=(--max-rows "${TABLE_MAX_ROWS_VALID}")
  FEATURE_MAX_ROWS_TRAIN_ARGS=(--max-rows "${FEATURE_MAX_ROWS_TRAIN}")
  FEATURE_MAX_ROWS_VALID_ARGS=(--max-rows "${FEATURE_MAX_ROWS_VALID}")
else
  if [[ -n "${TABLE_MAX_ROWS_TRAIN:-}" ]]; then
    TABLE_MAX_ROWS_TRAIN_ARGS=(--max-rows "${TABLE_MAX_ROWS_TRAIN}")
  fi
  if [[ -n "${TABLE_MAX_ROWS_VALID:-}" ]]; then
    TABLE_MAX_ROWS_VALID_ARGS=(--max-rows "${TABLE_MAX_ROWS_VALID}")
  fi
  if [[ -n "${FEATURE_MAX_ROWS_TRAIN:-}" ]]; then
    FEATURE_MAX_ROWS_TRAIN_ARGS=(--max-rows "${FEATURE_MAX_ROWS_TRAIN}")
  fi
  if [[ -n "${FEATURE_MAX_ROWS_VALID:-}" ]]; then
    FEATURE_MAX_ROWS_VALID_ARGS=(--max-rows "${FEATURE_MAX_ROWS_VALID}")
  fi
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${REGION_TABLE_DIR}" "${FEATURE_CACHE_DIR}" "${PREDICTOR_OUTDIR}"

echo "[Stage B 8-class] mode=$([[ "${ANAPRIOR_RUN_SMOKE}" == "1" ]] && echo smoke || echo full)"
echo "[Stage B 8-class] findings: ${EIGHT_FINDINGS}"
echo "[Stage B 8-class] label_policy: ${LABEL_POLICY}"
echo "[Stage B 8-class] feature_level: ${FEATURE_LEVEL}"
echo "[Stage B 8-class] batch_size: ${BATCH_SIZE}"
echo "[Stage B 8-class] eval_batch_size: ${EVAL_BATCH_SIZE}"
echo "[Stage B 8-class] feature_dtype: ${FEATURE_DTYPE}"
echo "[Stage B 8-class] metadata_mode: ${METADATA_MODE}"
echo "[Stage B 8-class] dcem_v2_rank_loss_weight: ${DCEM_V2_RANK_LOSS_WEIGHT}"
echo "[Stage B 8-class] dcem_v2_rank_margin: ${DCEM_V2_RANK_MARGIN}"
echo "[Stage B 8-class] dcem_v2_max_rank_pairs_per_finding: ${DCEM_V2_MAX_RANK_PAIRS_PER_FINDING}"
echo "[Stage B 8-class] output base: ${ANAPRIOR_OUTPUT_BASE}"
echo "[Stage B 8-class] output root: ${OUTROOT}"
echo "[Stage B 8-class] predictor outdir: ${PREDICTOR_OUTDIR}"
echo "[Stage B 8-class] AFLOC_HF_LOCAL_FILES_ONLY=${AFLOC_HF_LOCAL_FILES_ONLY}"
if [[ -n "${AFLOC_BERT_TYPE:-}" ]]; then
  echo "[Stage B 8-class] AFLOC_BERT_TYPE=${AFLOC_BERT_TYPE}"
fi

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[Stage B 8-class] ERROR: required input missing: ${label}=${path}" >&2
    exit 1
  fi
  echo "[Stage B 8-class] found ${label}: ${path}"
}

require_file "CKPT" "${CKPT}"
require_file "TRAIN_SPLIT_CSV" "${TRAIN_SPLIT_CSV}"
require_file "VALID_SPLIT_CSV" "${VALID_SPLIT_CSV}"

echo "[1/5] Building 8-class Chest ImaGenome train table"
python -m anaprior.data.build_region_finding_table \
  --split-csv "${TRAIN_SPLIT_CSV}" \
  --chest-imagenome-root "${CHEST_IMAGENOME_ROOT}" \
  --outdir "${REGION_TABLE_DIR}" \
  --split-name train \
  --findings "${EIGHT_FINDINGS}" \
  --label-policy "${LABEL_POLICY}" \
  "${TABLE_MAX_ROWS_TRAIN_ARGS[@]}"

echo "[2/5] Building 8-class Chest ImaGenome valid table"
python -m anaprior.data.build_region_finding_table \
  --split-csv "${VALID_SPLIT_CSV}" \
  --chest-imagenome-root "${CHEST_IMAGENOME_ROOT}" \
  --outdir "${REGION_TABLE_DIR}" \
  --split-name valid \
  --findings "${EIGHT_FINDINGS}" \
  --label-policy "${LABEL_POLICY}" \
  "${TABLE_MAX_ROWS_VALID_ARGS[@]}"

echo "[Stage B 8-class] Validating train/valid region table coverage"
python - <<'PY'
import json
import os
from pathlib import Path

expected = [item.strip() for item in os.environ["EIGHT_FINDINGS"].split(",") if item.strip()]
reports = {
    "train": Path(os.environ["TRAIN_REGION_REPORT_JSON"]),
    "valid": Path(os.environ["VALID_REGION_REPORT_JSON"]),
}
missing_or_empty = []

for split, path in reports.items():
    if not path.exists():
        missing_or_empty.append(
            {"split": split, "finding": "<report>", "rows": 0, "positive_rows": 0, "path": str(path)}
        )
        continue
    report = json.loads(path.read_text(encoding="utf-8"))
    findings = report.get("findings", {})
    print(f"[Stage B 8-class] {split} coverage from {path}")
    for finding in expected:
        info = findings.get(finding, {})
        rows = int(info.get("rows", 0) or 0)
        positive_rows = int(info.get("positive_rows", 0) or 0)
        print(f"  - {finding}: rows={rows}, positive_rows={positive_rows}")
        if rows <= 0 or positive_rows <= 0:
            missing_or_empty.append(
                {
                    "split": split,
                    "finding": finding,
                    "rows": rows,
                    "positive_rows": positive_rows,
                    "path": str(path),
                }
            )

if missing_or_empty:
    message = f"region table coverage missing_or_empty={missing_or_empty}"
    if os.environ.get("ANAPRIOR_RUN_SMOKE") == "1":
        print(f"[Stage B 8-class] smoke warning: {message}")
        raise SystemExit(0)
    raise SystemExit(f"[Stage B 8-class] ERROR: {message}")
PY

echo "[3/5] Extracting train region features"
python -m anaprior.features.extract_region_features \
  --region-table-csv "${TRAIN_REGION_TABLE_CSV}" \
  --output-path "${TRAIN_FEATURE_CACHE}" \
  --ckpt "${CKPT}" \
  --image-root "${MIMIC_IMAGE_ROOT}" \
  --feature-level "${FEATURE_LEVEL}" \
  --device "${DEVICE}" \
  --progress-every "${PROGRESS_EVERY}" \
  --feature-dtype "${FEATURE_DTYPE}" \
  --metadata-mode "${METADATA_MODE}" \
  "${FEATURE_MAX_ROWS_TRAIN_ARGS[@]}"

echo "[4/5] Extracting valid region features"
python -m anaprior.features.extract_region_features \
  --region-table-csv "${VALID_REGION_TABLE_CSV}" \
  --output-path "${VALID_FEATURE_CACHE}" \
  --ckpt "${CKPT}" \
  --image-root "${MIMIC_IMAGE_ROOT}" \
  --feature-level "${FEATURE_LEVEL}" \
  --device "${DEVICE}" \
  --progress-every "${PROGRESS_EVERY}" \
  --feature-dtype "${FEATURE_DTYPE}" \
  --metadata-mode "${METADATA_MODE}" \
  "${FEATURE_MAX_ROWS_VALID_ARGS[@]}"

echo "[5/5] Training true 8-class region abnormality predictor"
python -m anaprior.train.train_region_predictor \
  --train-cache "${TRAIN_FEATURE_CACHE}" \
  --valid-cache "${VALID_FEATURE_CACHE}" \
  --outdir "${PREDICTOR_OUTDIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --finding-embedding-dim "${FINDING_EMBEDDING_DIM}" \
  --dropout "${DROPOUT}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --rank-loss-weight "${DCEM_V2_RANK_LOSS_WEIGHT}" \
  --rank-margin "${DCEM_V2_RANK_MARGIN}" \
  --max-rank-pairs-per-finding "${DCEM_V2_MAX_RANK_PAIRS_PER_FINDING}"

echo "[Stage B 8-class] Validating checkpoint finding_vocab"
python - <<'PY'
import os
from pathlib import Path

import torch

ckpt_path = Path(os.environ["PREDICTOR_CKPT"])
expected = [item.strip() for item in os.environ["EIGHT_FINDINGS"].split(",") if item.strip()]
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
vocab = checkpoint.get("finding_vocab", {})
actual = sorted(str(key) for key in vocab)
missing = sorted(set(expected) - set(actual))
extra = sorted(set(actual) - set(expected))
print("[Stage B 8-class] checkpoint finding_vocab:", vocab)
if missing or extra:
    if os.environ.get("ANAPRIOR_RUN_SMOKE") == "1":
        print(f"[Stage B 8-class] smoke warning: partial checkpoint finding_vocab missing={missing}, extra={extra}")
        raise SystemExit(0)
    raise SystemExit(
        f"checkpoint finding_vocab mismatch: missing={missing}, extra={extra}, expected={expected}, actual={actual}"
    )
if len(vocab) != 8:
    if os.environ.get("ANAPRIOR_RUN_SMOKE") == "1":
        print(f"[Stage B 8-class] smoke warning: checkpoint finding_vocab has {len(vocab)} findings")
        raise SystemExit(0)
    raise SystemExit(f"checkpoint finding_vocab should have 8 findings, got {len(vocab)}")
PY

echo "[Stage B 8-class] done"
echo "[Stage B 8-class] checkpoint: ${PREDICTOR_CKPT}"
echo "[Stage B 8-class] train report: ${PREDICTOR_OUTDIR}/train_report.json"
echo "[Stage B 8-class] valid metrics: ${PREDICTOR_OUTDIR}/valid_metrics"
