#!/usr/bin/env bash
set -euo pipefail

# Stage B patient-disjoint split runner.
#
# Rebuilds the clean Chest ImaGenome splits after migration or output loss.
# All outputs default under ANAPRIOR_OUTPUT_BASE so regenerated artifacts do not
# scatter inside the repository checkout.

ANAPRIOR_OUTPUT_BASE="${ANAPRIOR_OUTPUT_BASE:-/mnt3/zhangran/anaprior_outputs}"
MS_CXR_JSON="${MS_CXR_JSON:-/home/zhangran/zr/G2-VLA-AFLoc/data/ms-cxr/1.1.0/MS_CXR_Local_Alignment_v1.1.0_radgraph_phrase.json}"
CHEST_IMAGENOME_ROOT="${CHEST_IMAGENOME_ROOT:-/mnt/chest-imagenome_1.0.0}"
SPLIT_DIR="${SPLIT_DIR:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_b_splits}"
export SPLIT_DIR

mkdir -p "${SPLIT_DIR}"

echo "[Stage B splits] output base: ${ANAPRIOR_OUTPUT_BASE}"
echo "[Stage B splits] split dir: ${SPLIT_DIR}"
echo "[Stage B splits] MS_CXR_JSON: ${MS_CXR_JSON}"
echo "[Stage B splits] CHEST_IMAGENOME_ROOT: ${CHEST_IMAGENOME_ROOT}"

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[Stage B splits] ERROR: required input missing: ${label}=${path}" >&2
    exit 1
  fi
  echo "[Stage B splits] found ${label}: ${path}"
}

require_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[Stage B splits] ERROR: required directory missing: ${label}=${path}" >&2
    exit 1
  fi
  echo "[Stage B splits] found ${label}: ${path}"
}

require_file "MS_CXR_JSON" "${MS_CXR_JSON}"
require_dir "CHEST_IMAGENOME_ROOT" "${CHEST_IMAGENOME_ROOT}"

python -m anaprior.data.build_patient_splits \
  --mscxr-json "${MS_CXR_JSON}" \
  --chest-imagenome-root "${CHEST_IMAGENOME_ROOT}" \
  --outdir "${SPLIT_DIR}"

echo "[Stage B splits] done"
echo "[Stage B splits] leakage report: ${SPLIT_DIR}/leakage_report.json"
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["SPLIT_DIR"]) / "leakage_report.json"
report = json.loads(path.read_text(encoding="utf-8"))
sanity = report.get("sanity", {})
print("[Stage B splits] train_mscxr_overlap:", sanity.get("train_mscxr_overlap"))
print("[Stage B splits] valid_mscxr_overlap:", sanity.get("valid_mscxr_overlap"))
print("[Stage B splits] test_mscxr_overlap:", sanity.get("test_mscxr_overlap"))
print("[Stage B splits] train_valid_overlap:", sanity.get("train_valid_overlap"))
if any(int(sanity.get(key, 0)) != 0 for key in ["train_mscxr_overlap", "valid_mscxr_overlap"]):
    raise SystemExit("clean split sanity failed: train/valid overlap with MS-CXR is nonzero")
PY
