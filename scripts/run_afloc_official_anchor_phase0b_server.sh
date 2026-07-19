#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
START_STAGE="${START_STAGE:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
DEVICE="${DEVICE:-cuda}"

ANAPRIOR_OUTPUT_BASE="${ANAPRIOR_OUTPUT_BASE:-/mnt3/zhangran/anaprior_outputs}"
OUTROOT="${OUTROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_k_afloc_official_anchor_phase0b_run1}"
METHOD_NAME="${METHOD_NAME:-afloc_official_anchor}"
EVAL_ROOT="${OUTROOT}/official_anchor_eval"
MANIFEST="${OUTROOT}/official_anchor_phase0b_manifest.json"
PARITY_JSON="${OUTROOT}/official_anchor_parity.json"

AFLOC_CHECKPOINT="${AFLOC_CHECKPOINT:-/mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt}"
AFLOC_BERT_TYPE="${AFLOC_BERT_TYPE:-/mnt/zhangran/Bio_ClinicalBERT}"
LOCALIZATION_MS_CXR_JSON="${LOCALIZATION_MS_CXR_JSON:-/mnt/zhangran/ms-cxr_1.1.0/MS_CXR_Local_Alignment_v1.1.0.json}"
LOCALIZATION_MIMIC_IMG_DIR="${LOCALIZATION_MIMIC_IMG_DIR:-/mnt/mimic-cxr/jpg}"
REFERENCE_HMAPS_ROOT="${REFERENCE_HMAPS_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_c_8class_phrase_validation_gate_alias_v2_dcem_v3/learned_repair_hmaps}"
REFERENCE_HMAPS="${REFERENCE_HMAPS_ROOT}/baseline/hmaps.npy"
GENERATED_HMAPS="${EVAL_ROOT}/${METHOD_NAME}/hmaps.npy"
MAX_CASES="${MAX_CASES:-}"

export AFLOC_BERT_TYPE
export AFLOC_HF_LOCAL_FILES_ONLY=1
export TRANSFORMERS_OFFLINE=1

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[AFLoc Official Anchor Phase 0b] ERROR: missing ${name}: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local name="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[AFLoc Official Anchor Phase 0b] ERROR: missing ${name}: ${path}" >&2
    exit 1
  fi
}

stage_enabled() {
  local stage="$1"
  (( START_STAGE <= stage ))
}

if [[ ! "${START_STAGE}" =~ ^[0-2]$ ]]; then
  echo "[AFLoc Official Anchor Phase 0b] ERROR: START_STAGE must be 0, 1, or 2" >&2
  exit 1
fi

echo "[AFLoc Official Anchor Phase 0b] repo root: ${REPO_ROOT}"
echo "[AFLoc Official Anchor Phase 0b] output root: ${OUTROOT}"
echo "[AFLoc Official Anchor Phase 0b] START_STAGE=${START_STAGE}"
echo "[AFLoc Official Anchor Phase 0b] DEVICE=${DEVICE}"

require_file "AFLoc checkpoint" "${AFLOC_CHECKPOINT}"
require_dir "local BioClinicalBERT" "${AFLOC_BERT_TYPE}"
require_file "MS-CXR JSON" "${LOCALIZATION_MS_CXR_JSON}"
require_dir "MIMIC image root" "${LOCALIZATION_MIMIC_IMG_DIR}"
require_file "saved AFLoc baseline heatmaps" "${REFERENCE_HMAPS}"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "[AFLoc Official Anchor Phase 0b] preflight complete"
  exit 0
fi

mkdir -p "${OUTROOT}" "${EVAL_ROOT}"

if stage_enabled 0; then
  echo "[0/2] freeze parity manifest"
  export OUTROOT METHOD_NAME AFLOC_CHECKPOINT LOCALIZATION_MS_CXR_JSON REFERENCE_HMAPS
  "${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": "frozen_before_export",
    "method_name": os.environ["METHOD_NAME"],
    "afloc_checkpoint": os.environ["AFLOC_CHECKPOINT"],
    "mscxr_json": os.environ["LOCALIZATION_MS_CXR_JSON"],
    "reference_hmaps": os.environ["REFERENCE_HMAPS"],
    "formula": "official_iel_x_global_report_gaussian_1.5_bilinear",
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
else
  echo "[0/2] skipping parity manifest"
  require_file "parity manifest" "${MANIFEST}"
fi

MAX_CASES_ARGS=()
if [[ -n "${MAX_CASES}" ]]; then
  MAX_CASES_ARGS=(--max-cases "${MAX_CASES}")
fi

if stage_enabled 1; then
  echo "[1/2] export exact official AFLoc heatmaps"
  "${PYTHON_BIN}" -m anaprior.eval.eval_mscxr_afloc_official_anchor \
    --dataset MS_CXR \
    --split test \
    --afloc-checkpoint "${AFLOC_CHECKPOINT}" \
    --bert-type "${AFLOC_BERT_TYPE}" \
    --hf-local-files-only \
    --ms-cxr-json "${LOCALIZATION_MS_CXR_JSON}" \
    --mimic-img-dir "${LOCALIZATION_MIMIC_IMG_DIR}" \
    --outdir "${EVAL_ROOT}" \
    --method-name "${METHOD_NAME}" \
    --device "${DEVICE}" \
    "${MAX_CASES_ARGS[@]}"
else
  echo "[1/2] skipping official AFLoc heatmap export"
fi

if stage_enabled 2; then
  require_file "generated official heatmaps" "${GENERATED_HMAPS}"
  echo "[2/2] compare pixels with saved AFLoc baseline"
  "${PYTHON_BIN}" -m anaprior.eval.report_afloc_official_anchor_parity \
    --generated-hmaps "${GENERATED_HMAPS}" \
    --reference-hmaps "${REFERENCE_HMAPS}" \
    --min-matched-cases 900 \
    --out "${PARITY_JSON}"
  "${PYTHON_BIN}" - "${MANIFEST}" "${PARITY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
parity_path = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
parity = json.loads(parity_path.read_text(encoding="utf-8"))
manifest["status"] = "parity_passed" if parity["parity_passed"] else "parity_failed"
manifest["parity_report"] = str(parity_path)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
if not parity["parity_passed"]:
    raise SystemExit("official AFLoc parity failed; do not start MRSG training")
PY
else
  echo "[2/2] skipping pixel parity report"
fi

echo "[AFLoc Official Anchor Phase 0b] done"
echo "[AFLoc Official Anchor Phase 0b] parity: ${PARITY_JSON}"
