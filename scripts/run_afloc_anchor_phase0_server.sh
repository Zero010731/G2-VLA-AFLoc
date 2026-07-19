#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
START_STAGE="${START_STAGE:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_DIRTY_OUTROOT="${ALLOW_DIRTY_OUTROOT:-0}"
DEVICE="${DEVICE:-cuda}"

ANAPRIOR_OUTPUT_BASE="${ANAPRIOR_OUTPUT_BASE:-/mnt3/zhangran/anaprior_outputs}"
OUTROOT="${OUTROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_k_afloc_anchor_phase0_run1}"
ANCHOR_EVAL_ROOT="${OUTROOT}/anchor_eval"
SCORE_HMAP_ROOT="${OUTROOT}/score_hmaps"
METRIC_ROOT="${OUTROOT}/metrics"
REPORT_ROOT="${OUTROOT}/report"
MANIFEST="${OUTROOT}/anchor_phase0_manifest.json"

METHOD_NAME="${METHOD_NAME:-afloc_anchor}"
BASELINE_METHOD="${BASELINE_METHOD:-baseline}"
DCEM_METHOD="${DCEM_METHOD:-phrase_anatomy_dcem}"
SCORE_METHODS="${SCORE_METHODS:-baseline,phrase_anatomy_dcem,afloc_anchor}"
FINDINGS="${FINDINGS:-Atelectasis,Cardiomegaly,Consolidation,Edema,Lung Opacity,Pleural Effusion,Pneumonia,Pneumothorax}"

AFLOC_CHECKPOINT="${AFLOC_CHECKPOINT:-/mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt}"
AFLOC_BERT_TYPE="${AFLOC_BERT_TYPE:-/mnt/zhangran/Bio_ClinicalBERT}"
LOCALIZATION_MS_CXR_JSON="${LOCALIZATION_MS_CXR_JSON:-/mnt/zhangran/ms-cxr_1.1.0/MS_CXR_Local_Alignment_v1.1.0.json}"
LOCALIZATION_MIMIC_IMG_DIR="${LOCALIZATION_MIMIC_IMG_DIR:-/mnt/mimic-cxr/jpg}"
REFERENCE_HMAPS_ROOT="${REFERENCE_HMAPS_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_c_8class_phrase_validation_gate_alias_v2_dcem_v3/learned_repair_hmaps}"

BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-1000}"
SCORE_VAL_FRACTION="${SCORE_VAL_FRACTION:-0.3}"
SEED="${SEED:-13}"
MAX_CASES="${MAX_CASES:-}"

ANCHOR_HMAPS="${ANCHOR_EVAL_ROOT}/${METHOD_NAME}/hmaps.npy"
ANCHOR_SUMMARY="${ANCHOR_EVAL_ROOT}/anchor_eval_summary.json"
METRIC_SUMMARY="${METRIC_ROOT}/learned_repair_metrics_summary.json"
DECISION_JSON="${REPORT_ROOT}/anchor_phase0_decision.json"

export AFLOC_BERT_TYPE
export AFLOC_HF_LOCAL_FILES_ONLY=1
export TRANSFORMERS_OFFLINE=1
export AFLOC_CHECKPOINT LOCALIZATION_MS_CXR_JSON LOCALIZATION_MIMIC_IMG_DIR

require_file() {
  local name="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[AFLoc Anchor Phase 0] ERROR: missing ${name}: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local name="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[AFLoc Anchor Phase 0] ERROR: missing ${name}: ${path}" >&2
    exit 1
  fi
}

stage_enabled() {
  local stage="$1"
  (( START_STAGE <= stage ))
}

run_cmd() {
  local stage="$1"
  shift
  local label="$1"
  shift
  echo "[${stage}/3] ${label}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

if [[ ! "${START_STAGE}" =~ ^[0-3]$ ]]; then
  echo "[AFLoc Anchor Phase 0] ERROR: START_STAGE must be 0, 1, 2, or 3" >&2
  exit 1
fi

echo "[AFLoc Anchor Phase 0] repo root: ${REPO_ROOT}"
echo "[AFLoc Anchor Phase 0] output root: ${OUTROOT}"
echo "[AFLoc Anchor Phase 0] START_STAGE=${START_STAGE}"
echo "[AFLoc Anchor Phase 0] DEVICE=${DEVICE}"

require_file "AFLOC_CHECKPOINT" "${AFLOC_CHECKPOINT}"
require_dir "AFLOC_BERT_TYPE" "${AFLOC_BERT_TYPE}"
require_file "LOCALIZATION_MS_CXR_JSON" "${LOCALIZATION_MS_CXR_JSON}"
require_dir "LOCALIZATION_MIMIC_IMG_DIR" "${LOCALIZATION_MIMIC_IMG_DIR}"
require_file "baseline hmaps" "${REFERENCE_HMAPS_ROOT}/${BASELINE_METHOD}/hmaps.npy"
require_file "DCEM-v3 hmaps" "${REFERENCE_HMAPS_ROOT}/${DCEM_METHOD}/hmaps.npy"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "[AFLoc Anchor Phase 0] preflight complete"
  exit 0
fi

if [[ "${START_STAGE}" == "0" && -d "${OUTROOT}" ]] \
  && find "${OUTROOT}" -mindepth 1 -print -quit | grep -q . \
  && [[ "${ALLOW_DIRTY_OUTROOT}" != "1" ]]; then
  echo "[AFLoc Anchor Phase 0] ERROR: OUTROOT is not empty: ${OUTROOT}" >&2
  exit 1
fi
if (( START_STAGE > 0 )) && [[ ! -f "${MANIFEST}" ]]; then
  echo "[AFLoc Anchor Phase 0] ERROR: resume requires manifest: ${MANIFEST}" >&2
  exit 1
fi

mkdir -p "${ANCHOR_EVAL_ROOT}" "${SCORE_HMAP_ROOT}" "${METRIC_ROOT}" "${REPORT_ROOT}"

if stage_enabled 0; then
  echo "[0/3] preflight and immutable manifest"
  GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
  export METHOD_NAME SCORE_METHODS SEED BOOTSTRAP_REPLICATES GIT_COMMIT
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


manifest = Path(sys.argv[1])
payload = {
    "status": "frozen_before_anchor_export",
    "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
    "afloc_checkpoint": os.environ["AFLOC_CHECKPOINT"],
    "afloc_checkpoint_sha256": sha256(os.environ["AFLOC_CHECKPOINT"]),
    "bert_type": os.environ["AFLOC_BERT_TYPE"],
    "mscxr_json": os.environ["LOCALIZATION_MS_CXR_JSON"],
    "mscxr_json_sha256": sha256(os.environ["LOCALIZATION_MS_CXR_JSON"]),
    "mimic_img_dir": os.environ["LOCALIZATION_MIMIC_IMG_DIR"],
    "method_name": os.environ["METHOD_NAME"],
    "score_methods": os.environ["SCORE_METHODS"].split(","),
    "seed": int(os.environ["SEED"]),
    "bootstrap_replicates": int(os.environ["BOOTSTRAP_REPLICATES"]),
    "test_evaluated": False,
    "output_hashes": {},
}
manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  fi
else
  echo "[0/3] skipping preflight and immutable manifest"
fi

MAX_CASES_ARGS=()
if [[ -n "${MAX_CASES}" ]]; then
  MAX_CASES_ARGS=(--max-cases "${MAX_CASES}")
fi

if stage_enabled 1; then
  run_cmd 1 "export frozen AFLoc anchor heatmaps" \
    "${PYTHON_BIN}" -m anaprior.eval.eval_mscxr_afloc_anchor \
    --dataset MS_CXR_CLS \
    --split test \
    --afloc-checkpoint "${AFLOC_CHECKPOINT}" \
    --bert-type "${AFLOC_BERT_TYPE}" \
    --hf-local-files-only \
    --ms-cxr-json "${LOCALIZATION_MS_CXR_JSON}" \
    --mimic-img-dir "${LOCALIZATION_MIMIC_IMG_DIR}" \
    --outdir "${ANCHOR_EVAL_ROOT}" \
    --method-name "${METHOD_NAME}" \
    --device "${DEVICE}" \
    "${MAX_CASES_ARGS[@]}"
else
  echo "[1/3] skipping frozen AFLoc anchor export"
fi

if stage_enabled 2; then
  if [[ "${DRY_RUN}" != "1" ]]; then
    require_file "anchor hmaps" "${ANCHOR_HMAPS}"
    mkdir -p \
      "${SCORE_HMAP_ROOT}/${BASELINE_METHOD}" \
      "${SCORE_HMAP_ROOT}/${DCEM_METHOD}" \
      "${SCORE_HMAP_ROOT}/${METHOD_NAME}"
    cp "${REFERENCE_HMAPS_ROOT}/${BASELINE_METHOD}/hmaps.npy" \
      "${SCORE_HMAP_ROOT}/${BASELINE_METHOD}/hmaps.npy"
    cp "${REFERENCE_HMAPS_ROOT}/${DCEM_METHOD}/hmaps.npy" \
      "${SCORE_HMAP_ROOT}/${DCEM_METHOD}/hmaps.npy"
    cp "${ANCHOR_HMAPS}" "${SCORE_HMAP_ROOT}/${METHOD_NAME}/hmaps.npy"
  fi
  run_cmd 2 "score anchor against AFLoc and DCEM-v3" \
    "${PYTHON_BIN}" -m anaprior.eval.score_mscxr_learned_repair_metrics \
    --hmaps-root "${SCORE_HMAP_ROOT}" \
    --outdir "${METRIC_ROOT}" \
    --methods "${SCORE_METHODS}" \
    --candidate-categories "${FINDINGS}" \
    --dataset MS_CXR_CLS \
    --val-fraction "${SCORE_VAL_FRACTION}" \
    --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
    --seed "${SEED}" \
    --candidate-effect-floor 0.0 \
    --macro-all-harm-floor -0.005 \
    --margin
else
  echo "[2/3] skipping anchor scoring"
fi

if stage_enabled 3; then
  if [[ "${DRY_RUN}" != "1" ]]; then
    require_file "anchor summary" "${ANCHOR_SUMMARY}"
    require_file "metric summary" "${METRIC_SUMMARY}"
  fi
  run_cmd 3 "write Phase 0 readiness report" \
    "${PYTHON_BIN}" -m anaprior.eval.report_afloc_anchor_phase0 \
    --anchor-summary "${ANCHOR_SUMMARY}" \
    --metrics-summary "${METRIC_SUMMARY}" \
    --outdir "${REPORT_ROOT}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${PYTHON_BIN}" - "${MANIFEST}" "${ANCHOR_HMAPS}" "${ANCHOR_SUMMARY}" "${METRIC_SUMMARY}" "${DECISION_JSON}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


manifest = Path(sys.argv[1])
payload = json.loads(manifest.read_text(encoding="utf-8"))
payload["status"] = "complete"
payload["test_evaluated"] = True
payload["output_hashes"] = {
    Path(path).name: {"path": path, "sha256": sha256(path)}
    for path in sys.argv[2:]
}
manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  fi
else
  echo "[3/3] skipping readiness report"
fi

echo "[AFLoc Anchor Phase 0] done"
echo "[AFLoc Anchor Phase 0] decision: ${DECISION_JSON}"
