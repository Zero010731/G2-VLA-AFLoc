#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
START_STAGE="${START_STAGE:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
DEVICE="${DEVICE:-cuda}"

OUTPUT_BASE="${ANAPRIOR_OUTPUT_BASE:-/mnt3/zhangran/anaprior_outputs}"
OUTROOT="${OUTROOT:-${OUTPUT_BASE}/anaprior_stage_l_afloc_anchor_preserving_phase1_run1}"
PHASE_DIR="${OUTROOT}/phase1_grounding"
EVAL_ROOT="${OUTROOT}/mscxr_eval"
SCORE_ROOT="${OUTROOT}/score_hmaps"
METRIC_ROOT="${OUTROOT}/metrics"
METHOD_NAME="afloc_mrsg"

CACHE_ROOT="${CACHE_ROOT:-${OUTPUT_BASE}/anaprior_stage_j_afloc_mrsg_box_free_full_run3/cache}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${CACHE_ROOT}/train_mrsg.jsonl}"
VALID_MANIFEST="${VALID_MANIFEST:-${CACHE_ROOT}/valid_mrsg.jsonl}"
PROTOCOL_MANIFEST="${PROTOCOL_MANIFEST:-${CACHE_ROOT}/mrsg_protocol_manifest.json}"
DESCRIPTIONS_JSON="${DESCRIPTIONS_JSON:-${REPO_ROOT}/anaprior/configs/mrsg_disease_descriptions.json}"
PHASE0B_PARITY_JSON="${PHASE0B_PARITY_JSON:-${OUTPUT_BASE}/anaprior_stage_k_afloc_official_anchor_phase0b_run1/official_anchor_parity.json}"

AFLOC_CHECKPOINT="${AFLOC_CHECKPOINT:-/mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt}"
AFLOC_BERT_TYPE="${AFLOC_BERT_TYPE:-/mnt/zhangran/Bio_ClinicalBERT}"
MS_CXR_JSON="${MS_CXR_JSON:-/mnt/zhangran/ms-cxr_1.1.0/MS_CXR_Local_Alignment_v1.1.0.json}"
MIMIC_IMAGE_ROOT="${MIMIC_IMAGE_ROOT:-/mnt/mimic-cxr/jpg}"
REFERENCE_HMAPS_ROOT="${REFERENCE_HMAPS_ROOT:-${OUTPUT_BASE}/anaprior_stage_c_8class_phrase_validation_gate_alias_v2_dcem_v3/learned_repair_hmaps}"

EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
MAX_VALID_STEPS="${MAX_VALID_STEPS:-200}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
SEED="${SEED:-13}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-1000}"
PHASE1_RESUME_CHECKPOINT="${PHASE1_RESUME_CHECKPOINT:-}"

CHECKPOINT="${PHASE_DIR}/mrsg_phase_b.pt"
TRAIN_REPORT="${PHASE_DIR}/train_report.json"
GENERATED_HMAPS="${EVAL_ROOT}/${METHOD_NAME}/hmaps.npy"

export AFLOC_BERT_TYPE
export AFLOC_HF_LOCAL_FILES_ONLY=1
export TRANSFORMERS_OFFLINE=1

require_file() {
  local label="$1"
  local path="$2"
  [[ -f "${path}" ]] || { echo "[Phase 1] ERROR: missing ${label}: ${path}" >&2; exit 1; }
}

require_dir() {
  local label="$1"
  local path="$2"
  [[ -d "${path}" ]] || { echo "[Phase 1] ERROR: missing ${label}: ${path}" >&2; exit 1; }
}

stage_enabled() {
  (( START_STAGE <= $1 ))
}

if [[ ! "${START_STAGE}" =~ ^[0-3]$ ]]; then
  echo "[Phase 1] ERROR: START_STAGE must be 0, 1, 2, or 3" >&2
  exit 1
fi

echo "[Phase 1] repo root: ${REPO_ROOT}"
echo "[Phase 1] output root: ${OUTROOT}"
echo "[Phase 1] START_STAGE=${START_STAGE}"

require_file "Phase 0b parity" "${PHASE0B_PARITY_JSON}"
require_file "train manifest" "${TRAIN_MANIFEST}"
require_file "valid manifest" "${VALID_MANIFEST}"
require_file "protocol manifest" "${PROTOCOL_MANIFEST}"
require_file "disease descriptions" "${DESCRIPTIONS_JSON}"
require_file "AFLoc checkpoint" "${AFLOC_CHECKPOINT}"
require_dir "BioClinicalBERT" "${AFLOC_BERT_TYPE}"
require_file "MS-CXR JSON" "${MS_CXR_JSON}"
require_dir "MIMIC image root" "${MIMIC_IMAGE_ROOT}"
require_file "AFLoc baseline heatmaps" "${REFERENCE_HMAPS_ROOT}/baseline/hmaps.npy"
require_file "DCEM-v3 heatmaps" "${REFERENCE_HMAPS_ROOT}/phrase_anatomy_dcem/hmaps.npy"

"${PYTHON_BIN}" - "${PHASE0B_PARITY_JSON}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if not payload["parity_passed"]:
    raise SystemExit("Phase 0b official AFLoc parity did not pass")
PY

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "[Phase 1] preflight complete"
  exit 0
fi

mkdir -p "${PHASE_DIR}" "${EVAL_ROOT}" "${SCORE_ROOT}" "${METRIC_ROOT}"

RESUME_ARGS=()
if [[ -n "${PHASE1_RESUME_CHECKPOINT}" ]]; then
  require_file "Phase 1 resume checkpoint" "${PHASE1_RESUME_CHECKPOINT}"
  RESUME_ARGS=(--resume-checkpoint "${PHASE1_RESUME_CHECKPOINT}")
fi

if stage_enabled 0; then
  echo "[0/3] train anchor-preserving grounding without EMA or cross-view"
  "${PYTHON_BIN}" -m anaprior.train.train_afloc_mrsg \
    --phase grounding \
    --train-manifest "${TRAIN_MANIFEST}" \
    --valid-manifest "${VALID_MANIFEST}" \
    --outdir "${PHASE_DIR}" \
    --afloc-checkpoint "${AFLOC_CHECKPOINT}" \
    --protocol-manifest "${PROTOCOL_MANIFEST}" \
    --descriptions-json "${DESCRIPTIONS_JSON}" \
    "${RESUME_ARGS[@]}" \
    --image-root "${MIMIC_IMAGE_ROOT}" \
    --feature-dim 256 \
    --num-heads 8 \
    --focal-slots 4 \
    --topk-fraction 0.15 \
    --route-temperature 1.0 \
    --residual-logit-bound 0.5 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --max-train-steps "${MAX_TRAIN_STEPS}" \
    --max-valid-steps "${MAX_VALID_STEPS}" \
    --num-workers "${NUM_WORKERS}" \
    --log-every-steps "${LOG_EVERY_STEPS}" \
    --learning-rate 1e-4 \
    --w-ground 1.0 \
    --w-teacher 0.0 \
    --w-mask 0.5 \
    --w-query 0.25 \
    --seed "${SEED}" \
    --device "${DEVICE}"
else
  echo "[0/3] skipping Phase 1 training"
fi

if stage_enabled 1; then
  echo "[1/3] verify bounded residual diagnostics"
  require_file "training report" "${TRAIN_REPORT}"
  require_file "Phase 1 checkpoint" "${CHECKPOINT}"
  "${PYTHON_BIN}" - "${TRAIN_REPORT}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], "r", encoding="utf-8"))
gate = report["phase_gate"]
diagnostics = report["diagnostics"]
keys = (
    "residual_abs_mean",
    "residual_max_abs",
    "correction_abs_mean",
    "correction_max_abs",
    "correction_bound_max",
    "final_anchor_mae",
    "final_anchor_pearson",
)
print("gate passed:", gate["passed"])
print("gate reasons:", gate["reasons"])
for key in keys:
    print(f"{key}: {diagnostics.get(key)}")
if not gate["passed"]:
    raise SystemExit("Phase 1 diagnostics failed; raw evaluation is blocked")
PY
else
  echo "[1/3] skipping residual diagnostic gate"
fi

if stage_enabled 2; then
  echo "[2/3] export raw MS-CXR anchor-preserving heatmaps"
  "${PYTHON_BIN}" -m anaprior.eval.eval_mscxr_afloc_mrsg \
    --dataset MS_CXR \
    --split test \
    --checkpoint "${CHECKPOINT}" \
    --afloc-checkpoint "${AFLOC_CHECKPOINT}" \
    --ms-cxr-json "${MS_CXR_JSON}" \
    --mimic-img-dir "${MIMIC_IMAGE_ROOT}" \
    --outdir "${EVAL_ROOT}" \
    --method-name "${METHOD_NAME}" \
    --device "${DEVICE}"
else
  echo "[2/3] skipping raw heatmap export"
fi

if stage_enabled 3; then
  echo "[3/3] score raw Phase 1 output against AFLoc and DCEM-v3"
  require_file "generated Phase 1 heatmaps" "${GENERATED_HMAPS}"
  mkdir -p "${SCORE_ROOT}/baseline" "${SCORE_ROOT}/phrase_anatomy_dcem" "${SCORE_ROOT}/${METHOD_NAME}"
  cp "${REFERENCE_HMAPS_ROOT}/baseline/hmaps.npy" "${SCORE_ROOT}/baseline/hmaps.npy"
  cp "${REFERENCE_HMAPS_ROOT}/phrase_anatomy_dcem/hmaps.npy" "${SCORE_ROOT}/phrase_anatomy_dcem/hmaps.npy"
  cp "${GENERATED_HMAPS}" "${SCORE_ROOT}/${METHOD_NAME}/hmaps.npy"
  "${PYTHON_BIN}" -m anaprior.eval.score_mscxr_learned_repair_metrics \
    --hmaps-root "${SCORE_ROOT}" \
    --outdir "${METRIC_ROOT}" \
    --methods baseline,phrase_anatomy_dcem,afloc_mrsg \
    --candidate-categories "Atelectasis,Cardiomegaly,Consolidation,Edema,Lung Opacity,Pleural Effusion,Pneumonia,Pneumothorax" \
    --dataset MS_CXR_CLS \
    --val-fraction 0.3 \
    --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
    --seed "${SEED}" \
    --candidate-effect-floor 0.0 \
    --macro-all-harm-floor -0.005 \
    --margin
else
  echo "[3/3] skipping raw scoring"
fi

echo "[Phase 1] done"
echo "[Phase 1] training report: ${TRAIN_REPORT}"
echo "[Phase 1] metrics: ${METRIC_ROOT}/learned_repair_metrics_summary.json"
