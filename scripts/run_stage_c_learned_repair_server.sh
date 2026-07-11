#!/usr/bin/env bash
set -euo pipefail

# Stage C learned repair server runner.
#
# Default mode is smoke mode to validate paths quickly:
#   ANAPRIOR_RUN_SMOKE=1 bash scripts/run_stage_c_learned_repair_server.sh
#
# Full run:
#   ANAPRIOR_RUN_SMOKE=0 bash scripts/run_stage_c_learned_repair_server.sh
#
# Override paths with environment variables as needed.

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
DATASET="${DATASET:-MS_CXR_CLS}"
FINDINGS="${FINDINGS:-Pneumothorax,Pleural Effusion}"
EXCLUDE_REGIONS="${EXCLUDE_REGIONS:-}"
STAGE_C_ALPHA="${STAGE_C_ALPHA:-0.3}"
SEED="${SEED:-0}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-1000}"
USE_PRECOMPUTED_REGION_SCORE_CSV="${USE_PRECOMPUTED_REGION_SCORE_CSV:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
ENABLE_VALIDATION_GATE="${ENABLE_VALIDATION_GATE:-1}"
STAGE_C_METHODS="${STAGE_C_METHODS:-baseline,learned_selective,disease_gated_learned,disease_pooled_learned,phrase_anatomy_dcem,all_class_learned,candidate_shuffled,candidate_uniform}"
VALIDATION_GATE_SOURCE_METHOD="${VALIDATION_GATE_SOURCE_METHOD:-disease_gated_learned}"
VALIDATION_GATE_METHOD_NAME="${VALIDATION_GATE_METHOD_NAME:-validation_gated_dcem}"
VALIDATION_GATE_EFFECT_FLOOR="${VALIDATION_GATE_EFFECT_FLOOR:-0.02}"
VALIDATION_GATE_CI_LOW_FLOOR="${VALIDATION_GATE_CI_LOW_FLOOR:-0.0}"

CKPT="${CKPT:-/mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt}"
MIMIC_IMAGE_ROOT="${MIMIC_IMAGE_ROOT:-/mnt/mimic-cxr/jpg}"
CHEST_IMAGENOME_ROOT="${CHEST_IMAGENOME_ROOT:-/mnt/chest-imagenome_1.0.0}"
PRIOR_TABLE="${PRIOR_TABLE:-/home/zhangran/zr/G2-VLA-AFLoc/anaprior_assets/prior_table.json}"
BASE_HMAPS_NPY="${BASE_HMAPS_NPY:-/mnt/zhangran/afloc_outputs/anaprior_fixed_gate1_reliable_adaptive/baseline/hmaps.npy}"

OUTROOT="${OUTROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_c}"
MS_CXR_DIR="${MS_CXR_DIR:-${OUTROOT}/mscxr}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${OUTROOT}/feature_cache}"
SCORE_DIR="${SCORE_DIR:-${OUTROOT}/learned_scores}"
HMAP_DIR="${HMAP_DIR:-${OUTROOT}/learned_repair_hmaps}"
METRIC_DIR="${METRIC_DIR:-${OUTROOT}/learned_repair_metrics}"
REPORT_MD="${REPORT_MD:-${METRIC_DIR}/stage_c_result_report.md}"

PREPARED_INPUTS_NPZ="${PREPARED_INPUTS_NPZ:-${MS_CXR_DIR}/mscxr_repair_inputs.npz}"
SCORE_REQUEST_CSV="${SCORE_REQUEST_CSV:-${MS_CXR_DIR}/mscxr_score_request.csv}"
MS_CXR_FEATURE_CACHE="${MS_CXR_FEATURE_CACHE:-${FEATURE_CACHE_DIR}/mscxr_${FEATURE_LEVEL}.pt}"
PREDICTOR_CKPT="${PREDICTOR_CKPT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_b_predictor_${FEATURE_LEVEL}/region_predictor.pt}"
REGION_SCORE_CSV="${REGION_SCORE_CSV:-${SCORE_DIR}/mscxr_region_scores.csv}"

MAX_CASES_ARGS=()
MAX_ROWS_ARGS=()
VALIDATION_GATE_ARGS=()
if [[ "${ANAPRIOR_RUN_SMOKE}" == "1" ]]; then
  MAX_CASES="${MAX_CASES:-20}"
  MAX_ROWS="${MAX_ROWS:-200}"
  MAX_CASES_ARGS=(--max-cases "${MAX_CASES}")
  MAX_ROWS_ARGS=(--max-rows "${MAX_ROWS}")
else
  MAX_CASES="${MAX_CASES:-}"
  MAX_ROWS="${MAX_ROWS:-}"
  if [[ -n "${MAX_CASES}" ]]; then
    MAX_CASES_ARGS=(--max-cases "${MAX_CASES}")
  fi
  if [[ -n "${MAX_ROWS}" ]]; then
    MAX_ROWS_ARGS=(--max-rows "${MAX_ROWS}")
  fi
fi

if [[ "${ENABLE_VALIDATION_GATE}" == "1" ]]; then
  VALIDATION_GATE_ARGS=(
    --validation-gate
    --validation-gate-effect-floor "${VALIDATION_GATE_EFFECT_FLOOR}"
    --validation-gate-ci-low-floor "${VALIDATION_GATE_CI_LOW_FLOOR}"
  )
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
mkdir -p "${MS_CXR_DIR}" "${FEATURE_CACHE_DIR}" "${SCORE_DIR}" "${HMAP_DIR}" "${METRIC_DIR}"

echo "[Stage C] mode=$([[ "${ANAPRIOR_RUN_SMOKE}" == "1" ]] && echo smoke || echo full)"
echo "[Stage C] output base: ${ANAPRIOR_OUTPUT_BASE}"
echo "[Stage C] output root: ${OUTROOT}"
echo "[Stage C] dataset: ${DATASET}"
if [[ -n "${AFLOC_BERT_TYPE:-}" ]]; then
  echo "[Stage C] AFLOC_BERT_TYPE=${AFLOC_BERT_TYPE}"
else
  echo "[Stage C] AFLOC_BERT_TYPE not set; checkpoint BERT name will be used"
fi
echo "[Stage C] AFLOC_HF_LOCAL_FILES_ONLY=${AFLOC_HF_LOCAL_FILES_ONLY}"
echo "[Stage C] USE_PRECOMPUTED_REGION_SCORE_CSV=${USE_PRECOMPUTED_REGION_SCORE_CSV}"
echo "[Stage C] PROGRESS_EVERY=${PROGRESS_EVERY}"
echo "[Stage C] EXCLUDE_REGIONS=${EXCLUDE_REGIONS:-<none>}"
echo "[Stage C] ENABLE_VALIDATION_GATE=${ENABLE_VALIDATION_GATE}"
echo "[Stage C] STAGE_C_METHODS=${STAGE_C_METHODS}"
echo "[Stage C] VALIDATION_GATE_SOURCE_METHOD=${VALIDATION_GATE_SOURCE_METHOD}"
echo "[Stage C] VALIDATION_GATE_METHOD_NAME=${VALIDATION_GATE_METHOD_NAME}"
echo "[Stage C] VALIDATION_GATE_EFFECT_FLOOR=${VALIDATION_GATE_EFFECT_FLOOR}"
echo "[Stage C] VALIDATION_GATE_CI_LOW_FLOOR=${VALIDATION_GATE_CI_LOW_FLOOR}"

echo "[0/5] Preflight checks"
python -m anaprior.eval.stage_c_preflight \
  --ckpt "${CKPT}" \
  --mimic-image-root "${MIMIC_IMAGE_ROOT}" \
  --chest-imagenome-root "${CHEST_IMAGENOME_ROOT}" \
  --prior-table "${PRIOR_TABLE}" \
  --base-hmaps-npy "${BASE_HMAPS_NPY}" \
  --predictor-ckpt "${PREDICTOR_CKPT}" \
  --outroot "${OUTROOT}" \
  --report-json "${OUTROOT}/preflight_report.json"

echo "[1/5] Preparing MS-CXR repair inputs and score-request table"
python -m anaprior.eval.prepare_mscxr_repair_inputs \
  --base-hmaps-npy "${BASE_HMAPS_NPY}" \
  --prior-table "${PRIOR_TABLE}" \
  --chest-imagenome-root "${CHEST_IMAGENOME_ROOT}" \
  --output-npz "${PREPARED_INPUTS_NPZ}" \
  --score-request-csv "${SCORE_REQUEST_CSV}" \
  --dataset "${DATASET}" \
  --findings "${FINDINGS}" \
  --exclude-regions "${EXCLUDE_REGIONS}" \
  "${MAX_CASES_ARGS[@]}"

echo "[2/5] Extracting AFLoc region features for MS-CXR score requests"
if [[ "${USE_PRECOMPUTED_REGION_SCORE_CSV}" == "1" ]]; then
  if [[ ! -f "${REGION_SCORE_CSV}" ]]; then
    echo "[Stage C] ERROR: USE_PRECOMPUTED_REGION_SCORE_CSV=1 but REGION_SCORE_CSV does not exist: ${REGION_SCORE_CSV}" >&2
    exit 1
  fi
  echo "[2/5] Skipping AFLoc feature extraction; using precomputed region score CSV: ${REGION_SCORE_CSV}"
else
  python -m anaprior.features.extract_region_features \
    --region-table-csv "${SCORE_REQUEST_CSV}" \
    --output-path "${MS_CXR_FEATURE_CACHE}" \
    --ckpt "${CKPT}" \
    --image-root "${MIMIC_IMAGE_ROOT}" \
    --feature-level "${FEATURE_LEVEL}" \
    --device "${DEVICE}" \
    --progress-every "${PROGRESS_EVERY}" \
    "${MAX_ROWS_ARGS[@]}"
fi

echo "[3/5] Exporting learned region abnormality scores"
if [[ "${USE_PRECOMPUTED_REGION_SCORE_CSV}" == "1" ]]; then
  echo "[3/5] Skipping learned score export; keeping precomputed score CSV: ${REGION_SCORE_CSV}"
else
  python -m anaprior.eval.predict_region_scores \
    --checkpoint "${PREDICTOR_CKPT}" \
    --cache "${MS_CXR_FEATURE_CACHE}" \
    --output-csv "${REGION_SCORE_CSV}" \
    --device "${DEVICE}"
fi

echo "[4/5] Building pre-registered learned repair heatmaps"
python -m anaprior.eval.eval_mscxr_learned_repair \
  --prepared-inputs-npz "${PREPARED_INPUTS_NPZ}" \
  --region-score-csv "${REGION_SCORE_CSV}" \
  --outdir "${HMAP_DIR}" \
  --candidate-categories "${FINDINGS}" \
  --alpha "${STAGE_C_ALPHA}" \
  --seed "${SEED}"

echo "[5/5] Scoring MS-CXR IoU/CNR/Dice and paired bootstrap"
python -m anaprior.eval.score_mscxr_learned_repair_metrics \
  --hmaps-root "${HMAP_DIR}" \
  --outdir "${METRIC_DIR}" \
  --methods "${STAGE_C_METHODS}" \
  --candidate-categories "${FINDINGS}" \
  --dataset "${DATASET}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
  --seed "${SEED}" \
  --margin \
  --validation-gate-source-method "${VALIDATION_GATE_SOURCE_METHOD}" \
  --validation-gate-method-name "${VALIDATION_GATE_METHOD_NAME}" \
  "${VALIDATION_GATE_ARGS[@]}" \
  "${MAX_CASES_ARGS[@]}"

echo "[Stage C] Generating Chinese result report"
python -m anaprior.eval.report_stage_c_results \
  --metrics-dir "${METRIC_DIR}" \
  --output-md "${REPORT_MD}"

echo "[Stage C] done"
echo "[Stage C] decision: ${METRIC_DIR}/learned_repair_decision.json"
if [[ -f "${METRIC_DIR}/learned_repair_decision.json" ]]; then
  cat "${METRIC_DIR}/learned_repair_decision.json"
fi
echo "[Stage C] report: ${REPORT_MD}"
