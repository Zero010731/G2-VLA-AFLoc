#!/usr/bin/env bash
set -euo pipefail

# End-to-end DP-MSA-v2 property-conditioned refinement runner.
#
# This script assumes Stage C DCEM-v3 artifacts already exist:
#   - mscxr/mscxr_repair_inputs.npz
#   - learned_scores/mscxr_region_scores.csv
#   - learned_repair_hmaps/phrase_anatomy_dcem/hmaps.npy
#
# Full run:
#   ANAPRIOR_RUN_SMOKE=0 bash scripts/run_stage_gh_dp_msa_v2_full_server.sh
#
# Optional quick run:
#   MAX_CASES=64 EPOCHS=1 BOOTSTRAP_REPLICATES=50 bash scripts/run_stage_gh_dp_msa_v2_full_server.sh

export AFLOC_TMPDIR="${AFLOC_TMPDIR:-/mnt3/zhangran/tmp}"
export TMPDIR="${AFLOC_TMPDIR}"
export TEMP="${AFLOC_TMPDIR}"
export TMP="${AFLOC_TMPDIR}"
mkdir -p "${AFLOC_TMPDIR}"

ANAPRIOR_OUTPUT_BASE="${ANAPRIOR_OUTPUT_BASE:-/mnt3/zhangran/anaprior_outputs}"
STAGE_C_ROOT="${STAGE_C_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_c_8class_phrase_validation_gate_alias_v2_dcem_v3}"

RUN_NAME="${RUN_NAME:-dp_msa_v2_property_over_v3}"
DP_MSA_CACHE_ROOT="${DP_MSA_CACHE_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_g_${RUN_NAME}_cache}"
DP_MSA_TRAIN_ROOT="${DP_MSA_TRAIN_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_g_${RUN_NAME}_train}"
DP_MSA_EVAL_ROOT="${DP_MSA_EVAL_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_h_${RUN_NAME}_eval}"
DP_MSA_HMAP_ROOT="${DP_MSA_HMAP_ROOT:-${DP_MSA_EVAL_ROOT}/learned_repair_hmaps}"
DP_MSA_METRIC_ROOT="${DP_MSA_METRIC_ROOT:-${DP_MSA_EVAL_ROOT}/learned_repair_metrics}"
REPORT_MD="${REPORT_MD:-${DP_MSA_METRIC_ROOT}/stage_h_${RUN_NAME}_report.md}"

PREPARED_INPUTS_NPZ="${PREPARED_INPUTS_NPZ:-${STAGE_C_ROOT}/mscxr/mscxr_repair_inputs.npz}"
REGION_SCORE_CSV="${REGION_SCORE_CSV:-${STAGE_C_ROOT}/learned_scores/mscxr_region_scores.csv}"
BASE_HMAPS_NPY="${BASE_HMAPS_NPY:-${STAGE_C_ROOT}/learned_repair_hmaps/phrase_anatomy_dcem/hmaps.npy}"
BASE_METHOD_NAME="${BASE_METHOD_NAME:-phrase_anatomy_dcem}"
DP_MSA_TRAIN_CACHE="${DP_MSA_TRAIN_CACHE:-${DP_MSA_CACHE_ROOT}/train_dp_msa_v0.pt}"
DP_MSA_VALID_CACHE="${DP_MSA_VALID_CACHE:-${DP_MSA_CACHE_ROOT}/valid_dp_msa_v0.pt}"
DP_MSA_CKPT="${DP_MSA_CKPT:-${DP_MSA_TRAIN_ROOT}/dp_msa_adapter.pt}"

DATASET="${DATASET:-MS_CXR_CLS}"
EIGHT_FINDINGS="${FINDINGS:-Atelectasis,Cardiomegaly,Consolidation,Edema,Lung Opacity,Pleural Effusion,Pneumonia,Pneumothorax}"
SOURCE_SPLIT="${SOURCE_SPLIT:-val}"
SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION:-0.3}"
SOURCE_SEED="${SOURCE_SEED:-0}"
CACHE_VALID_FRACTION="${CACHE_VALID_FRACTION:-0.2}"
TARGET_MIX_BETA="${TARGET_MIX_BETA:-0.10}"
DISEASE_BETA_JSON="${DISEASE_BETA_JSON:-}"
MIN_SCORE_SUM="${MIN_SCORE_SUM:-1e-6}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-16}"
EMBEDDING_DIM="${EMBEDDING_DIM:-32}"
LAMBDA_WEIGHT="${LAMBDA_WEIGHT:-0.1}"
LAMBDA_OVERRIDE="${LAMBDA_OVERRIDE:-}"
RESIDUAL_L1_WEIGHT="${RESIDUAL_L1_WEIGHT:-0.01}"
SEED="${SEED:-13}"
DEVICE="${DEVICE:-cpu}"

BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-1000}"
VAL_FRACTION="${VAL_FRACTION:-0.3}"
METHOD_NAME="${METHOD_NAME:-${RUN_NAME}}"
SCORE_METHODS="${SCORE_METHODS:-baseline,${BASE_METHOD_NAME},${METHOD_NAME}}"
ENABLE_VALIDATION_GATE="${ENABLE_VALIDATION_GATE:-1}"
VALIDATION_GATE_SOURCE_METHOD="${VALIDATION_GATE_SOURCE_METHOD:-${METHOD_NAME}}"
VALIDATION_GATE_FALLBACK_METHOD="${VALIDATION_GATE_FALLBACK_METHOD:-${BASE_METHOD_NAME}}"
# Default gate output: validation_gated_dp_msa_v2_property_over_v3
VALIDATION_GATE_METHOD_NAME="${VALIDATION_GATE_METHOD_NAME:-validation_gated_${METHOD_NAME}}"
VALIDATION_GATE_EFFECT_FLOOR="${VALIDATION_GATE_EFFECT_FLOOR:-0.0}"
VALIDATION_GATE_CI_LOW_FLOOR="${VALIDATION_GATE_CI_LOW_FLOOR:--0.005}"
CANDIDATE_EFFECT_FLOOR="${CANDIDATE_EFFECT_FLOOR:-0.0}"
MACRO_ALL_HARM_FLOOR="${MACRO_ALL_HARM_FLOOR:--0.005}"
MAX_CASES="${MAX_CASES:-}"

MAX_CASES_ARGS=()
if [[ -n "${MAX_CASES}" ]]; then
  MAX_CASES_ARGS=(--max-cases "${MAX_CASES}")
fi

VALIDATION_GATE_ARGS=()
if [[ "${ENABLE_VALIDATION_GATE}" == "1" ]]; then
  VALIDATION_GATE_ARGS=(
    --validation-gate
    --validation-gate-source-method "${VALIDATION_GATE_SOURCE_METHOD}"
    --validation-gate-fallback-method "${VALIDATION_GATE_FALLBACK_METHOD}"
    --validation-gate-method-name "${VALIDATION_GATE_METHOD_NAME}"
    --validation-gate-effect-floor "${VALIDATION_GATE_EFFECT_FLOOR}"
    --validation-gate-ci-low-floor "${VALIDATION_GATE_CI_LOW_FLOOR}"
  )
fi

echo "[Stage GH DP-MSA-v2] run name: ${RUN_NAME}"
echo "[Stage GH DP-MSA-v2] output base: ${ANAPRIOR_OUTPUT_BASE}"
echo "[Stage GH DP-MSA-v2] Stage C root: ${STAGE_C_ROOT}"
echo "[Stage GH DP-MSA-v2] cache root: ${DP_MSA_CACHE_ROOT}"
echo "[Stage GH DP-MSA-v2] train root: ${DP_MSA_TRAIN_ROOT}"
echo "[Stage GH DP-MSA-v2] eval root: ${DP_MSA_EVAL_ROOT}"
echo "[Stage GH DP-MSA-v2] hmap root: ${DP_MSA_HMAP_ROOT}"
echo "[Stage GH DP-MSA-v2] metric root: ${DP_MSA_METRIC_ROOT}"
echo "[Stage GH DP-MSA-v2] base method: ${BASE_METHOD_NAME}"
echo "[Stage GH DP-MSA-v2] method name: ${METHOD_NAME}"
echo "[Stage GH DP-MSA-v2] validation gate: ${ENABLE_VALIDATION_GATE}"
echo "[Stage GH DP-MSA-v2] validation gate fallback: ${VALIDATION_GATE_FALLBACK_METHOD}"

mkdir -p "${DP_MSA_CACHE_ROOT}" "${DP_MSA_TRAIN_ROOT}" "${DP_MSA_HMAP_ROOT}" "${DP_MSA_METRIC_ROOT}"

echo "[0/5] Preflight checks"
for required_file in "${PREPARED_INPUTS_NPZ}" "${REGION_SCORE_CSV}" "${BASE_HMAPS_NPY}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[Stage GH DP-MSA-v2] ERROR: required input missing: ${required_file}" >&2
    find "${ANAPRIOR_OUTPUT_BASE}" -maxdepth 6 -type f \( -name "mscxr_repair_inputs.npz" -o -name "mscxr_region_scores.csv" -o -name "hmaps.npy" \) 2>/dev/null | sort | head -100 >&2 || true
    exit 1
  fi
  echo "[Stage GH DP-MSA-v2] found: ${required_file}"
done

echo "[1/5] Building DP-MSA-v2 property-conditioned weak train/valid cache over DCEM-v3"
CACHE_CMD=(
  python -m anaprior.train.build_dp_msa_training_cache
  --prepared-inputs-npz "${PREPARED_INPUTS_NPZ}"
  --region-score-csv "${REGION_SCORE_CSV}"
  --base-hmaps-npy "${BASE_HMAPS_NPY}"
  --base-method-name "${BASE_METHOD_NAME}"
  --outdir "${DP_MSA_CACHE_ROOT}"
  --findings "${EIGHT_FINDINGS}"
  --source-split "${SOURCE_SPLIT}"
  --source-val-fraction "${SOURCE_VAL_FRACTION}"
  --source-seed "${SOURCE_SEED}"
  --valid-fraction "${CACHE_VALID_FRACTION}"
  --seed "${SEED}"
  --min-score-sum "${MIN_SCORE_SUM}"
  --target-mix-beta "${TARGET_MIX_BETA}"
)
if [[ -n "${DISEASE_BETA_JSON}" ]]; then
  CACHE_CMD+=(--disease-beta-json "${DISEASE_BETA_JSON}")
fi
"${CACHE_CMD[@]}"

echo "[2/5] Training DP-MSA-v2 property-conditioned refinement adapter"
python -m anaprior.train.train_dp_msa_adapter \
  --train-cache "${DP_MSA_TRAIN_CACHE}" \
  --valid-cache "${DP_MSA_VALID_CACHE}" \
  --outdir "${DP_MSA_TRAIN_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --hidden-channels "${HIDDEN_CHANNELS}" \
  --embedding-dim "${EMBEDDING_DIM}" \
  --lambda-weight "${LAMBDA_WEIGHT}" \
  --residual-l1-weight "${RESIDUAL_L1_WEIGHT}" \
  --seed "${SEED}" \
  --device "${DEVICE}"

echo "[3/5] Building DP-MSA-v2 repair heatmaps over DCEM-v3 base heatmaps"
EVAL_CMD=(
  python -m anaprior.eval.eval_mscxr_dp_msa_repair
  --prepared-inputs-npz "${PREPARED_INPUTS_NPZ}"
  --region-score-csv "${REGION_SCORE_CSV}"
  --base-hmaps-npy "${BASE_HMAPS_NPY}"
  --base-method-name "${BASE_METHOD_NAME}"
  --checkpoint "${DP_MSA_CKPT}"
  --outdir "${DP_MSA_HMAP_ROOT}"
  --device "${DEVICE}"
  --method-name "${METHOD_NAME}"
)
if [[ -n "${LAMBDA_OVERRIDE}" ]]; then
  EVAL_CMD+=(--lambda-override "${LAMBDA_OVERRIDE}")
fi
"${EVAL_CMD[@]}"

echo "[4/5] Scoring DP-MSA-v2 against baseline and optional validation gate"
mkdir -p "${DP_MSA_HMAP_ROOT}/${BASE_METHOD_NAME}"
cp "${BASE_HMAPS_NPY}" "${DP_MSA_HMAP_ROOT}/${BASE_METHOD_NAME}/hmaps.npy"
python -m anaprior.eval.score_mscxr_learned_repair_metrics \
  --hmaps-root "${DP_MSA_HMAP_ROOT}" \
  --outdir "${DP_MSA_METRIC_ROOT}" \
  --methods "${SCORE_METHODS}" \
  --candidate-categories "${EIGHT_FINDINGS}" \
  --dataset "${DATASET}" \
  --val-fraction "${VAL_FRACTION}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
  --seed "${SEED}" \
  --candidate-effect-floor "${CANDIDATE_EFFECT_FLOOR}" \
  --macro-all-harm-floor "${MACRO_ALL_HARM_FLOOR}" \
  --margin \
  "${VALIDATION_GATE_ARGS[@]}" \
  "${MAX_CASES_ARGS[@]}"

echo "[5/5] Writing result report"
python -m anaprior.eval.report_stage_c_results \
  --metrics-dir "${DP_MSA_METRIC_ROOT}" \
  --output-md "${REPORT_MD}"

echo "[Stage GH DP-MSA-v2] done"
echo "[Stage GH DP-MSA-v2] cache report: ${DP_MSA_CACHE_ROOT}/dp_msa_cache_report.json"
echo "[Stage GH DP-MSA-v2] train report: ${DP_MSA_TRAIN_ROOT}/train_report.json"
echo "[Stage GH DP-MSA-v2] repair summary: ${DP_MSA_HMAP_ROOT}/dp_msa_repair_summary.json"
echo "[Stage GH DP-MSA-v2] decision: ${DP_MSA_METRIC_ROOT}/learned_repair_decision.json"
echo "[Stage GH DP-MSA-v2] report: ${REPORT_MD}"
if [[ -f "${DP_MSA_METRIC_ROOT}/learned_repair_decision.json" ]]; then
  cat "${DP_MSA_METRIC_ROOT}/learned_repair_decision.json"
fi
