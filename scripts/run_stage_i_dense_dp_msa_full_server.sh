#!/usr/bin/env bash
set -euo pipefail

# End-to-end Dense DP-MSA runner.
#
# This is the full DenseCLIP/CLIPSeg-style path:
# spatial AFLoc feature cache -> dense training cache -> dense adapter train
# -> dense hmaps -> scoring with DCEM-v3 fallback gate.

export AFLOC_TMPDIR="${AFLOC_TMPDIR:-/mnt3/zhangran/tmp}"
export TMPDIR="${AFLOC_TMPDIR}"
export TEMP="${AFLOC_TMPDIR}"
export TMP="${AFLOC_TMPDIR}"
export AFLOC_HF_LOCAL_FILES_ONLY="${AFLOC_HF_LOCAL_FILES_ONLY:-1}"
mkdir -p "${AFLOC_TMPDIR}"
if [[ -z "${AFLOC_BERT_TYPE:-}" && -d "/mnt/zhangran/Bio_ClinicalBERT" ]]; then
  export AFLOC_BERT_TYPE="/mnt/zhangran/Bio_ClinicalBERT"
fi

ANAPRIOR_OUTPUT_BASE="${ANAPRIOR_OUTPUT_BASE:-/mnt3/zhangran/anaprior_outputs}"
STAGE_C_ROOT="${STAGE_C_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_c_8class_phrase_validation_gate_alias_v2_dcem_v3}"
RUN_NAME="${RUN_NAME:-dense_dp_msa_over_v3}"

PREPARED_INPUTS_NPZ="${PREPARED_INPUTS_NPZ:-${STAGE_C_ROOT}/mscxr/mscxr_repair_inputs.npz}"
REGION_SCORE_CSV="${REGION_SCORE_CSV:-${STAGE_C_ROOT}/learned_scores/mscxr_region_scores.csv}"
BASE_HMAPS_NPY="${BASE_HMAPS_NPY:-${STAGE_C_ROOT}/learned_repair_hmaps/phrase_anatomy_dcem/hmaps.npy}"
BASE_METHOD_NAME="${BASE_METHOD_NAME:-phrase_anatomy_dcem}"
CKPT="${CKPT:-/mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt}"

DENSE_SPATIAL_CACHE="${DENSE_SPATIAL_CACHE:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_i_${RUN_NAME}/spatial_features.pt}"
DENSE_CACHE_ROOT="${DENSE_CACHE_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_i_${RUN_NAME}/cache}"
DENSE_TRAIN_ROOT="${DENSE_TRAIN_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_i_${RUN_NAME}/train}"
DENSE_EVAL_ROOT="${DENSE_EVAL_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_i_${RUN_NAME}/eval}"
DENSE_HMAP_ROOT="${DENSE_HMAP_ROOT:-${DENSE_EVAL_ROOT}/learned_repair_hmaps}"
DENSE_METRIC_ROOT="${DENSE_METRIC_ROOT:-${DENSE_EVAL_ROOT}/learned_repair_metrics}"
DENSE_CKPT="${DENSE_CKPT:-${DENSE_TRAIN_ROOT}/dense_dp_msa_adapter.pt}"
REPORT_MD="${REPORT_MD:-${DENSE_METRIC_ROOT}/stage_i_${RUN_NAME}_report.md}"

DATASET="${DATASET:-MS_CXR_CLS}"
FINDINGS="${FINDINGS:-Atelectasis,Cardiomegaly,Consolidation,Edema,Lung Opacity,Pleural Effusion,Pneumonia,Pneumothorax}"
FEATURE_LEVEL="${FEATURE_LEVEL:-img_emb_l}"
FEATURE_DTYPE="${FEATURE_DTYPE:-float16}"
SOURCE_SPLIT="${SOURCE_SPLIT:-val}"
SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION:-0.3}"
SOURCE_SEED="${SOURCE_SEED:-0}"
CACHE_VALID_FRACTION="${CACHE_VALID_FRACTION:-0.2}"
TARGET_MIX_BETA="${TARGET_MIX_BETA:-0.10}"
MIN_SCORE_SUM="${MIN_SCORE_SUM:-1e-6}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-32}"
CONDITION_DIM="${CONDITION_DIM:-64}"
LAMBDA_WEIGHT="${LAMBDA_WEIGHT:-0.05}"
LAMBDA_OVERRIDE="${LAMBDA_OVERRIDE:-}"
RESIDUAL_L1_WEIGHT="${RESIDUAL_L1_WEIGHT:-0.01}"
RANKING_LOSS_WEIGHT="${RANKING_LOSS_WEIGHT:-0.10}"
SEED="${SEED:-13}"
DEVICE="${DEVICE:-cuda}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-1000}"
VAL_FRACTION="${VAL_FRACTION:-0.3}"
METHOD_NAME="${METHOD_NAME:-${RUN_NAME}}"
SCORE_METHODS="${SCORE_METHODS:-baseline,${BASE_METHOD_NAME},${METHOD_NAME}}"
ENABLE_VALIDATION_GATE="${ENABLE_VALIDATION_GATE:-1}"
VALIDATION_GATE_SOURCE_METHOD="${VALIDATION_GATE_SOURCE_METHOD:-${METHOD_NAME}}"
VALIDATION_GATE_FALLBACK_METHOD="${VALIDATION_GATE_FALLBACK_METHOD:-${BASE_METHOD_NAME}}"
VALIDATION_GATE_METHOD_NAME="${VALIDATION_GATE_METHOD_NAME:-validation_gated_${METHOD_NAME}}"
VALIDATION_GATE_EFFECT_FLOOR="${VALIDATION_GATE_EFFECT_FLOOR:-0.0}"
VALIDATION_GATE_CI_LOW_FLOOR="${VALIDATION_GATE_CI_LOW_FLOOR:--0.005}"
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

mkdir -p "$(dirname "${DENSE_SPATIAL_CACHE}")" "${DENSE_CACHE_ROOT}" "${DENSE_TRAIN_ROOT}" "${DENSE_HMAP_ROOT}" "${DENSE_METRIC_ROOT}"

echo "[Stage I Dense DP-MSA] run name: ${RUN_NAME}"
echo "[Stage I Dense DP-MSA] Stage C root: ${STAGE_C_ROOT}"
echo "[Stage I Dense DP-MSA] spatial cache: ${DENSE_SPATIAL_CACHE}"
echo "[Stage I Dense DP-MSA] dense cache root: ${DENSE_CACHE_ROOT}"
echo "[Stage I Dense DP-MSA] dense train root: ${DENSE_TRAIN_ROOT}"
echo "[Stage I Dense DP-MSA] dense hmap root: ${DENSE_HMAP_ROOT}"
echo "[Stage I Dense DP-MSA] dense metric root: ${DENSE_METRIC_ROOT}"
echo "[Stage I Dense DP-MSA] AFLOC_HF_LOCAL_FILES_ONLY=${AFLOC_HF_LOCAL_FILES_ONLY}"
echo "[Stage I Dense DP-MSA] AFLOC_BERT_TYPE=${AFLOC_BERT_TYPE:-<checkpoint default>}"

echo "[0/6] Preflight checks"
for required_file in "${PREPARED_INPUTS_NPZ}" "${REGION_SCORE_CSV}" "${BASE_HMAPS_NPY}" "${CKPT}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[Stage I Dense DP-MSA] ERROR: required input missing: ${required_file}" >&2
    exit 1
  fi
  echo "[Stage I Dense DP-MSA] found: ${required_file}"
done

echo "[1/6] Extracting AFLoc spatial feature maps"
python -m anaprior.features.extract_dp_msa_spatial_features \
  --prepared-inputs-npz "${PREPARED_INPUTS_NPZ}" \
  --output-path "${DENSE_SPATIAL_CACHE}" \
  --ckpt "${CKPT}" \
  --bert-type "${AFLOC_BERT_TYPE:-}" \
  --hf-local-files-only "${AFLOC_HF_LOCAL_FILES_ONLY}" \
  --feature-level "${FEATURE_LEVEL}" \
  --device "${DEVICE}" \
  --feature-dtype "${FEATURE_DTYPE}" \
  "${MAX_CASES_ARGS[@]}"

echo "[2/6] Building dense DP-MSA train/valid cache"
python -m anaprior.train.build_dp_msa_training_cache \
  --prepared-inputs-npz "${PREPARED_INPUTS_NPZ}" \
  --region-score-csv "${REGION_SCORE_CSV}" \
  --base-hmaps-npy "${BASE_HMAPS_NPY}" \
  --base-method-name "${BASE_METHOD_NAME}" \
  --spatial-feature-cache "${DENSE_SPATIAL_CACHE}" \
  --outdir "${DENSE_CACHE_ROOT}" \
  --findings "${FINDINGS}" \
  --source-split "${SOURCE_SPLIT}" \
  --source-val-fraction "${SOURCE_VAL_FRACTION}" \
  --source-seed "${SOURCE_SEED}" \
  --valid-fraction "${CACHE_VALID_FRACTION}" \
  --seed "${SEED}" \
  --min-score-sum "${MIN_SCORE_SUM}" \
  --target-mix-beta "${TARGET_MIX_BETA}"

echo "[3/6] Training Dense DP-MSA"
python -m anaprior.train.train_dense_dp_msa_adapter \
  --train-cache "${DENSE_CACHE_ROOT}/train_dp_msa_v0.pt" \
  --valid-cache "${DENSE_CACHE_ROOT}/valid_dp_msa_v0.pt" \
  --outdir "${DENSE_TRAIN_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --hidden-channels "${HIDDEN_CHANNELS}" \
  --condition-dim "${CONDITION_DIM}" \
  --lambda-weight "${LAMBDA_WEIGHT}" \
  --residual-l1-weight "${RESIDUAL_L1_WEIGHT}" \
  --ranking-loss-weight "${RANKING_LOSS_WEIGHT}" \
  --seed "${SEED}" \
  --device "${DEVICE}"

echo "[4/6] Building Dense DP-MSA repair heatmaps"
EVAL_CMD=(
  python -m anaprior.eval.eval_mscxr_dense_dp_msa_repair
  --prepared-inputs-npz "${PREPARED_INPUTS_NPZ}"
  --region-score-csv "${REGION_SCORE_CSV}"
  --spatial-feature-cache "${DENSE_SPATIAL_CACHE}"
  --base-hmaps-npy "${BASE_HMAPS_NPY}"
  --base-method-name "${BASE_METHOD_NAME}"
  --checkpoint "${DENSE_CKPT}"
  --outdir "${DENSE_HMAP_ROOT}"
  --device "${DEVICE}"
  --method-name "${METHOD_NAME}"
)
if [[ -n "${LAMBDA_OVERRIDE}" ]]; then
  EVAL_CMD+=(--lambda-override "${LAMBDA_OVERRIDE}")
fi
"${EVAL_CMD[@]}"

echo "[5/6] Scoring Dense DP-MSA"
mkdir -p "${DENSE_HMAP_ROOT}/${BASE_METHOD_NAME}"
cp "${BASE_HMAPS_NPY}" "${DENSE_HMAP_ROOT}/${BASE_METHOD_NAME}/hmaps.npy"
python -m anaprior.eval.score_mscxr_learned_repair_metrics \
  --hmaps-root "${DENSE_HMAP_ROOT}" \
  --outdir "${DENSE_METRIC_ROOT}" \
  --methods "${SCORE_METHODS}" \
  --candidate-categories "${FINDINGS}" \
  --dataset "${DATASET}" \
  --val-fraction "${VAL_FRACTION}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
  --seed "${SEED}" \
  --margin \
  "${VALIDATION_GATE_ARGS[@]}" \
  "${MAX_CASES_ARGS[@]}"

echo "[6/6] Writing dense result report"
python -m anaprior.eval.report_stage_c_results \
  --metrics-dir "${DENSE_METRIC_ROOT}" \
  --output-md "${REPORT_MD}"

echo "[Stage I Dense DP-MSA] done"
echo "[Stage I Dense DP-MSA] report: ${REPORT_MD}"
echo "[Stage I Dense DP-MSA] decision: ${DENSE_METRIC_ROOT}/learned_repair_decision.json"
if [[ -f "${DENSE_METRIC_ROOT}/learned_repair_decision.json" ]]; then
  cat "${DENSE_METRIC_ROOT}/learned_repair_decision.json"
fi
