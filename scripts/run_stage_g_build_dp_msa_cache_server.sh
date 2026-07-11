#!/usr/bin/env bash
set -euo pipefail

export AFLOC_TMPDIR="${AFLOC_TMPDIR:-/mnt3/zhangran/tmp}"
export TMPDIR="${AFLOC_TMPDIR}"
export TEMP="${AFLOC_TMPDIR}"
export TMP="${AFLOC_TMPDIR}"
mkdir -p "${AFLOC_TMPDIR}"

ANAPRIOR_OUTPUT_BASE="${ANAPRIOR_OUTPUT_BASE:-/mnt3/zhangran/anaprior_outputs}"
STAGE_C_ROOT="${STAGE_C_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_c_8class_phrase_validation_gate_alias_v2_dcem_v3}"
OUTROOT="${OUTROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_g_dp_msa_cache}"
PREPARED_INPUTS_NPZ="${PREPARED_INPUTS_NPZ:-${STAGE_C_ROOT}/mscxr/mscxr_repair_inputs.npz}"
REGION_SCORE_CSV="${REGION_SCORE_CSV:-${STAGE_C_ROOT}/learned_scores/mscxr_region_scores.csv}"
BASE_HMAPS_NPY="${BASE_HMAPS_NPY:-${STAGE_C_ROOT}/learned_repair_hmaps/phrase_anatomy_dcem/hmaps.npy}"
BASE_METHOD_NAME="${BASE_METHOD_NAME:-phrase_anatomy_dcem}"
EIGHT_FINDINGS="${FINDINGS:-Atelectasis,Cardiomegaly,Consolidation,Edema,Lung Opacity,Pleural Effusion,Pneumonia,Pneumothorax}"
SOURCE_SPLIT="${SOURCE_SPLIT:-val}"
SOURCE_VAL_FRACTION="${SOURCE_VAL_FRACTION:-0.3}"
SOURCE_SEED="${SOURCE_SEED:-0}"
CACHE_VALID_FRACTION="${CACHE_VALID_FRACTION:-0.2}"
SEED="${SEED:-13}"
MIN_SCORE_SUM="${MIN_SCORE_SUM:-1e-6}"
TARGET_MIX_BETA="${TARGET_MIX_BETA:-0.10}"
DISEASE_BETA_JSON="${DISEASE_BETA_JSON:-}"

echo "[Stage G0 DP-MSA cache] mode=${ANAPRIOR_RUN_SMOKE:-full}"
echo "[Stage G0 DP-MSA cache] output base: ${ANAPRIOR_OUTPUT_BASE}"
echo "[Stage G0 DP-MSA cache] Stage C root: ${STAGE_C_ROOT}"
echo "[Stage G0 DP-MSA cache] output root: ${OUTROOT}"
echo "[Stage G0 DP-MSA cache] prepared inputs: ${PREPARED_INPUTS_NPZ}"
echo "[Stage G0 DP-MSA cache] region scores: ${REGION_SCORE_CSV}"
echo "[Stage G0 DP-MSA cache] base hmaps: ${BASE_HMAPS_NPY}"
echo "[Stage G0 DP-MSA cache] base method: ${BASE_METHOD_NAME}"
echo "[Stage G0 DP-MSA cache] findings: ${EIGHT_FINDINGS}"
echo "[Stage G0 DP-MSA cache] source split: ${SOURCE_SPLIT}"
echo "[Stage G0 DP-MSA cache] target mix beta: ${TARGET_MIX_BETA}"

mkdir -p "${OUTROOT}"

echo "[Stage G0 DP-MSA cache] Preflight checks"
for required_file in "${PREPARED_INPUTS_NPZ}" "${REGION_SCORE_CSV}" "${BASE_HMAPS_NPY}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[Stage G0 DP-MSA cache] ERROR: required input missing: ${required_file}" >&2
    find "${ANAPRIOR_OUTPUT_BASE}" -maxdepth 5 -type f \( -name "mscxr_repair_inputs.npz" -o -name "mscxr_region_scores.csv" \) 2>/dev/null | sort | head -80 >&2 || true
    exit 1
  fi
  echo "[Stage G0 DP-MSA cache] found: ${required_file}"
done

echo "[Stage G0 DP-MSA cache] Building weak DP-MSA train/valid cache"
CMD=(
  python -m anaprior.train.build_dp_msa_training_cache
  --prepared-inputs-npz "${PREPARED_INPUTS_NPZ}" \
  --region-score-csv "${REGION_SCORE_CSV}" \
  --base-hmaps-npy "${BASE_HMAPS_NPY}" \
  --base-method-name "${BASE_METHOD_NAME}" \
  --outdir "${OUTROOT}" \
  --findings "${EIGHT_FINDINGS}" \
  --source-split "${SOURCE_SPLIT}" \
  --source-val-fraction "${SOURCE_VAL_FRACTION}" \
  --source-seed "${SOURCE_SEED}" \
  --valid-fraction "${CACHE_VALID_FRACTION}" \
  --seed "${SEED}" \
  --min-score-sum "${MIN_SCORE_SUM}" \
  --target-mix-beta "${TARGET_MIX_BETA}"
)
if [[ -n "${DISEASE_BETA_JSON}" ]]; then
  CMD+=(--disease-beta-json "${DISEASE_BETA_JSON}")
fi
"${CMD[@]}"

echo "[Stage G0 DP-MSA cache] train cache: ${OUTROOT}/train_dp_msa_v0.pt"
echo "[Stage G0 DP-MSA cache] valid cache: ${OUTROOT}/valid_dp_msa_v0.pt"
echo "[Stage G0 DP-MSA cache] report: ${OUTROOT}/dp_msa_cache_report.json"
