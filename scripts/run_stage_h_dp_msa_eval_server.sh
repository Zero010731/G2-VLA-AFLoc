#!/usr/bin/env bash
set -euo pipefail

export AFLOC_TMPDIR="${AFLOC_TMPDIR:-/mnt3/zhangran/tmp}"
export TMPDIR="${AFLOC_TMPDIR}"
export TEMP="${AFLOC_TMPDIR}"
export TMP="${AFLOC_TMPDIR}"
mkdir -p "${AFLOC_TMPDIR}"

ANAPRIOR_OUTPUT_BASE="${ANAPRIOR_OUTPUT_BASE:-/mnt3/zhangran/anaprior_outputs}"
STAGE_C_ROOT="${STAGE_C_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_c_8class_phrase_validation_gate_alias_v2_dcem_v3}"
STAGE_G_ROOT="${STAGE_G_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_g_dp_msa_v0}"
OUTROOT="${OUTROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_h_dp_msa_eval_v0/learned_repair_hmaps}"
PREPARED_INPUTS_NPZ="${PREPARED_INPUTS_NPZ:-${STAGE_C_ROOT}/mscxr/mscxr_repair_inputs.npz}"
REGION_SCORE_CSV="${REGION_SCORE_CSV:-${STAGE_C_ROOT}/learned_scores/mscxr_region_scores.csv}"
BASE_HMAPS_NPY="${BASE_HMAPS_NPY:-${STAGE_C_ROOT}/learned_repair_hmaps/phrase_anatomy_dcem/hmaps.npy}"
BASE_METHOD_NAME="${BASE_METHOD_NAME:-phrase_anatomy_dcem}"
DP_MSA_CKPT="${DP_MSA_CKPT:-${STAGE_G_ROOT}/dp_msa_adapter.pt}"
DEVICE="${DEVICE:-cpu}"
METHOD_NAME="${METHOD_NAME:-dp_msa}"
LAMBDA_OVERRIDE="${LAMBDA_OVERRIDE:-}"

echo "[Stage H DP-MSA] mode=${ANAPRIOR_RUN_SMOKE:-full}"
echo "[Stage H DP-MSA] output base: ${ANAPRIOR_OUTPUT_BASE}"
echo "[Stage H DP-MSA] Stage C root: ${STAGE_C_ROOT}"
echo "[Stage H DP-MSA] Stage G root: ${STAGE_G_ROOT}"
echo "[Stage H DP-MSA] output root: ${OUTROOT}"
echo "[Stage H DP-MSA] prepared inputs: ${PREPARED_INPUTS_NPZ}"
echo "[Stage H DP-MSA] region scores: ${REGION_SCORE_CSV}"
echo "[Stage H DP-MSA] base hmaps: ${BASE_HMAPS_NPY}"
echo "[Stage H DP-MSA] base method: ${BASE_METHOD_NAME}"
echo "[Stage H DP-MSA] checkpoint: ${DP_MSA_CKPT}"
echo "[Stage H DP-MSA] method name: ${METHOD_NAME}"
echo "[Stage H DP-MSA] lambda override: ${LAMBDA_OVERRIDE:-<checkpoint>}"

mkdir -p "${OUTROOT}"

echo "[Stage H DP-MSA] Preflight checks"
for required_file in "${PREPARED_INPUTS_NPZ}" "${REGION_SCORE_CSV}" "${BASE_HMAPS_NPY}" "${DP_MSA_CKPT}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[Stage H DP-MSA] ERROR: required input missing: ${required_file}" >&2
    find "${ANAPRIOR_OUTPUT_BASE}" -maxdepth 5 -type f \( -name "mscxr_repair_inputs.npz" -o -name "mscxr_region_scores.csv" -o -name "dp_msa_adapter.pt" \) 2>/dev/null | sort | head -80 >&2 || true
    exit 1
  fi
  echo "[Stage H DP-MSA] found: ${required_file}"
done

echo "[Stage H DP-MSA] Building DP-MSA repair heatmaps"
CMD=(
  python -m anaprior.eval.eval_mscxr_dp_msa_repair
  --prepared-inputs-npz "${PREPARED_INPUTS_NPZ}"
  --region-score-csv "${REGION_SCORE_CSV}"
  --base-hmaps-npy "${BASE_HMAPS_NPY}"
  --base-method-name "${BASE_METHOD_NAME}"
  --checkpoint "${DP_MSA_CKPT}"
  --outdir "${OUTROOT}"
  --device "${DEVICE}"
  --method-name "${METHOD_NAME}"
)
if [[ -n "${LAMBDA_OVERRIDE}" ]]; then
  CMD+=(--lambda-override "${LAMBDA_OVERRIDE}")
fi
"${CMD[@]}"

echo "[Stage H DP-MSA] summary: ${OUTROOT}/dp_msa_repair_summary.json"
echo "[Stage H DP-MSA] build stats: ${OUTROOT}/build_stats.json"
