#!/usr/bin/env bash
set -euo pipefail

# Stage D oracle recoverability profile server runner.
#
# Smoke run:
#   ANAPRIOR_RUN_SMOKE=1 bash scripts/run_stage_d_oracle_recoverability_server.sh
#
# Full run:
#   ANAPRIOR_RUN_SMOKE=0 bash scripts/run_stage_d_oracle_recoverability_server.sh

export AFLOC_TMPDIR="${AFLOC_TMPDIR:-/mnt3/zhangran/tmp}"
export TMPDIR="${AFLOC_TMPDIR}"
export TEMP="${AFLOC_TMPDIR}"
export TMP="${AFLOC_TMPDIR}"
mkdir -p "${AFLOC_TMPDIR}"

ANAPRIOR_RUN_SMOKE="${ANAPRIOR_RUN_SMOKE:-1}"
ANAPRIOR_OUTPUT_BASE="${ANAPRIOR_OUTPUT_BASE:-/mnt3/zhangran/anaprior_outputs}"
DATASET="${DATASET:-MS_CXR_CLS}"
STAGE_D_ALPHA="${STAGE_D_ALPHA:-0.3}"
SEED="${SEED:-0}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-1000}"
EFFECT_FLOOR="${EFFECT_FLOOR:-0.02}"

OUTROOT="${OUTROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_d_oracle_profile_full}"
PREPARED_INPUTS_NPZ="${PREPARED_INPUTS_NPZ:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_c/mscxr/mscxr_repair_inputs.npz}"

MAX_CASES_ARGS=()
if [[ "${ANAPRIOR_RUN_SMOKE}" == "1" ]]; then
  MAX_CASES="${MAX_CASES:-20}"
  MAX_CASES_ARGS=(--max-cases "${MAX_CASES}")
else
  MAX_CASES="${MAX_CASES:-}"
  if [[ -n "${MAX_CASES}" ]]; then
    MAX_CASES_ARGS=(--max-cases "${MAX_CASES}")
  fi
fi

mkdir -p "${OUTROOT}"

echo "[Stage D] mode=$([[ "${ANAPRIOR_RUN_SMOKE}" == "1" ]] && echo smoke || echo full)"
echo "[Stage D] output base: ${ANAPRIOR_OUTPUT_BASE}"
echo "[Stage D] output root: ${OUTROOT}"
echo "[Stage D] prepared inputs: ${PREPARED_INPUTS_NPZ}"
echo "[Stage D] dataset: ${DATASET}"
echo "[Stage D] alpha: ${STAGE_D_ALPHA}"
echo "[Stage D] bootstrap replicates: ${BOOTSTRAP_REPLICATES}"

python -m anaprior.eval.eval_mscxr_oracle_recoverability \
  --prepared-inputs-npz "${PREPARED_INPUTS_NPZ}" \
  --outdir "${OUTROOT}" \
  --dataset "${DATASET}" \
  --alpha "${STAGE_D_ALPHA}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
  --seed "${SEED}" \
  --effect-floor "${EFFECT_FLOOR}" \
  --margin \
  "${MAX_CASES_ARGS[@]}"

echo "[Stage D] done"
echo "[Stage D] summary: ${OUTROOT}/oracle_recoverability_run_summary.json"
echo "[Stage D] metrics summary: ${OUTROOT}/oracle_recoverability_metrics/oracle_recoverability_summary.json"
echo "[Stage D] profile: ${OUTROOT}/oracle_recoverability_metrics/recoverability_profile.csv"
echo "[Stage D] report: ${OUTROOT}/oracle_recoverability_metrics/oracle_profile_report.md"
if [[ -f "${OUTROOT}/oracle_recoverability_metrics/recoverability_profile.csv" ]]; then
  head -20 "${OUTROOT}/oracle_recoverability_metrics/recoverability_profile.csv"
fi
