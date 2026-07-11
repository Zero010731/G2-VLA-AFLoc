#!/usr/bin/env bash
set -euo pipefail

# Stage F parenchymal failure audit for DCEM-v4 design.
#
# This runner does not train a model and does not tune on MS-CXR boxes. It reads
# already computed Stage C/E artifacts and summarizes the failure modes of:
#   Pneumonia, Consolidation, Lung Opacity.

export AFLOC_TMPDIR="${AFLOC_TMPDIR:-/mnt3/zhangran/tmp}"
export TMPDIR="${AFLOC_TMPDIR}"
export TEMP="${AFLOC_TMPDIR}"
export TMP="${AFLOC_TMPDIR}"
mkdir -p "${AFLOC_TMPDIR}"

ANAPRIOR_OUTPUT_BASE="${ANAPRIOR_OUTPUT_BASE:-/mnt3/zhangran/anaprior_outputs}"
STAGE_C_ROOT="${STAGE_C_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_c_8class_phrase_validation_gate_alias_v2_dcem_v3}"
STAGE_E_ROOT="${STAGE_E_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_e_8class_gap_diagnosis_alias_v2_dcem_v3}"
OUTROOT="${OUTROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_f_failure_audit_dcem_v3}"

CASE_GAP_CSV="${CASE_GAP_CSV:-${STAGE_E_ROOT}/case_gap_table.csv}"
REGION_GAP_CSV="${REGION_GAP_CSV:-${STAGE_E_ROOT}/region_gap_table.csv}"
METRICS_CSV="${METRICS_CSV:-${STAGE_C_ROOT}/learned_repair_metrics/all_per_case_metrics.csv}"
PREPARED_INPUTS_NPZ="${PREPARED_INPUTS_NPZ:-${STAGE_C_ROOT}/mscxr/mscxr_repair_inputs.npz}"

FAILURE_CATEGORIES="${FAILURE_CATEGORIES:-Pneumonia,Consolidation,Lung Opacity}"
METHOD="${METHOD:-phrase_anatomy_dcem}"
BASELINE_METHOD="${BASELINE_METHOD:-baseline}"
MAX_EXAMPLES_PER_CATEGORY="${MAX_EXAMPLES_PER_CATEGORY:-10}"

mkdir -p "${OUTROOT}"

echo "[Stage F] output base: ${ANAPRIOR_OUTPUT_BASE}"
echo "[Stage F] output root: ${OUTROOT}"
echo "[Stage F] Stage C root: ${STAGE_C_ROOT}"
echo "[Stage F] Stage E root: ${STAGE_E_ROOT}"
echo "[Stage F] case gap CSV: ${CASE_GAP_CSV}"
echo "[Stage F] region gap CSV: ${REGION_GAP_CSV}"
echo "[Stage F] metrics CSV: ${METRICS_CSV}"
echo "[Stage F] prepared inputs NPZ: ${PREPARED_INPUTS_NPZ}"
echo "[Stage F] failure categories: ${FAILURE_CATEGORIES}"
echo "[Stage F] method: ${METHOD}"
echo "[Stage F] baseline method: ${BASELINE_METHOD}"

echo "[Stage F] Preflight checks"
missing=0
require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[Stage F] ERROR: required input missing: ${label}=${path}" >&2
    missing=1
  else
    echo "[Stage F] found ${label}: ${path}"
  fi
}

require_file "CASE_GAP_CSV" "${CASE_GAP_CSV}"
require_file "REGION_GAP_CSV" "${REGION_GAP_CSV}"
require_file "METRICS_CSV" "${METRICS_CSV}"
require_file "PREPARED_INPUTS_NPZ" "${PREPARED_INPUTS_NPZ}"

if [[ "${missing}" != "0" ]]; then
  echo "[Stage F] Candidate case gap tables:" >&2
  find "${ANAPRIOR_OUTPUT_BASE}" -type f -name 'case_gap_table.csv' -print 2>/dev/null | sort >&2 || true
  echo "[Stage F] Candidate region gap tables:" >&2
  find "${ANAPRIOR_OUTPUT_BASE}" -type f -name 'region_gap_table.csv' -print 2>/dev/null | sort >&2 || true
  echo "[Stage F] Candidate per-case metrics:" >&2
  find "${ANAPRIOR_OUTPUT_BASE}" -type f -name 'all_per_case_metrics.csv' -print 2>/dev/null | sort >&2 || true
  echo "[Stage F] Candidate prepared inputs:" >&2
  find "${ANAPRIOR_OUTPUT_BASE}" -type f -name '*repair_inputs*.npz' -print 2>/dev/null | sort >&2 || true
  exit 1
fi

python -m anaprior.eval.audit_failure_modes \
  --case-gap-csv "${CASE_GAP_CSV}" \
  --region-gap-csv "${REGION_GAP_CSV}" \
  --metrics-csv "${METRICS_CSV}" \
  --prepared-inputs-npz "${PREPARED_INPUTS_NPZ}" \
  --outdir "${OUTROOT}" \
  --categories "${FAILURE_CATEGORIES}" \
  --method "${METHOD}" \
  --baseline-method "${BASELINE_METHOD}" \
  --max-examples-per-category "${MAX_EXAMPLES_PER_CATEGORY}"

echo "[Stage F] done"
echo "[Stage F] phrase summary: ${OUTROOT}/failure_phrase_summary.csv"
echo "[Stage F] region confusion: ${OUTROOT}/failure_region_confusion.csv"
echo "[Stage F] shortcut summary: ${OUTROOT}/failure_shortcut_summary.csv"
echo "[Stage F] case examples: ${OUTROOT}/failure_case_examples.csv"
echo "[Stage F] recommendations: ${OUTROOT}/failure_v4_recommendations.csv"
echo "[Stage F] report: ${OUTROOT}/failure_audit_report.md"
