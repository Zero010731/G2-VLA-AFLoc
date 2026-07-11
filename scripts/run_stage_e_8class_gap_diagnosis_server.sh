#!/usr/bin/env bash
set -euo pipefail

# Stage E 8-class learned-vs-oracle gap diagnosis.
#
# This runner does not train a new model. It merges:
#   1) prepared MS-CXR region maps,
#   2) learned region scores,
#   3) learned repair per-case metrics,
#   4) Stage D oracle recoverability profile.
#
# Output: gap_diagnosis.csv with one diagnosis row per pathology.

export AFLOC_TMPDIR="${AFLOC_TMPDIR:-/mnt3/zhangran/tmp}"
export TMPDIR="${AFLOC_TMPDIR}"
export TEMP="${AFLOC_TMPDIR}"
export TMP="${AFLOC_TMPDIR}"
mkdir -p "${AFLOC_TMPDIR}"

ANAPRIOR_OUTPUT_BASE="${ANAPRIOR_OUTPUT_BASE:-/mnt3/zhangran/anaprior_outputs}"
DATASET="${DATASET:-MS_CXR_CLS}"
EIGHT_FINDINGS="${EIGHT_FINDINGS:-Atelectasis,Cardiomegaly,Consolidation,Edema,Lung Opacity,Pleural Effusion,Pneumonia,Pneumothorax}"
OUTROOT="${OUTROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_e_8class_gap_diagnosis}"

PREPARED_INPUTS_NPZ="${PREPARED_INPUTS_NPZ:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_c/mscxr/mscxr_repair_inputs.npz}"
REGION_SCORE_CSV="${REGION_SCORE_CSV:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_c/learned_scores/mscxr_region_scores.csv}"
LEARNED_METRICS_CSV="${LEARNED_METRICS_CSV:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_c/learned_repair_metrics/all_per_case_metrics.csv}"
ORACLE_PROFILE_CSV="${ORACLE_PROFILE_CSV:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_d_oracle_profile_full/oracle_recoverability_metrics/recoverability_profile.csv}"

METHOD="${METHOD:-learned_selective}"
BASELINE_METHOD="${BASELINE_METHOD:-baseline}"
EVIDENCE_MODE="${EVIDENCE_MODE:-learned}"
ORACLE_OVERLAP_FLOOR="${ORACLE_OVERLAP_FLOOR:-0.25}"
EFFECT_FLOOR="${EFFECT_FLOOR:-0.02}"
LEARNED_SUCCESS_FLOOR="${LEARNED_SUCCESS_FLOOR:-0.02}"

mkdir -p "${OUTROOT}"

echo "[Stage E] output root: ${OUTROOT}"
echo "[Stage E] output base: ${ANAPRIOR_OUTPUT_BASE}"
echo "[Stage E] dataset: ${DATASET}"
echo "[Stage E] findings: ${EIGHT_FINDINGS}"
echo "[Stage E] prepared inputs: ${PREPARED_INPUTS_NPZ}"
echo "[Stage E] learned scores: ${REGION_SCORE_CSV}"
echo "[Stage E] learned metrics: ${LEARNED_METRICS_CSV}"
echo "[Stage E] oracle profile: ${ORACLE_PROFILE_CSV}"
echo "[Stage E] method: ${METHOD}"
echo "[Stage E] evidence mode: ${EVIDENCE_MODE}"

echo "[Stage E] Preflight checks"
missing=0
require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[Stage E] ERROR: required input missing: ${label}=${path}" >&2
    missing=1
  else
    echo "[Stage E] found ${label}: ${path}"
  fi
}

require_file "PREPARED_INPUTS_NPZ" "${PREPARED_INPUTS_NPZ}"
require_file "REGION_SCORE_CSV" "${REGION_SCORE_CSV}"
require_file "LEARNED_METRICS_CSV" "${LEARNED_METRICS_CSV}"
require_file "ORACLE_PROFILE_CSV" "${ORACLE_PROFILE_CSV}"

if [[ "${missing}" != "0" ]]; then
  echo "[Stage E] Candidate prepared input files:" >&2
  find "${ANAPRIOR_OUTPUT_BASE}" -type f \( -name '*repair_inputs*.npz' -o -name '*mscxr*.npz' \) -print 2>/dev/null | sort >&2 || true
  echo "[Stage E] Candidate region score CSV files:" >&2
  find "${ANAPRIOR_OUTPUT_BASE}" -type f -name '*region_scores*.csv' -print 2>/dev/null | sort >&2 || true
  echo "[Stage E] Candidate learned metrics CSV files:" >&2
  find "${ANAPRIOR_OUTPUT_BASE}" -type f -name 'all_per_case_metrics.csv' -print 2>/dev/null | sort >&2 || true
  echo "[Stage E] Candidate oracle profile CSV files:" >&2
  find "${ANAPRIOR_OUTPUT_BASE}" -type f -name 'recoverability_profile.csv' -print 2>/dev/null | sort >&2 || true
  exit 1
fi

python -m anaprior.eval.analyze_learned_oracle_gap \
  --prepared-inputs-npz "${PREPARED_INPUTS_NPZ}" \
  --region-score-csv "${REGION_SCORE_CSV}" \
  --metrics-csv "${LEARNED_METRICS_CSV}" \
  --outdir "${OUTROOT}" \
  --dataset "${DATASET}" \
  --candidate-categories "${EIGHT_FINDINGS}" \
  --method "${METHOD}" \
  --baseline-method "${BASELINE_METHOD}" \
  --evidence-mode "${EVIDENCE_MODE}" \
  --oracle-profile-csv "${ORACLE_PROFILE_CSV}" \
  --oracle-overlap-floor "${ORACLE_OVERLAP_FLOOR}" \
  --effect-floor "${EFFECT_FLOOR}" \
  --learned-success-floor "${LEARNED_SUCCESS_FLOOR}"

echo "[Stage E] done"
echo "[Stage E] diagnosis: ${OUTROOT}/gap_diagnosis.csv"
echo "[Stage E] per-class gap summary: ${OUTROOT}/per_class_gap_summary.csv"
echo "[Stage E] report: ${OUTROOT}/gap_analysis_report.md"
if [[ -f "${OUTROOT}/gap_diagnosis.csv" ]]; then
  cat "${OUTROOT}/gap_diagnosis.csv"
fi
