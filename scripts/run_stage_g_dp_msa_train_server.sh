#!/usr/bin/env bash
set -euo pipefail

export AFLOC_TMPDIR="${AFLOC_TMPDIR:-/mnt3/zhangran/tmp}"
export TMPDIR="${AFLOC_TMPDIR}"
export TEMP="${AFLOC_TMPDIR}"
export TMP="${AFLOC_TMPDIR}"
mkdir -p "${AFLOC_TMPDIR}"

ANAPRIOR_OUTPUT_BASE="${ANAPRIOR_OUTPUT_BASE:-/mnt3/zhangran/anaprior_outputs}"
OUTROOT="${OUTROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_g_dp_msa_v0}"
DP_MSA_TRAIN_CACHE="${DP_MSA_TRAIN_CACHE:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_g_dp_msa_cache/train_dp_msa_v0.pt}"
DP_MSA_VALID_CACHE="${DP_MSA_VALID_CACHE:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_g_dp_msa_cache/valid_dp_msa_v0.pt}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-16}"
EMBEDDING_DIM="${EMBEDDING_DIM:-32}"
LAMBDA_WEIGHT="${LAMBDA_WEIGHT:-0.1}"
RESIDUAL_L1_WEIGHT="${RESIDUAL_L1_WEIGHT:-0.01}"
SEED="${SEED:-13}"
DEVICE="${DEVICE:-cpu}"

echo "[Stage G DP-MSA] mode=${ANAPRIOR_RUN_SMOKE:-full}"
echo "[Stage G DP-MSA] output base: ${ANAPRIOR_OUTPUT_BASE}"
echo "[Stage G DP-MSA] output root: ${OUTROOT}"
echo "[Stage G DP-MSA] train cache: ${DP_MSA_TRAIN_CACHE}"
echo "[Stage G DP-MSA] valid cache: ${DP_MSA_VALID_CACHE}"
echo "[Stage G DP-MSA] device: ${DEVICE}"

mkdir -p "${OUTROOT}"

echo "[Stage G DP-MSA] Preflight checks"
for required_file in "${DP_MSA_TRAIN_CACHE}" "${DP_MSA_VALID_CACHE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[Stage G DP-MSA] ERROR: required input missing: ${required_file}" >&2
    find "${ANAPRIOR_OUTPUT_BASE}" -maxdepth 4 -type f \( -name "*dp_msa*.pt" -o -name "dp_msa_adapter.pt" \) 2>/dev/null | sort | head -50 >&2 || true
    exit 1
  fi
  echo "[Stage G DP-MSA] found: ${required_file}"
done

echo "[Stage G DP-MSA] Training residual disease-phrase spatial adapter"
python -m anaprior.train.train_dp_msa_adapter \
  --train-cache "${DP_MSA_TRAIN_CACHE}" \
  --valid-cache "${DP_MSA_VALID_CACHE}" \
  --outdir "${OUTROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --hidden-channels "${HIDDEN_CHANNELS}" \
  --embedding-dim "${EMBEDDING_DIM}" \
  --lambda-weight "${LAMBDA_WEIGHT}" \
  --residual-l1-weight "${RESIDUAL_L1_WEIGHT}" \
  --seed "${SEED}" \
  --device "${DEVICE}"

echo "[Stage G DP-MSA] checkpoint: ${OUTROOT}/dp_msa_adapter.pt"
echo "[Stage G DP-MSA] report: ${OUTROOT}/train_report.json"
