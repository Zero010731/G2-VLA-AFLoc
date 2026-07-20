#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
START_STAGE="${START_STAGE:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
DEVICE="${DEVICE:-cuda}"
BASE="${ANAPRIOR_OUTPUT_BASE:-/mnt3/zhangran/anaprior_outputs}"
OUTROOT="${OUTROOT:-${BASE}/anaprior_stage_m_afloc_mrsg_phase2_run1}"
CACHE_ROOT="${CACHE_ROOT:-${BASE}/anaprior_stage_j_afloc_mrsg_box_free_full_run3/cache}"
PHASE1_CHECKPOINT="${PHASE1_CHECKPOINT:-${BASE}/anaprior_stage_l_afloc_anchor_preserving_phase1_run1/phase1_grounding/mrsg_phase_b.pt}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-${CACHE_ROOT}/train_mrsg.jsonl}"
VALID_MANIFEST="${VALID_MANIFEST:-${CACHE_ROOT}/valid_mrsg.jsonl}"
PROTOCOL_MANIFEST="${PROTOCOL_MANIFEST:-${CACHE_ROOT}/mrsg_protocol_manifest.json}"
DESCRIPTIONS_JSON="${DESCRIPTIONS_JSON:-${ROOT}/anaprior/configs/mrsg_disease_descriptions.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/mimic-cxr/jpg}"
AFLOC_CHECKPOINT="${AFLOC_CHECKPOINT:-/mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt}"
AFLOC_BERT_TYPE="${AFLOC_BERT_TYPE:-/mnt/zhangran/Bio_ClinicalBERT}"

EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
MAX_VALID_STEPS="${MAX_VALID_STEPS:-200}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PHASE2A_RESUME_CHECKPOINT="${PHASE2A_RESUME_CHECKPOINT:-}"
PHASE2B_RESUME_CHECKPOINT="${PHASE2B_RESUME_CHECKPOINT:-}"
PHASE2C_RESUME_CHECKPOINT="${PHASE2C_RESUME_CHECKPOINT:-}"

A_DIR="${OUTROOT}/phase2a_stability"
B_DIR="${OUTROOT}/phase2b_cross_correction"
C_DIR="${OUTROOT}/phase2c_phrase_swap"
A_CKPT="${A_DIR}/mrsg_phase_b.pt"
B_CKPT="${B_DIR}/mrsg_phase_b.pt"
C_CKPT="${C_DIR}/mrsg_phase_b.pt"

export AFLOC_BERT_TYPE AFLOC_HF_LOCAL_FILES_ONLY=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

req_file() { [[ -f "$2" ]] || { echo "[Phase 2] missing $1: $2" >&2; exit 1; }; }
req_dir() { [[ -d "$2" ]] || { echo "[Phase 2] missing $1: $2" >&2; exit 1; }; }
enabled() { (( START_STAGE <= $1 )); }

if [[ ! "${START_STAGE}" =~ ^[0-3]$ ]]; then
  echo "[Phase 2] START_STAGE must be 0, 1, 2, or 3" >&2
  exit 1
fi

for path in "${TRAIN_MANIFEST}" "${VALID_MANIFEST}" "${PROTOCOL_MANIFEST}" \
  "${DESCRIPTIONS_JSON}" "${PHASE1_CHECKPOINT}" "${AFLOC_CHECKPOINT}"; do
  req_file input "${path}"
done
req_dir images "${IMAGE_ROOT}"
req_dir bert "${AFLOC_BERT_TYPE}"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "[Phase 2] preflight complete"
  exit 0
fi

resume_args() {
  local path="$1"
  if [[ -n "${path}" ]]; then
    req_file resume "${path}"
    printf '%s\n' "--resume-checkpoint" "${path}"
  fi
}

check_gate() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
print("gate:", r["phase_gate"])
for k in ("raw_residual_max_abs", "residual_saturation_ratio", "correction_abs_mean", "final_anchor_pearson", "residual_stability", "cross_view_patch", "phrase_swap_contrast"):
    print(f"{k}: {r['diagnostics'].get(k)}")
if not r["phase_gate"]["passed"]:
    raise SystemExit("phase diagnostic gate failed")
PY
}

COMMON=(--phase grounding --train-manifest "${TRAIN_MANIFEST}" --valid-manifest "${VALID_MANIFEST}"
  --afloc-checkpoint "${AFLOC_CHECKPOINT}" --protocol-manifest "${PROTOCOL_MANIFEST}"
  --descriptions-json "${DESCRIPTIONS_JSON}" --image-root "${IMAGE_ROOT}"
  --feature-dim 256 --num-heads 8 --focal-slots 4 --topk-fraction 0.15
  --residual-logit-bound 0.5 --residual-logit-cap 2.0 --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}" --max-train-steps "${MAX_TRAIN_STEPS}"
  --max-valid-steps "${MAX_VALID_STEPS}" --num-workers "${NUM_WORKERS}"
  --log-every-steps 50 --learning-rate 5e-5 --w-ground 1.0 --w-mask 0.5
  --w-query 0.25 --seed 13 --device "${DEVICE}")

mkdir -p "${A_DIR}" "${B_DIR}" "${C_DIR}"

if enabled 1; then
  mapfile -t RESUME < <(resume_args "${PHASE2A_RESUME_CHECKPOINT}")
  "${PYTHON_BIN}" -m anaprior.train.train_afloc_mrsg "${COMMON[@]}" \
    --outdir "${A_DIR}" --initial-checkpoint "${PHASE1_CHECKPOINT}" \
    --w-residual-stability 0.01 --w-cross-correction 0.0 --w-phrase-swap 0.0 "${RESUME[@]}"
  check_gate "${A_DIR}/train_report.json"
fi

if enabled 2; then
  req_file phase2a "${A_CKPT}"
  mapfile -t RESUME < <(resume_args "${PHASE2B_RESUME_CHECKPOINT}")
  "${PYTHON_BIN}" -m anaprior.train.train_afloc_mrsg "${COMMON[@]}" \
    --outdir "${B_DIR}" --initial-checkpoint "${A_CKPT}" \
    --w-residual-stability 0.01 --w-cross-correction 0.25 --w-phrase-swap 0.0 "${RESUME[@]}"
  check_gate "${B_DIR}/train_report.json"
fi

if enabled 3; then
  req_file phase2b "${B_CKPT}"
  mapfile -t RESUME < <(resume_args "${PHASE2C_RESUME_CHECKPOINT}")
  "${PYTHON_BIN}" -m anaprior.train.train_afloc_mrsg "${COMMON[@]}" \
    --outdir "${C_DIR}" --initial-checkpoint "${B_CKPT}" \
    --w-residual-stability 0.01 --w-cross-correction 0.25 --w-phrase-swap 0.10 "${RESUME[@]}"
  check_gate "${C_DIR}/train_report.json"
fi

echo "[Phase 2] complete checkpoint: ${C_CKPT}"
