#!/usr/bin/env bash
set -euo pipefail

# End-to-end AFLoc-MRSG production runner.
#
# Full run:
#   bash scripts/run_afloc_mrsg_full_server.sh
#
# Preflight only:
#   PREFLIGHT_ONLY=1 bash scripts/run_afloc_mrsg_full_server.sh
#
# Resume from a later stage:
#   START_STAGE=7 bash scripts/run_afloc_mrsg_full_server.sh
#
# Preview commands without running them:
#   DRY_RUN=1 bash scripts/run_afloc_mrsg_full_server.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export AFLOC_TMPDIR="${AFLOC_TMPDIR:-/mnt3/zhangran/tmp}"
export TMPDIR="${AFLOC_TMPDIR}"
export TEMP="${AFLOC_TMPDIR}"
export TMP="${AFLOC_TMPDIR}"
export AFLOC_HF_LOCAL_FILES_ONLY="${AFLOC_HF_LOCAL_FILES_ONLY:-1}"
mkdir -p "${AFLOC_TMPDIR}"
if [[ -z "${AFLOC_BERT_TYPE:-}" && -d "/mnt/zhangran/Bio_ClinicalBERT" ]]; then
  export AFLOC_BERT_TYPE="/mnt/zhangran/Bio_ClinicalBERT"
fi

START_STAGE="${START_STAGE:-0}"
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
ALLOW_DIRTY_OUTROOT="${ALLOW_DIRTY_OUTROOT:-0}"
AUTO_RESUME_LATEST="${AUTO_RESUME_LATEST:-1}"

ANAPRIOR_OUTPUT_BASE="${ANAPRIOR_OUTPUT_BASE:-/mnt3/zhangran/anaprior_outputs}"
RUN_NAME="${RUN_NAME:-afloc_mrsg_box_free_full}"
OUTROOT="${OUTROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_j_${RUN_NAME}}"

CACHE_ROOT="${CACHE_ROOT:-${OUTROOT}/cache}"
PHASE_A_ROOT="${PHASE_A_ROOT:-${OUTROOT}/phase_a}"
PHASE_B_ROOT="${PHASE_B_ROOT:-${OUTROOT}/phase_b}"
PHASE_C_ROOT="${PHASE_C_ROOT:-${OUTROOT}/phase_c}"
MSCXR_EVAL_ROOT="${MSCXR_EVAL_ROOT:-${OUTROOT}/mscxr_eval}"
SCORE_HMAP_ROOT="${SCORE_HMAP_ROOT:-${OUTROOT}/score_hmaps}"
METRIC_ROOT="${METRIC_ROOT:-${OUTROOT}/learned_repair_metrics}"
CHEXLOCALIZE_EVAL_ROOT="${CHEXLOCALIZE_EVAL_ROOT:-${OUTROOT}/chexlocalize_eval}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${OUTROOT}/bundle}"
REPORT_MD="${REPORT_MD:-${BUNDLE_ROOT}/afloc_mrsg_result_report.md}"
BUNDLE_JSON="${BUNDLE_JSON:-${BUNDLE_ROOT}/bundle_manifest.json}"
FROZEN_MANIFEST="${FROZEN_MANIFEST:-${OUTROOT}/frozen_experiment_manifest.json}"

MIMIC_CSV="${MIMIC_CSV:-}"
MSCXR_EXCLUSION_JSON="${MSCXR_EXCLUSION_JSON:-/mnt/mimic-cxr/ms-cxr_1.1.0/MS_CXR_Local_Alignment_v1.1.0.json}"
AFLOC_CHECKPOINT="${AFLOC_CHECKPOINT:-/mnt/zhangran/AFLoc_weight_path/pretrained/Pretrained_CXR.ckpt}"
MIMIC_IMAGE_ROOT="${MIMIC_IMAGE_ROOT:-/mnt/mimic-cxr/jpg}"
DESCRIPTIONS_JSON="${DESCRIPTIONS_JSON:-${REPO_ROOT}/anaprior/configs/mrsg_disease_descriptions.json}"
REFERENCE_HMAPS_ROOT="${REFERENCE_HMAPS_ROOT:-${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_c_8class_phrase_validation_gate_alias_v2_dcem_v3/learned_repair_hmaps}"

LOCALIZATION_MS_CXR_JSON="${LOCALIZATION_MS_CXR_JSON:-/mnt/mimic-cxr/ms-cxr_1.1.0/MS_CXR_Local_Alignment_v1.1.0.json}"
LOCALIZATION_MIMIC_IMG_DIR="${LOCALIZATION_MIMIC_IMG_DIR:-/mnt/mimic-cxr/jpg}"
CHEXLOCALIZE_TEST_JSON="${CHEXLOCALIZE_TEST_JSON:-/mnt/siat225_disk1/yh/datasets/hwj/Foundation-DATA-HWJ/chexlocalize/CheXlocalize/gt_segmentations_test.json}"
CHEXLOCALIZE_TEST_IMG_DIR="${CHEXLOCALIZE_TEST_IMG_DIR:-/mnt/siat225_disk1/yh/datasets/hwj/Foundation-DATA-HWJ/chexlocalize/CheXpert/test}"

BASELINE_METHOD="${BASELINE_METHOD:-baseline}"
DCEM_METHOD="${DCEM_METHOD:-phrase_anatomy_dcem}"
METHOD_NAME="${METHOD_NAME:-afloc_mrsg}"
SCORE_METHODS="${SCORE_METHODS:-${BASELINE_METHOD},${DCEM_METHOD},${METHOD_NAME}}"
SCORE_DATASET="${SCORE_DATASET:-MS_CXR_CLS}"
FINDINGS="${FINDINGS:-Atelectasis,Cardiomegaly,Consolidation,Edema,Lung Opacity,Pleural Effusion,Pneumonia,Pneumothorax}"
VALID_FRACTION="${VALID_FRACTION:-0.2}"
SEED="${SEED:-13}"
DEVICE="${DEVICE:-cuda}"
MAX_CASES="${MAX_CASES:-}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-1000}"
SCORE_VAL_FRACTION="${SCORE_VAL_FRACTION:-0.3}"
CANDIDATE_EFFECT_FLOOR="${CANDIDATE_EFFECT_FLOOR:-0.0}"
MACRO_ALL_HARM_FLOOR="${MACRO_ALL_HARM_FLOOR:--0.005}"

FEATURE_DIM="${FEATURE_DIM:-256}"
TEXT_DIM="${TEXT_DIM:-}"
NUM_HEADS="${NUM_HEADS:-8}"
FOCAL_SLOTS="${FOCAL_SLOTS:-4}"
TOPK_FRACTION="${TOPK_FRACTION:-0.15}"
ROUTE_TEMPERATURE="${ROUTE_TEMPERATURE:-1.0}"
PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-5}"
PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-10}"
PHASE_C_EPOCHS="${PHASE_C_EPOCHS:-10}"
PHASE_A_BATCH_SIZE="${PHASE_A_BATCH_SIZE:-8}"
PHASE_B_BATCH_SIZE="${PHASE_B_BATCH_SIZE:-8}"
PHASE_C_BATCH_SIZE="${PHASE_C_BATCH_SIZE:-8}"
PHASE_A_LEARNING_RATE="${PHASE_A_LEARNING_RATE:-5e-4}"
PHASE_B_LEARNING_RATE="${PHASE_B_LEARNING_RATE:-1e-4}"
PHASE_C_LEARNING_RATE="${PHASE_C_LEARNING_RATE:-1e-4}"
PHASE_C_TEACHER_DECAY="${PHASE_C_TEACHER_DECAY:-0.99}"
W_GROUND="${W_GROUND:-1.0}"
W_TEACHER="${W_TEACHER:-1.0}"
W_MASK="${W_MASK:-1.0}"
W_QUERY="${W_QUERY:-1.0}"

CACHE_REPORT="${CACHE_ROOT}/mrsg_image_report_cache_report.json"
TRAIN_MANIFEST="${CACHE_ROOT}/train_mrsg.jsonl"
VALID_MANIFEST="${CACHE_ROOT}/valid_mrsg.jsonl"
PROTOCOL_MANIFEST="${CACHE_ROOT}/mrsg_protocol_manifest.json"
PHASE_A_REPORT="${PHASE_A_ROOT}/train_report.json"
PHASE_B_REPORT="${PHASE_B_ROOT}/train_report.json"
PHASE_C_REPORT="${PHASE_C_ROOT}/train_report.json"
PHASE_A_BEST="${PHASE_A_ROOT}/mrsg_phase_a.pt"
PHASE_B_BEST="${PHASE_B_ROOT}/mrsg_phase_b.pt"
PHASE_C_BEST="${PHASE_C_ROOT}/mrsg_phase_c.pt"
PHASE_A_LATEST="${PHASE_A_ROOT}/mrsg_phase_a_latest.pt"
PHASE_B_LATEST="${PHASE_B_ROOT}/mrsg_phase_b_latest.pt"
PHASE_C_LATEST="${PHASE_C_ROOT}/mrsg_phase_c_latest.pt"
MSCXR_HMAPS="${MSCXR_EVAL_ROOT}/${METHOD_NAME}/hmaps.npy"
MSCXR_SUMMARY_JSON="${MSCXR_EVAL_ROOT}/mrsg_eval_summary.json"
MSCXR_CASE_DIAGNOSTICS_JSON="${MSCXR_EVAL_ROOT}/mrsg_case_diagnostics.json"
METRIC_SUMMARY_JSON="${METRIC_ROOT}/learned_repair_metrics_summary.json"
METRIC_DECISION_JSON="${METRIC_ROOT}/learned_repair_decision.json"
CHEXLOCALIZE_HMAPS="${CHEXLOCALIZE_EVAL_ROOT}/${METHOD_NAME}/hmaps.npy"
CHEXLOCALIZE_SUMMARY_JSON="${CHEXLOCALIZE_EVAL_ROOT}/mrsg_eval_summary.json"
CHEXLOCALIZE_CASE_DIAGNOSTICS_JSON="${CHEXLOCALIZE_EVAL_ROOT}/mrsg_case_diagnostics.json"

MAX_CASES_ARGS=()
MAX_CASES_JSON="null"
if [[ -n "${MAX_CASES}" ]]; then
  MAX_CASES_ARGS=(--max-cases "${MAX_CASES}")
  MAX_CASES_JSON="${MAX_CASES}"
fi

TEXT_DIM_ARGS=()
TEXT_DIM_JSON="null"
if [[ -n "${TEXT_DIM}" ]]; then
  TEXT_DIM_ARGS=(--text-dim "${TEXT_DIM}")
  TEXT_DIM_JSON="${TEXT_DIM}"
fi

export AFLOC_CHECKPOINT
export PHASE_C_ROOT
export MSCXR_EVAL_ROOT
export CHEXLOCALIZE_EVAL_ROOT
export DEVICE
export METHOD_NAME
export MAX_CASES_JSON
export SCORE_HMAP_ROOT
export METRIC_ROOT
export SCORE_METHODS
export FINDINGS
export SCORE_DATASET
export SCORE_VAL_FRACTION
export BOOTSTRAP_REPLICATES
export SEED
export CANDIDATE_EFFECT_FLOOR
export MACRO_ALL_HARM_FLOOR
export DESCRIPTIONS_JSON
export FEATURE_DIM
export TEXT_DIM_JSON
export NUM_HEADS
export FOCAL_SLOTS
export TOPK_FRACTION
export ROUTE_TEMPERATURE
export PHASE_A_EPOCHS
export PHASE_B_EPOCHS
export PHASE_C_EPOCHS
export PHASE_A_BATCH_SIZE
export PHASE_B_BATCH_SIZE
export PHASE_C_BATCH_SIZE
export PHASE_A_LEARNING_RATE
export PHASE_B_LEARNING_RATE
export PHASE_C_LEARNING_RATE
export PHASE_C_TEACHER_DECAY
export W_GROUND
export W_TEACHER
export W_MASK
export W_QUERY

PHASE_A_RESUME_ARGS=()
PHASE_B_RESUME_ARGS=()
PHASE_C_RESUME_ARGS=()
if [[ "${AUTO_RESUME_LATEST}" == "1" && -f "${PHASE_A_LATEST}" ]]; then
  PHASE_A_RESUME_ARGS=(--resume-checkpoint "${PHASE_A_LATEST}")
fi
if [[ "${AUTO_RESUME_LATEST}" == "1" && -f "${PHASE_B_LATEST}" ]]; then
  PHASE_B_RESUME_ARGS=(--resume-checkpoint "${PHASE_B_LATEST}")
fi
if [[ "${AUTO_RESUME_LATEST}" == "1" && -f "${PHASE_C_LATEST}" ]]; then
  PHASE_C_RESUME_ARGS=(--resume-checkpoint "${PHASE_C_LATEST}")
fi

MSCXR_EVAL_ARGS_JSON="$(python - <<'PY'
import json
import os

payload = {
    "dataset": "MS_CXR",
    "split": "test",
    "afloc_checkpoint": os.environ["AFLOC_CHECKPOINT"],
    "checkpoint": os.path.join(os.environ["PHASE_C_ROOT"], "mrsg_phase_c.pt"),
    "outdir": os.environ["MSCXR_EVAL_ROOT"],
    "device": os.environ["DEVICE"],
    "method_name": os.environ["METHOD_NAME"],
    "max_cases": None if os.environ["MAX_CASES_JSON"] == "null" else int(os.environ["MAX_CASES_JSON"]),
}
print(json.dumps(payload, sort_keys=True))
PY
)"

CHEXLOCALIZE_EVAL_ARGS_JSON="$(python - <<'PY'
import json
import os

payload = {
    "dataset": "CHEXLOCALIZE",
    "split": "test",
    "afloc_checkpoint": os.environ["AFLOC_CHECKPOINT"],
    "checkpoint": os.path.join(os.environ["PHASE_C_ROOT"], "mrsg_phase_c.pt"),
    "outdir": os.environ["CHEXLOCALIZE_EVAL_ROOT"],
    "device": os.environ["DEVICE"],
    "method_name": os.environ["METHOD_NAME"],
    "max_cases": None if os.environ["MAX_CASES_JSON"] == "null" else int(os.environ["MAX_CASES_JSON"]),
}
print(json.dumps(payload, sort_keys=True))
PY
)"

SCORE_ARGS_JSON="$(python - <<'PY'
import json
import os

payload = {
    "hmaps_root": os.environ["SCORE_HMAP_ROOT"],
    "outdir": os.environ["METRIC_ROOT"],
    "methods": os.environ["SCORE_METHODS"].split(","),
    "candidate_categories": os.environ["FINDINGS"].split(","),
    "dataset": os.environ["SCORE_DATASET"],
    "val_fraction": float(os.environ["SCORE_VAL_FRACTION"]),
    "bootstrap_replicates": int(os.environ["BOOTSTRAP_REPLICATES"]),
    "seed": int(os.environ["SEED"]),
    "margin": True,
    "max_cases": None if os.environ["MAX_CASES_JSON"] == "null" else int(os.environ["MAX_CASES_JSON"]),
    "candidate_effect_floor": float(os.environ["CANDIDATE_EFFECT_FLOOR"]),
    "macro_all_harm_floor": float(os.environ["MACRO_ALL_HARM_FLOOR"]),
}
print(json.dumps(payload, sort_keys=True))
PY
)"
export MSCXR_EVAL_ARGS_JSON
export CHEXLOCALIZE_EVAL_ARGS_JSON
export SCORE_ARGS_JSON

require_nonempty_env() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    echo "[AFLoc-MRSG] ERROR: required environment variable ${name} is empty" >&2
    exit 1
  fi
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[AFLoc-MRSG] ERROR: missing required file for ${label}: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[AFLoc-MRSG] ERROR: missing required directory for ${label}: ${path}" >&2
    exit 1
  fi
}

print_cmd() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

run_cmd() {
  local stage="$1"
  local label="$2"
  shift 2
  echo "[${stage}/10] ${label}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_cmd "$@"
    return 0
  fi
  "$@"
}

stage_enabled() {
  local stage="$1"
  (( START_STAGE <= stage ))
}

ensure_clean_outroot() {
  if (( START_STAGE != 0 )); then
    return 0
  fi
  if [[ ! -d "${OUTROOT}" ]]; then
    return 0
  fi
  if [[ "${ALLOW_DIRTY_OUTROOT}" == "1" ]]; then
    return 0
  fi
  if find "${OUTROOT}" -mindepth 1 -print -quit | grep -q .; then
    echo "[AFLoc-MRSG] ERROR: OUTROOT already exists and is not empty: ${OUTROOT}" >&2
    echo "[AFLoc-MRSG] Use a fresh OUTROOT or set ALLOW_DIRTY_OUTROOT=1 for an intentional rerun." >&2
    exit 1
  fi
}

check_phase_gate_report() {
  local report_path="$1"
  local expected_phase="$2"
  local expected_checkpoint="$3"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[AFLoc-MRSG] DRY_RUN would verify ${expected_phase} gate via ${report_path}"
    return 0
  fi
  python - "${report_path}" "${expected_phase}" "${expected_checkpoint}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
expected_phase = sys.argv[2]
expected_checkpoint = Path(sys.argv[3]).resolve()

if not report_path.exists():
    raise SystemExit(f"missing train report: {report_path}")
payload = json.loads(report_path.read_text(encoding="utf-8"))
if payload.get("phase") != expected_phase:
    raise SystemExit(f"{report_path} phase mismatch: expected {expected_phase}, got {payload.get('phase')}")
gate = payload.get("phase_gate") or {}
if not bool(gate.get("passed")):
    raise SystemExit(f"{report_path} phase_gate.passed is false")
checkpoint = Path(payload.get("checkpoint", "")).resolve()
if checkpoint != expected_checkpoint:
    raise SystemExit(
        f"{report_path} selected checkpoint mismatch: expected {expected_checkpoint}, got {checkpoint}"
    )
if not checkpoint.exists():
    raise SystemExit(f"selected checkpoint does not exist: {checkpoint}")
PY
}

write_frozen_manifest() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[AFLoc-MRSG] DRY_RUN would write ${FROZEN_MANIFEST}"
    return 0
  fi
  python - "${FROZEN_MANIFEST}" "${CACHE_REPORT}" "${PROTOCOL_MANIFEST}" "${PHASE_A_REPORT}" "${PHASE_B_REPORT}" "${PHASE_C_REPORT}" "${AFLOC_CHECKPOINT}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest_path = Path(sys.argv[1])
cache_report_path = Path(sys.argv[2])
protocol_manifest_path = Path(sys.argv[3])
phase_a_report_path = Path(sys.argv[4])
phase_b_report_path = Path(sys.argv[5])
phase_c_report_path = Path(sys.argv[6])
afloc_checkpoint = Path(sys.argv[7])

cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
phase_a = json.loads(phase_a_report_path.read_text(encoding="utf-8"))
phase_b = json.loads(phase_b_report_path.read_text(encoding="utf-8"))
phase_c = json.loads(phase_c_report_path.read_text(encoding="utf-8"))

phase_reports = {
    "phase_a": phase_a,
    "phase_b": phase_b,
    "phase_c": phase_c,
}

for label, payload in phase_reports.items():
    gate = payload.get("phase_gate") or {}
    if not bool(gate.get("passed")):
        raise SystemExit(f"{label} cannot be frozen because phase_gate.passed is false")

payload = {
    "status": "frozen",
    "git_commit": str(phase_c["git_commit"]),
    "cache_report_path": str(cache_report_path),
    "cache_report_sha256": sha256_file(cache_report_path),
    "train_manifest_path": str(cache_report["train_jsonl"]),
    "train_manifest_sha256": sha256_file(Path(cache_report["train_jsonl"])),
    "valid_manifest_path": str(cache_report["valid_jsonl"]),
    "valid_manifest_sha256": sha256_file(Path(cache_report["valid_jsonl"])),
    "protocol_manifest_path": str(protocol_manifest_path),
    "data_protocol_sha256": sha256_file(protocol_manifest_path),
    "description_file_sha256": str(phase_c["description_file_sha256"]),
    "descriptions_json_path": str(os.environ["DESCRIPTIONS_JSON"]),
    "afloc_checkpoint_path": str(afloc_checkpoint),
    "afloc_checkpoint_sha256": sha256_file(afloc_checkpoint),
    "model_config": phase_c["model_config"],
    "four_top_level_loss_weights": phase_c["four_top_level_loss_weights"],
    "trainer_settings": {
        "feature_dim": int(os.environ["FEATURE_DIM"]),
        "text_dim": None if os.environ["TEXT_DIM_JSON"] == "null" else int(os.environ["TEXT_DIM_JSON"]),
        "num_heads": int(os.environ["NUM_HEADS"]),
        "focal_slots": int(os.environ["FOCAL_SLOTS"]),
        "topk_fraction": float(os.environ["TOPK_FRACTION"]),
        "route_temperature": float(os.environ["ROUTE_TEMPERATURE"]),
        "phase_a_epochs": int(os.environ["PHASE_A_EPOCHS"]),
        "phase_b_epochs": int(os.environ["PHASE_B_EPOCHS"]),
        "phase_c_epochs": int(os.environ["PHASE_C_EPOCHS"]),
        "phase_a_batch_size": int(os.environ["PHASE_A_BATCH_SIZE"]),
        "phase_b_batch_size": int(os.environ["PHASE_B_BATCH_SIZE"]),
        "phase_c_batch_size": int(os.environ["PHASE_C_BATCH_SIZE"]),
        "phase_a_learning_rate": float(os.environ["PHASE_A_LEARNING_RATE"]),
        "phase_b_learning_rate": float(os.environ["PHASE_B_LEARNING_RATE"]),
        "phase_c_learning_rate": float(os.environ["PHASE_C_LEARNING_RATE"]),
        "phase_c_teacher_decay": float(os.environ["PHASE_C_TEACHER_DECAY"]),
        "w_ground": float(os.environ["W_GROUND"]),
        "w_teacher": float(os.environ["W_TEACHER"]),
        "w_mask": float(os.environ["W_MASK"]),
        "w_query": float(os.environ["W_QUERY"]),
        "seed": int(os.environ["SEED"]),
        "device": os.environ["DEVICE"],
    },
    "phase_artifacts": {},
    "MSCXR_EVAL_ARGS_JSON": json.loads(os.environ["MSCXR_EVAL_ARGS_JSON"]),
    "CHEXLOCALIZE_EVAL_ARGS_JSON": json.loads(os.environ["CHEXLOCALIZE_EVAL_ARGS_JSON"]),
    "SCORE_ARGS_JSON": json.loads(os.environ["SCORE_ARGS_JSON"]),
    "test_evaluated": False,
    "output_hashes": {},
}

for name, report in phase_reports.items():
    report_path = Path(report["report"])
    checkpoint_path = Path(report["checkpoint"])
    latest_path = Path(report["latest_checkpoint"])
    payload["phase_artifacts"][name] = {
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "latest_checkpoint_path": str(latest_path),
        "latest_checkpoint_sha256": sha256_file(latest_path),
        "phase_gate": report["phase_gate"],
        "best_valid_loss": float(report["best_valid_loss"]),
    }

immutable_expected = {
    key: value
    for key, value in payload.items()
    if key not in {"test_evaluated", "output_hashes"}
}

if manifest_path.exists():
    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    immutable_current = {
        key: value
        for key, value in current.items()
        if key not in {"test_evaluated", "output_hashes"}
    }
    if immutable_current != immutable_expected:
        raise SystemExit("existing frozen manifest does not match current training/evaluation settings")
    payload["test_evaluated"] = bool(current.get("test_evaluated", False))
    payload["output_hashes"] = dict(current.get("output_hashes", {}))

manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

validate_frozen_manifest() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[AFLoc-MRSG] DRY_RUN would validate ${FROZEN_MANIFEST}"
    return 0
  fi
  if [[ ! -f "${FROZEN_MANIFEST}" ]]; then
    echo "[AFLoc-MRSG] ERROR: frozen manifest is required for START_STAGE=${START_STAGE}: ${FROZEN_MANIFEST}" >&2
    exit 1
  fi
  write_frozen_manifest
}

update_frozen_manifest_outputs() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[AFLoc-MRSG] DRY_RUN would update output hashes in ${FROZEN_MANIFEST}"
    return 0
  fi
  python - "${FROZEN_MANIFEST}" "$@" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest_path = Path(sys.argv[1])
if not manifest_path.exists():
    raise SystemExit(f"missing frozen manifest: {manifest_path}")
if (len(sys.argv) - 2) % 2 != 0:
    raise SystemExit("output hash updates must be provided as name/path pairs")

payload = json.loads(manifest_path.read_text(encoding="utf-8"))
immutable_before = {
    key: value
    for key, value in payload.items()
    if key not in {"test_evaluated", "output_hashes"}
}
output_hashes = dict(payload.get("output_hashes", {}))

for idx in range(2, len(sys.argv), 2):
    name = sys.argv[idx]
    path = Path(sys.argv[idx + 1])
    if not path.exists():
        raise SystemExit(f"cannot hash missing output: {path}")
    output_hashes[name] = {
        "path": str(path),
        "sha256": sha256_file(path),
    }

payload["test_evaluated"] = True
payload["output_hashes"] = output_hashes

immutable_after = {
    key: value
    for key, value in payload.items()
    if key not in {"test_evaluated", "output_hashes"}
}
if immutable_before != immutable_after:
    raise SystemExit("frozen manifest update attempted to modify immutable experiment settings")

manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

install_reference_hmap() {
  local method="$1"
  local source="${REFERENCE_HMAPS_ROOT}/${method}/hmaps.npy"
  local target_dir="${SCORE_HMAP_ROOT}/${method}"
  require_file "${method}_reference_hmap" "${source}"
  mkdir -p "${target_dir}"
  run_cmd 8 "install ${method} reference heatmaps" cp "${source}" "${target_dir}/hmaps.npy"
}

echo "[AFLoc-MRSG] repo root: ${REPO_ROOT}"
echo "[AFLoc-MRSG] output root: ${OUTROOT}"
echo "[AFLoc-MRSG] START_STAGE=${START_STAGE}"
echo "[AFLoc-MRSG] DRY_RUN=${DRY_RUN}"
echo "[AFLoc-MRSG] PREFLIGHT_ONLY=${PREFLIGHT_ONLY}"
echo "[AFLoc-MRSG] AFLOC_HF_LOCAL_FILES_ONLY=${AFLOC_HF_LOCAL_FILES_ONLY}"
echo "[AFLoc-MRSG] AFLOC_BERT_TYPE=${AFLOC_BERT_TYPE:-<checkpoint default>}"

if [[ ! "${START_STAGE}" =~ ^[0-9]+$ ]]; then
  echo "[AFLoc-MRSG] ERROR: START_STAGE must be an integer from 0 to 10" >&2
  exit 1
fi
if (( START_STAGE < 0 || START_STAGE > 10 )); then
  echo "[AFLoc-MRSG] ERROR: START_STAGE must be between 0 and 10" >&2
  exit 1
fi

require_nonempty_env "MIMIC_CSV" "${MIMIC_CSV}"
require_file "MIMIC_CSV" "${MIMIC_CSV}"
require_file "MSCXR_EXCLUSION_JSON" "${MSCXR_EXCLUSION_JSON}"
require_file "AFLOC_CHECKPOINT" "${AFLOC_CHECKPOINT}"
require_dir "MIMIC_IMAGE_ROOT" "${MIMIC_IMAGE_ROOT}"
require_file "DESCRIPTIONS_JSON" "${DESCRIPTIONS_JSON}"
require_dir "REFERENCE_HMAPS_ROOT" "${REFERENCE_HMAPS_ROOT}"
require_file "LOCALIZATION_MS_CXR_JSON" "${LOCALIZATION_MS_CXR_JSON}"
require_dir "LOCALIZATION_MIMIC_IMG_DIR" "${LOCALIZATION_MIMIC_IMG_DIR}"
require_file "CHEXLOCALIZE_TEST_JSON" "${CHEXLOCALIZE_TEST_JSON}"
require_dir "CHEXLOCALIZE_TEST_IMG_DIR" "${CHEXLOCALIZE_TEST_IMG_DIR}"
require_file "REFERENCE_BASELINE_HMAP" "${REFERENCE_HMAPS_ROOT}/${BASELINE_METHOD}/hmaps.npy"
require_file "REFERENCE_DCEM_HMAP" "${REFERENCE_HMAPS_ROOT}/${DCEM_METHOD}/hmaps.npy"

ensure_clean_outroot
mkdir -p "${CACHE_ROOT}" "${PHASE_A_ROOT}" "${PHASE_B_ROOT}" "${PHASE_C_ROOT}" "${MSCXR_EVAL_ROOT}" "${SCORE_HMAP_ROOT}" "${METRIC_ROOT}" "${CHEXLOCALIZE_EVAL_ROOT}" "${BUNDLE_ROOT}"

if (( START_STAGE >= 7 )); then
  validate_frozen_manifest
fi

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "[AFLoc-MRSG] preflight complete"
  exit 0
fi

# [0/10] protocol manifest and MS-CXR exclusion audit
if stage_enabled 0; then
  run_cmd 0 "protocol manifest and MS-CXR exclusion audit" \
    python -m anaprior.train.build_mrsg_image_report_cache \
    --mimic-csv "${MIMIC_CSV}" \
    --mscxr-json "${MSCXR_EXCLUSION_JSON}" \
    --descriptions-json "${DESCRIPTIONS_JSON}" \
    --outdir "${CACHE_ROOT}" \
    --valid-fraction "${VALID_FRACTION}" \
    --seed "${SEED}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    python - "${CACHE_REPORT}" "${PROTOCOL_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

cache_report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
protocol = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if bool(cache_report.get("uses_spatial_annotations")):
    raise SystemExit("cache report unexpectedly declares spatial supervision")
if bool(cache_report.get("uses_dcem")):
    raise SystemExit("cache report unexpectedly declares DCEM usage")
sanity = protocol.get("sanity") or {}
if int(sanity.get("mscxr_overlap", 1)) != 0:
    raise SystemExit("protocol manifest reports MS-CXR overlap")
if int(sanity.get("train_valid_subject_overlap", 1)) != 0:
    raise SystemExit("protocol manifest reports train/valid overlap")
PY
  fi
else
  echo "[0/10] skipping protocol manifest and MS-CXR exclusion audit"
fi

# [1/10] Phase A locality warm-up
if stage_enabled 1; then
  run_cmd 1 "Phase A locality warm-up" \
    python -m anaprior.train.train_afloc_mrsg \
    --phase "locality" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --valid-manifest "${VALID_MANIFEST}" \
    --outdir "${PHASE_A_ROOT}" \
    --afloc-checkpoint "${AFLOC_CHECKPOINT}" \
    --protocol-manifest "${PROTOCOL_MANIFEST}" \
    --descriptions-json "${DESCRIPTIONS_JSON}" \
    --image-root "${MIMIC_IMAGE_ROOT}" \
    --feature-dim "${FEATURE_DIM}" \
    "${TEXT_DIM_ARGS[@]}" \
    --num-heads "${NUM_HEADS}" \
    --focal-slots "${FOCAL_SLOTS}" \
    --topk-fraction "${TOPK_FRACTION}" \
    --route-temperature "${ROUTE_TEMPERATURE}" \
    --epochs "${PHASE_A_EPOCHS}" \
    --batch-size "${PHASE_A_BATCH_SIZE}" \
    --learning-rate "${PHASE_A_LEARNING_RATE}" \
    --w-ground "${W_GROUND}" \
    --w-teacher "${W_TEACHER}" \
    --w-mask "${W_MASK}" \
    --w-query "${W_QUERY}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    "${PHASE_A_RESUME_ARGS[@]}"
else
  echo "[1/10] skipping Phase A locality warm-up"
fi

# [2/10] Gate A check
if stage_enabled 2; then
  echo "[2/10] Gate A check"
  check_phase_gate_report "${PHASE_A_REPORT}" "locality" "${PHASE_A_BEST}"
else
  echo "[2/10] skipping Gate A check"
fi

# [3/10] Phase B sparse grounding
if stage_enabled 3; then
  run_cmd 3 "Phase B sparse grounding" \
    python -m anaprior.train.train_afloc_mrsg \
    --phase "grounding" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --valid-manifest "${VALID_MANIFEST}" \
    --outdir "${PHASE_B_ROOT}" \
    --afloc-checkpoint "${AFLOC_CHECKPOINT}" \
    --protocol-manifest "${PROTOCOL_MANIFEST}" \
    --descriptions-json "${DESCRIPTIONS_JSON}" \
    --previous-checkpoint "${PHASE_A_BEST}" \
    --image-root "${MIMIC_IMAGE_ROOT}" \
    --feature-dim "${FEATURE_DIM}" \
    "${TEXT_DIM_ARGS[@]}" \
    --num-heads "${NUM_HEADS}" \
    --focal-slots "${FOCAL_SLOTS}" \
    --topk-fraction "${TOPK_FRACTION}" \
    --route-temperature "${ROUTE_TEMPERATURE}" \
    --epochs "${PHASE_B_EPOCHS}" \
    --batch-size "${PHASE_B_BATCH_SIZE}" \
    --learning-rate "${PHASE_B_LEARNING_RATE}" \
    --w-ground "${W_GROUND}" \
    --w-teacher "${W_TEACHER}" \
    --w-mask "${W_MASK}" \
    --w-query "${W_QUERY}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    "${PHASE_B_RESUME_ARGS[@]}"
else
  echo "[3/10] skipping Phase B sparse grounding"
fi

# [4/10] Gate B check
if stage_enabled 4; then
  echo "[4/10] Gate B check"
  check_phase_gate_report "${PHASE_B_REPORT}" "grounding" "${PHASE_B_BEST}"
else
  echo "[4/10] skipping Gate B check"
fi

# [5/10] Phase C dual consistency
if stage_enabled 5; then
  run_cmd 5 "Phase C dual consistency" \
    python -m anaprior.train.train_afloc_mrsg \
    --phase "consistency" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --valid-manifest "${VALID_MANIFEST}" \
    --outdir "${PHASE_C_ROOT}" \
    --afloc-checkpoint "${AFLOC_CHECKPOINT}" \
    --protocol-manifest "${PROTOCOL_MANIFEST}" \
    --descriptions-json "${DESCRIPTIONS_JSON}" \
    --previous-checkpoint "${PHASE_B_BEST}" \
    --image-root "${MIMIC_IMAGE_ROOT}" \
    --feature-dim "${FEATURE_DIM}" \
    "${TEXT_DIM_ARGS[@]}" \
    --num-heads "${NUM_HEADS}" \
    --focal-slots "${FOCAL_SLOTS}" \
    --topk-fraction "${TOPK_FRACTION}" \
    --route-temperature "${ROUTE_TEMPERATURE}" \
    --epochs "${PHASE_C_EPOCHS}" \
    --batch-size "${PHASE_C_BATCH_SIZE}" \
    --learning-rate "${PHASE_C_LEARNING_RATE}" \
    --teacher-decay "${PHASE_C_TEACHER_DECAY}" \
    --w-ground "${W_GROUND}" \
    --w-teacher "${W_TEACHER}" \
    --w-mask "${W_MASK}" \
    --w-query "${W_QUERY}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    "${PHASE_C_RESUME_ARGS[@]}"
else
  echo "[5/10] skipping Phase C dual consistency"
fi

# [6/10] Gate C and experiment freeze manifest
if stage_enabled 6; then
  echo "[6/10] Gate C and experiment freeze manifest"
  check_phase_gate_report "${PHASE_C_REPORT}" "consistency" "${PHASE_C_BEST}"
  write_frozen_manifest
else
  echo "[6/10] skipping Gate C and experiment freeze manifest"
fi

# [7/10] raw MS-CXR heatmaps
if stage_enabled 7; then
  validate_frozen_manifest
  run_cmd 7 "raw MS-CXR heatmaps" \
    python -m anaprior.eval.eval_mscxr_afloc_mrsg \
    --dataset "MS_CXR" \
    --split "test" \
    --afloc-checkpoint "${AFLOC_CHECKPOINT}" \
    --checkpoint "${PHASE_C_BEST}" \
    --outdir "${MSCXR_EVAL_ROOT}" \
    --device "${DEVICE}" \
    --method-name "${METHOD_NAME}" \
    "${MAX_CASES_ARGS[@]}"
  update_frozen_manifest_outputs \
    "mscxr_hmaps_npy" "${MSCXR_HMAPS}" \
    "mscxr_eval_summary_json" "${MSCXR_SUMMARY_JSON}" \
    "mscxr_case_diagnostics_json" "${MSCXR_CASE_DIAGNOSTICS_JSON}"
else
  echo "[7/10] skipping raw MS-CXR heatmaps"
fi

# [8/10] raw scoring against AFLoc/DCEM baselines
if stage_enabled 8; then
  validate_frozen_manifest
  mkdir -p "${SCORE_HMAP_ROOT}/${METHOD_NAME}"
  run_cmd 8 "install AFLoc-MRSG heatmaps for scoring" cp "${MSCXR_HMAPS}" "${SCORE_HMAP_ROOT}/${METHOD_NAME}/hmaps.npy"
  install_reference_hmap "${BASELINE_METHOD}"
  install_reference_hmap "${DCEM_METHOD}"
  run_cmd 8 "raw scoring against AFLoc/DCEM baselines" \
    python -m anaprior.eval.score_mscxr_learned_repair_metrics \
    --hmaps-root "${SCORE_HMAP_ROOT}" \
    --outdir "${METRIC_ROOT}" \
    --methods "${SCORE_METHODS}" \
    --candidate-categories "${FINDINGS}" \
    --dataset "${SCORE_DATASET}" \
    --val-fraction "${SCORE_VAL_FRACTION}" \
    --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
    --seed "${SEED}" \
    --candidate-effect-floor "${CANDIDATE_EFFECT_FLOOR}" \
    --macro-all-harm-floor "${MACRO_ALL_HARM_FLOOR}" \
    --margin \
    "${MAX_CASES_ARGS[@]}"
  update_frozen_manifest_outputs \
    "score_metrics_summary_json" "${METRIC_SUMMARY_JSON}" \
    "score_decision_json" "${METRIC_DECISION_JSON}"
else
  echo "[8/10] skipping raw scoring against AFLoc/DCEM baselines"
fi

# [9/10] frozen CheXlocalize external evaluation
if stage_enabled 9; then
  validate_frozen_manifest
  run_cmd 9 "frozen CheXlocalize external evaluation" \
    python -m anaprior.eval.eval_mscxr_afloc_mrsg \
    --dataset "CHEXLOCALIZE" \
    --split "test" \
    --afloc-checkpoint "${AFLOC_CHECKPOINT}" \
    --checkpoint "${PHASE_C_BEST}" \
    --outdir "${CHEXLOCALIZE_EVAL_ROOT}" \
    --device "${DEVICE}" \
    --method-name "${METHOD_NAME}" \
    "${MAX_CASES_ARGS[@]}"
  update_frozen_manifest_outputs \
    "chexlocalize_hmaps_npy" "${CHEXLOCALIZE_HMAPS}" \
    "chexlocalize_eval_summary_json" "${CHEXLOCALIZE_SUMMARY_JSON}" \
    "chexlocalize_case_diagnostics_json" "${CHEXLOCALIZE_CASE_DIAGNOSTICS_JSON}"
else
  echo "[9/10] skipping frozen CheXlocalize external evaluation"
fi

# [10/10] report and diagnostics bundle
if stage_enabled 10; then
  validate_frozen_manifest
  run_cmd 10 "report and diagnostics bundle" \
    python -m anaprior.eval.report_stage_c_results \
    --metrics-dir "${METRIC_ROOT}" \
    --output-md "${REPORT_MD}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    python - "${BUNDLE_JSON}" "${REPORT_MD}" "${FROZEN_MANIFEST}" "${CACHE_REPORT}" "${PHASE_A_REPORT}" "${PHASE_B_REPORT}" "${PHASE_C_REPORT}" "${MSCXR_SUMMARY_JSON}" "${METRIC_SUMMARY_JSON}" "${CHEXLOCALIZE_SUMMARY_JSON}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


bundle_json = Path(sys.argv[1])
paths = [Path(item) for item in sys.argv[2:]]
bundle = {
    "status": "ok",
    "artifacts": [],
}
for path in paths:
    if not path.exists():
      raise SystemExit(f"bundle artifact missing: {path}")
    bundle["artifacts"].append(
        {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    )
bundle_json.parent.mkdir(parents=True, exist_ok=True)
bundle_json.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  fi
  update_frozen_manifest_outputs \
    "report_md" "${REPORT_MD}" \
    "bundle_json" "${BUNDLE_JSON}"
else
  echo "[10/10] skipping report and diagnostics bundle"
fi

echo "[AFLoc-MRSG] done"
echo "[AFLoc-MRSG] cache report: ${CACHE_REPORT}"
echo "[AFLoc-MRSG] phase A report: ${PHASE_A_REPORT}"
echo "[AFLoc-MRSG] phase B report: ${PHASE_B_REPORT}"
echo "[AFLoc-MRSG] phase C report: ${PHASE_C_REPORT}"
echo "[AFLoc-MRSG] frozen manifest: ${FROZEN_MANIFEST}"
echo "[AFLoc-MRSG] MS-CXR summary: ${MSCXR_SUMMARY_JSON}"
echo "[AFLoc-MRSG] score summary: ${METRIC_SUMMARY_JSON}"
echo "[AFLoc-MRSG] CheXlocalize summary: ${CHEXLOCALIZE_SUMMARY_JSON}"
echo "[AFLoc-MRSG] report: ${REPORT_MD}"
echo "[AFLoc-MRSG] bundle: ${BUNDLE_JSON}"
