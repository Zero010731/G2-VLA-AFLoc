#!/usr/bin/env bash
set -euo pipefail

# Example server runner for AFLoc MS-CXR localization.
# Edit these two paths to match your server.

export MS_CXR_JSON="${MS_CXR_JSON:-/mnt/mimic-cxr/ms-cxr_1.1.0/MS_CXR_Local_Alignment_v1.1.0.json}"

# Important:
# AFLoc strips the leading "files/" from MS-CXR image paths before joining.
# If a sample image is:
#   /mnt/mimic-cxr/jpg/files/p10/.../xxx.jpg
# then use:
#   /mnt/mimic-cxr/jpg/files
#
# If a sample image is:
#   /mnt/mimic-cxr/jpg/p10/.../xxx.jpg
# then use:
#   /mnt/mimic-cxr/jpg
export MIMIC_CXR_JPG_ROOT="${MIMIC_CXR_JPG_ROOT:-/mnt/mimic-cxr/jpg/files}"

CKPT="${CKPT:-weight/pretrained/Pretrained_CXR.ckpt}"
GPU="${GPU:-0}"

echo "MS_CXR_JSON=${MS_CXR_JSON}"
echo "MIMIC_CXR_JPG_ROOT=${MIMIC_CXR_JPG_ROOT}"
echo "CKPT=${CKPT}"
echo "GPU=${GPU}"

python scripts/check_ms_cxr_paths.py \
  --ms-cxr "${MS_CXR_JSON}" \
  --mimic-root "${MIMIC_CXR_JPG_ROOT}" \
  --n 20

python localization.py \
  --ckpt "${CKPT}" \
  --dataset MS_CXR \
  --gpu "${GPU}" \
  --redo
