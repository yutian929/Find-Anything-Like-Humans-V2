#!/usr/bin/env bash
# Download the local weights expected by anchor-map/config.py.
set -uo pipefail

# 默认下载与 config.py 一致的 RynnBrain-2B。
# 环境变量可临时覆盖，便于同时下载两个版本而不改动默认值。
RYNNBRAIN_SIZE="${RYNNBRAIN_SIZE:-2B}"

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
models_dir="${project_dir}/models"
selection="${1:-all}"

case "${selection}" in
  all|rynnbrain|sam2|sam3) ;;
  *) echo "Usage: $0 [all|rynnbrain|sam2|sam3]" >&2; exit 2 ;;
esac
case "${RYNNBRAIN_SIZE}" in
  8B|2B) ;;
  *) echo "RYNNBRAIN_SIZE must be 8B or 2B" >&2; exit 2 ;;
esac

if [[ "${selection}" != sam3 ]] && ! command -v hf >/dev/null 2>&1; then
  echo "Missing 'hf' CLI. Activate the project environment and install huggingface_hub[cli]." >&2
  exit 2
fi
if [[ "${selection}" == all || "${selection}" == sam3 ]] && ! command -v modelscope >/dev/null 2>&1; then
  echo "Missing 'modelscope' CLI. Activate the project environment and install modelscope." >&2
  exit 2
fi

mkdir -p "${models_dir}"
failed=0

verify_model() {
  local name="$1" directory="$2"
  if ! python - "${directory}" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
config = directory / "config.json"
index = directory / "model.safetensors.index.json"
if not config.is_file():
    raise SystemExit(f"Missing {config}")
if index.is_file():
    weight_files = {directory / name for name in json.loads(index.read_text())["weight_map"].values()}
else:
    weight_files = {directory / "model.safetensors"}
missing = [str(path) for path in weight_files if not path.is_file() or path.stat().st_size < 1024]
if missing:
    raise SystemExit("Missing or incomplete weights: " + ", ".join(missing))
print(f"Verified {directory.name}: config + {len(weight_files)} weight file(s)")
PY
  then
    echo "Validation failed: ${name}" >&2
    return 1
  fi
}

download_hf_model() {
  local name="$1" repo="$2" directory="$3"
  echo "==> ${name}: Hugging Face ${repo} -> ${directory}"
  # The runtime uses Transformers safetensors, not the original PyTorch checkpoints.
  if ! HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf download "${repo}" \
      --local-dir "${directory}" --exclude '*.pt' '*.pth' '*.bin' --max-workers 4; then
    echo "Download failed: ${name}" >&2
    return 1
  fi
  verify_model "${name}" "${directory}"
}

download_modelscope_model() {
  local name="$1" repo="$2" directory="$3"
  echo "==> ${name}: ModelScope ${repo} -> ${directory}"
  if ! modelscope download "${repo}" --local-dir "${directory}" \
      --exclude '*.pt' '*.pth' '*.bin' --max-workers 4; then
    echo "Download failed: ${name}" >&2
    return 1
  fi
  verify_model "${name}" "${directory}"
}

if [[ "${selection}" == all || "${selection}" == rynnbrain ]]; then
  download_hf_model "RynnBrain-${RYNNBRAIN_SIZE}" \
    "Alibaba-DAMO-Academy/RynnBrain-${RYNNBRAIN_SIZE}" \
    "${models_dir}/RynnBrain-${RYNNBRAIN_SIZE}" || failed=1
fi
if [[ "${selection}" == all || "${selection}" == sam2 ]]; then
  download_hf_model SAM2.1 facebook/sam2.1-hiera-small \
    "${models_dir}/sam2.1-hiera-small" || failed=1
fi
if [[ "${selection}" == all || "${selection}" == sam3 ]]; then
  download_modelscope_model SAM3 facebook/sam3 "${models_dir}/sam3" || failed=1
fi

exit "${failed}"
