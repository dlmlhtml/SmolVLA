#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Please run inside Linux or WSL2, not macOS / native Windows.' >&2
  exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12"'
command -v git >/dev/null
if command -v nvidia-smi >/dev/null; then
  nvidia-smi
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  /usr/lib/wsl/lib/nvidia-smi
else
  echo 'GPU not detected. On Windows, update the Windows NVIDIA driver and use WSL2. See WINDOWS_4060.md.' >&2
  exit 1
fi
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
mkdir -p vendor
if [[ ! -d vendor/lerobot ]]; then
  git init vendor/lerobot
  git -C vendor/lerobot remote add origin https://github.com/huggingface/lerobot.git
  git -C vendor/lerobot fetch --depth 1 origin b6ec0060779550c0a157ae34feb89e0cf86012a8
  git -C vendor/lerobot checkout --detach FETCH_HEAD
fi
test "$(git -C vendor/lerobot rev-parse HEAD)" = b6ec0060779550c0a157ae34feb89e0cf86012a8
python -m pip install -e 'vendor/lerobot[smolvla,libero]'
python -c 'import torch; print("torch:", torch.__version__); assert torch.cuda.is_available(), "CUDA unavailable: check NVIDIA driver / PyTorch installation"; print("GPU:", torch.cuda.get_device_name(0))'
python rollout_smolvla.py --check
python -m pip freeze > environment-installed.txt
echo 'Environment dependencies installed. Actual LIBERO rendering still needs the rollout test.'
