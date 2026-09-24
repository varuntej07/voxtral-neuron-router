#!/usr/bin/env bash
# Build the trn1 training venv. Run once per instance: bash scripts/setup_train_venv.sh
#
# This is the recipe from cpu-torch-first-or-cuda-wheels-land and
# pin-neuronx-cc-to-match-torch-neuronx in lessons.txt, plus peft for training.
# Order matters: CPU torch goes in first, or pip pulls about 6 GB of CUDA wheels.
set -euo pipefail

VENV="$HOME/venv-train"
export PIP_NO_CACHE_DIR=1
started=$(date +%s)

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip

pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
pip install 'torch-neuronx==2.9.*' 'neuronx-cc==2.26.6360.0' \
    --extra-index-url https://pip.repos.neuron.amazonaws.com
# peft 0.14 or later is needed for exclude_modules, which keeps LoRA off the audio tower.
pip install 'transformers==4.56.*' 'peft==0.17.*' 'huggingface_hub>=0.34' \
    'mistral_common[audio]' soundfile librosa

nvidia=$(pip list 2>/dev/null | grep -ci '^nvidia' || true)
cc=$(pip show neuronx-cc | awk '/^Version:/ {print $2}')
echo "nvidia packages: $nvidia"
echo "neuronx-cc: $cc"
pip list 2>/dev/null | grep -E '^(torch|torch-neuronx|torch-xla|libneuronxla|neuronx-cc|transformers|peft) '
echo "venv build seconds: $(( $(date +%s) - started ))"

if [ "$nvidia" != "0" ]; then echo "FAIL: CUDA wheels landed, CPU torch was not pinned first"; exit 1; fi
if [ "$cc" != "2.26.6360.0" ]; then echo "FAIL: neuronx-cc is $cc, want 2.26.6360.0"; exit 1; fi
echo OK
