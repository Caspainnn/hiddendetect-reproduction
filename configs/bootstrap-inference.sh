#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
env/bin/python -m pip install -r configs/requirements-inference-author.txt
if [ ! -d source/LLaVA/.git ]; then
  git clone --filter=blob:none --no-checkout https://github.com/haotian-liu/LLaVA.git source/LLaVA
fi
git -C source/LLaVA sparse-checkout init --cone
git -C source/LLaVA sparse-checkout set llava
git -C source/LLaVA checkout --detach c121f0432da27facab705978f83c4ada465e46fd
env/bin/python -m pip install --no-deps -e source/LLaVA
env/bin/python -m pip freeze > logs/environment-author-attempt.txt
env/bin/python -c 'import torch, transformers; print("TORCH", torch.__version__, "CUDA", torch.cuda.is_available()); from llava.model.builder import load_pretrained_model; print("LLAVA_IMPORT_OK")'
