#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
env/bin/python -m pip install -r configs/requirements-inference.txt
if [ ! -d source/LLaVA/.git ]; then
  mkdir -p source/LLaVA
  git -C source/LLaVA init
  git -C source/LLaVA remote add origin https://github.com/haotian-liu/LLaVA.git
fi
git -C source/LLaVA sparse-checkout init --cone
git -C source/LLaVA sparse-checkout set llava
git -C source/LLaVA config core.repositoryformatversion 1
git -C source/LLaVA config extensions.partialClone origin
git -C source/LLaVA config remote.origin.promisor true
git -C source/LLaVA config remote.origin.partialCloneFilter blob:none
git -C source/LLaVA -c http.version=HTTP/1.1 fetch --depth 1 --filter=blob:none origin c121f0432da27facab705978f83c4ada465e46fd
git -C source/LLaVA checkout --detach c121f0432da27facab705978f83c4ada465e46fd
env/bin/python -m pip install --no-deps -e source/LLaVA
env/bin/python -m pip freeze > logs/environment-compatible.txt
env/bin/python -c 'import torch, transformers, peft; print("TORCH", torch.__version__, "CUDA", torch.cuda.is_available(), "PEFT", peft.__version__); from llava.model.builder import load_pretrained_model; print("LLAVA_IMPORT_OK")'
