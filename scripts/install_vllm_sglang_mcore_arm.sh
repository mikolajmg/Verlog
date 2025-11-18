#!/bin/bash

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}

# Set build jobs, 32 is a good number for a powerful build server
export MAX_JOBS=32
# Tell pip to build from source if a wheel isn't available
export PIP_NO_BINARY=:all:

echo "1. install inference frameworks and pytorch they need"
if [ $USE_SGLANG -eq 1 ]; then
    # MODIFIED: Removed [all] to avoid sgl-kernel build issues on ARM
    pip install "sglang==0.4.6.post1" --no-cache-dir --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python
    
    # torch-memory-saver may not have an aarch64 wheel, this will try to build it
    pip install torch-memory-saver --no-cache-dir
fi

# MODIFIED: Removed version pin "==0.6.2" from tensordict
# NOTE: vllm 0.8.5.post1 is very old and may fail to build on ARM.
# If it fails, try removing the version pin: "vllm"
pip install --no-cache-dir "vllm==0.8.5.post1" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" tensordict torchdata

echo "2. install basic packages"
# MODIFIED: Removed 'pyext' as it is incompatible with Python 3.11
pip install "transformers<4.43.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
# MODIFIED: Removed all wget commands.
# We will now build flash-attn and flashinfer from source.
# This will take a long time.
echo "Building FlashAttention from source... This will take a long time."
# --no-build-isolation is often needed to use the system's C++ compiler/CUDA
pip install flash-attn --no-build-isolation --no-cache-dir

# flashinfer will be installed as a dependency of sglang or vllm
# If it's missing, you can try: pip install flashinfer --no-build-isolation
echo "FlashAttention build complete."


if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    # This is already building from source, which is correct for ARM.
    NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.0rc3
fi


echo "5. May need to fix opencv"
# These packages have aarch64 wheels, so this step is fine.
pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"


if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    # This package should be compatible with aarch64
    pip install nvidia-cudnn-cu12==9.8.0.87
fi

echo "Successfully installed all packages"