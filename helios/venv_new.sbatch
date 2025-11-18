#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:29:58
#SBATCH --gres=gpu:1
#SBATCH --account=plgbro4ppo-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm/venv-%j.out

export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

ml ML-bundle/24.06a
ml GCCcore/13.3.0





rm -rf .venv
python3 -m venv .venv


source .venv/bin/activate


pip3 install vllm-0.9.2.dev0+gb6553be1b.d20250704-cp311-cp311-linux_aarch64.whl
pip install --no-cache-dir torch==2.8.0+cu124  --index-url https://download.pytorch.org/whl/cu124
#cd ../BALROG

#pip install -e .

#balrog-post-install

pip install --no-cache-dir tensordict

pip install --no-cache-dir  accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff

pip install --no-cache-dir torchdata



# pip install triton --no-cache-dir
# pip install trl --no-cache-dir
# pip install flash-attn --no-cache-dir
pip install flash-attn==2.8.3 --no-cache-dir --no-build-isolation
#pip install vllm==0.11.0 --no-binary :all: --no-cache-dir
#pip install vllm --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128
#pip install --no-cache-dir "vllm>=0.8.4" --extra-index-url https://download.pytorch.org/whl/cu128

pip install "transformers<4.42.0" --no-cache-dir
pip install --no-cache-dir tensordict codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff torchdata
cd ../Verlog

#USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore_arm.sh

pip install -e --no-cache-dir .

#pip install --no-cache-dir ".[vllm]"



pip list



exit 0
