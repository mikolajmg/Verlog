#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=05:29:58
#SBATCH --gres=gpu:0
#SBATCH --account=plgbro4ppo-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm/venv-%j.out

export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

ml ML-bundle/24.06a
ml GCCcore/13.3.0
ml CUDA/12.8.0




rm -rf .venv2
python3 -m venv .venv2


source .venv2/bin/activate

pip install wheel
python3 -c "from wheel.bdist_wheel import bdist_wheel; print('OK')"
pip install --upgrade pip setuptools wheel packaging ninja psutil
#pip install --no-cache-dir torch==2.7.1+cu128  --index-url https://download.pytorch.org/whl/cu128

# vvlm needs to be predownloaded

pip install --no-cache-dir torch==2.7.1+cu128  --index-url https://download.pytorch.org/whl/cu128

# vvlm needs to be predownloaded
pip3 install vllm-0.9.2.dev0+gb6553be1b.d20250704-cp311-cp311-linux_aarch64.whl


cd ../BALROG

pip install -e .

balrog-post-install

pip install --no-cache-dir tensordict

pip install --no-cache-dir  accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff


pip install --no-cache-dir tensordict codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff torchdata
pip install /net/scratch/hscra/plgrid/plgmikolajg/Verlog/flash_attn_3-3.0.0b1-cp39-abi3-manylinux2014_aarch64.whl --no-cache-dir
# cd ../flash-attention
# export TORCH_CUDA_ARCH_LIST="80;90"
# MAX_JOBS=1 python setup.py install
pip install bitsandbytes

pip install --upgrade pip wheel --no-cache-dir 
python3 -c "from wheel.bdist_wheel import bdist_wheel; print('OK')"
pip install --upgrade pip setuptools wheel

pip install --no-cache-dir tensordict codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff torchdata
cd ../Verlog


pip install -e --no-cache-dir .

pip install triton --no-cache-dir


pip list
python3 -c "from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis"
python3 -c "from flash_attn_interface.bert_padding import pad_input, unpad_input, rearrange, index_first_axis"
python3 -c "import vllm._C"






exit 0

