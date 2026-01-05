. /home/zqiu/anaconda3/etc/profile.d/conda.sh
module load cuda/12.9
conda activate vllm

export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_TORCH_COMPILE_LEVEL=0

python lora.py