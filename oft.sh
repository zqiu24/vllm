. /home/zqiu/anaconda3/etc/profile.d/conda.sh
module load cuda/12.9
conda activate vllm1

export VLLM_ATTENTION_BACKEND=FLASHINFER

python oft.py