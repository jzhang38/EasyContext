# You can observe that the number of steps for different stage is quite different. They are not magic number. They are set to those numbers simply because I esitimate the time it takes to finish the training, and 
# choose the number such that it fits my daily schedule>_<. This is for you to exactly reproduce my results. You many change the steps to other numbers if you want to.

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 

accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 4 \
--output-dir ./output/7B_32K_bs_1M_rope_1M_step_1000_lr_2e-5 \
--wandb EasyContext \
--max-train-steps 1000  \
--learning-rate 2e-5  \
--dataset yaofu/slimpajama-per-source-length-upsample \
--model meta-llama/Llama-2-7b-hf  \
--seq-length 32768 \
--rope-theta 1000000 \
--parallel_mode data_parallel

# In the saved files, there are model-00001-of-00003.safetensors to model-00001-of-00003.safetensors. Somehow model.safetensors is unnecessary and should be removed.
rm output/7B_32K_bs_1M_rope_1M_step_1000_lr_2e-5/model.safetensors

accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 2 \
--output-dir ./output/7B_64K_bs_1M_rope_5M_step_1000_lr_2e-5 \
--seed 2022 \
--wandb EasyContext \
--max-train-steps 1000  \
--learning-rate 2e-5  \
--dataset yaofu/slimpajama-per-source-length-upsample \
--model output/7B_32K_bs_1M_rope_1M_step_1000_lr_2e-5  \
--seq-length 65536 \
--rope-theta 5000000 \
--parallel_mode data_parallel

rm output/7B_64K_bs_1M_rope_5M_step_1000_lr_2e-5/model.safetensors

accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 4  \
--output-dir  ./output/7B_0.256M_bs_1M_rope_10M_step_500_lr_2e-5 \
--seed 2023 \
--wandb EasyContext \
--max-train-steps 500  \
--learning-rate 2e-5  \
--dataset PY007/slimpajama_llama_tokenized_upsample_4096_chunk_256K \
--model output/7B_64K_bs_1M_rope_5M_step_1000_lr_2e-5  \
--seq-length 256000 \
--rope-theta 10000000 \
--parallel_mode zigzag_ring_attn

rm output/7B_0.256M_bs_1M_rope_10M_step_500_lr_2e-5/model.safetensors

accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 4  \
--output-dir  ./output/7B_0.256M_bs_1M_rope_25M_step_500_lr_2e-5 \
--seed 2024 \
--wandb EasyContext \
--max-train-steps 500  \
--learning-rate 2e-5  \
--dataset PY007/slimpajama_llama_tokenized_upsample_4096_chunk_256K \
--model output/7B_0.256M_bs_1M_rope_10M_step_500_lr_2e-5  \
--seq-length 256000 \
--rope-theta 25000000 \
--parallel_mode zigzag_ring_attn

rm output/7B_0.256M_bs_1M_rope_25M_step_500_lr_2e-5/model.safetensors

accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 4  \
--output-dir  ./output/7B_0.256M_bs_1M_rope_50M_step_150_lr_2e-5 \
--seed 2025 \
--wandb EasyContext \
--max-train-steps 150  \
--learning-rate 2e-5  \
--dataset PY007/slimpajama_llama_tokenized_upsample_4096_chunk_256K \
--model output/7B_0.256M_bs_1M_rope_25M_step_500_lr_2e-5  \
--seq-length 256000 \
--rope-theta 50000000 \
--parallel_mode zigzag_ring_attn

rm output/7B_0.256M_bs_1M_rope_50M_step_150_lr_2e-5/model.safetensors

accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 2  \
--output-dir  ./output/7B_0.5M_bs_1M_rope_100M_step_300_lr_2e-5 \
--seed 2026 \
--wandb EasyContext \
--max-train-steps 300  \
--learning-rate 2e-5  \
--dataset PY007/slimpajama_llama_tokenized_upsample_4096_chunk_1M \
--model output/7B_0.256M_bs_1M_rope_50M_step_150_lr_2e-5  \
--seq-length 512000 \
--rope-theta 100000000 \
--parallel_mode zigzag_ring_attn


rm output/7B_0.5M_bs_1M_rope_100M_step_300_lr_2e-5/model.safetensors

accelerate launch \
--config_file accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 2  \
--output-dir  ./output/7B_0.5M_bs_1M_rope_250M_step_90_lr_2e-5 \
--seed 2027 \
--wandb EasyContext \
--max-train-steps 90  \
--learning-rate 1e-5  \
--dataset PY007/slimpajama_llama_tokenized_upsample_4096_chunk_1M \
--model output/7B_0.5M_bs_1M_rope_100M_step_300_lr_2e-5  \
--seq-length 512000 \
--rope-theta 250000000 \
--parallel_mode zigzag_ring_attn

rm output/7B_0.5M_bs_1M_rope_250M_step_90_lr_2e-5/model.safetensors


### Finally we directly set the rope_theta in output/7B_0.5M_bs_1M_rope_250M_step_90_lr_2e-5/config.json to 1,000,000,000
