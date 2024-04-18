# First, download h2oai/h2o-danube2-1.8b-base to output/h2o-danube2-1.8b-base and change the "architectures" from MistralForCausalLM to LlamaForCausalLM and "model_type" from mistral to llama in the config.json file.
# HF's mistral architecture is off when used together with ring attention due to the way it implements rotary embedding. Llama has no such issue.

accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 2 \
--gradient-accumulate-every 4 \
--output-dir ./output/h2o_bs_1M_step_400_lr_2e-5_16K_rope_30K \
--wandb EasyContext \
--seed 2024 \
--max-train-steps 400  \
--learning-rate 2e-5  \
--dataset PY007/slimpajama_mistral_tokenized_upsample_4096_chunk_128K \
--model output/h2o-danube2-1.8b-base  \
--seq-length 16000 \
--rope-theta 30000 \
--parallel_mode data_parallel


accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 1 \
--gradient-accumulate-every 4 \
--output-dir ./output/h2o_bs_1M_step_400_lr_2e-5_32K_rope_100K \
--wandb EasyContext \
--seed 2025 \
--max-train-steps 400  \
--learning-rate 2e-5  \
--dataset PY007/slimpajama_mistral_tokenized_upsample_4096_chunk_128K \
--model output/h2o_bs_1M_step_400_lr_2e-5_16K_rope_30K  \
--seq-length 32000 \
--rope-theta 100000 \
--parallel_mode data_parallel


accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 1 \
--gradient-accumulate-every 4 \
--output-dir ./output/h2o_bs_1M_step_400_lr_2e-5_32K_rope_300K \
--wandb EasyContext \
--seed 2026 \
--max-train-steps 400  \
--learning-rate 2e-5  \
--dataset PY007/slimpajama_mistral_tokenized_upsample_4096_chunk_128K \
--model output/h2o_bs_1M_step_400_lr_2e-5_32K_rope_100K  \
--seq-length 32000 \
--rope-theta 300000 \
--parallel_mode data_parallel



accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 4 \
--gradient-accumulate-every 4 \
--output-dir ./output/h2o_bs_1M_step_400_lr_2e-5_64K_rope_1M \
--wandb EasyContext \
--seed 2027 \
--max-train-steps 400  \
--learning-rate 2e-5  \
--dataset PY007/slimpajama_mistral_tokenized_upsample_4096_chunk_128K \
--model output/h2o_bs_1M_step_400_lr_2e-5_32K_rope_300K  \
--seq-length 64000 \
--rope-theta 1000000 \
--parallel_mode zigzag_ring_attn



accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 4 \
--gradient-accumulate-every 4 \
--output-dir ./output/h2o_bs_1M_step_400_lr_2e-5_64K_rope_2.5M \
--wandb EasyContext \
--seed 2028 \
--max-train-steps 400  \
--learning-rate 2e-5  \
--dataset PY007/slimpajama_mistral_tokenized_upsample_4096_chunk_128K \
--model output/h2o_bs_1M_step_400_lr_2e-5_64K_rope_1M  \
--seq-length 64000 \
--rope-theta 2500000 \
--parallel_mode zigzag_ring_attn

accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 4 \
--gradient-accumulate-every 4 \
--output-dir ./output/h2o_bs_1M_step_400_lr_2e-5_64K_rope_5M \
--wandb EasyContext \
--seed 2029 \
--max-train-steps 400  \
--learning-rate 2e-5  \
--dataset PY007/slimpajama_mistral_tokenized_upsample_4096_chunk_128K \
--model output/h2o_bs_1M_step_400_lr_2e-5_64K_rope_2.5M  \
--seq-length 64000 \
--rope-theta 5000000 \
--parallel_mode zigzag_ring_attn

accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 2 \
--gradient-accumulate-every 4 \
--output-dir ./output/h2o_bs_1M_step_300_lr_2e-5_128K_rope_10M \
--wandb EasyContext \
--seed 2030 \
--max-train-steps 300  \
--learning-rate 2e-5  \
--dataset PY007/slimpajama_mistral_tokenized_upsample_4096_chunk_128K \
--model PY007/EasyContext-64K-h2o  \
--seq-length 128000 \
--rope-theta 10000000 \
--parallel_mode zigzag_ring_attn


accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 2 \
--gradient-accumulate-every 4 \
--output-dir ./output/h2o_bs_1M_step_200_lr_2e-5_128K_rope_30M \
--wandb EasyContext \
--seed 2031 \
--max-train-steps 200  \
--learning-rate 2e-5  \
--dataset PY007/slimpajama_mistral_tokenized_upsample_4096_chunk_128K \
--model output/h2o_bs_1M_step_300_lr_2e-5_128K_rope_10M  \
--seq-length 128000 \
--rope-theta 30000000 \
--parallel_mode zigzag_ring_attn


accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 1 \
--gradient-accumulate-every 4 \
--output-dir ./output/h2o_bs_1M_step_200_lr_2e-5_256K_rope_100M \
--wandb EasyContext \
--seed 2032 \
--max-train-steps 200  \
--learning-rate 2e-5  \
--dataset PY007/slimpajama_mistral_tokenized_upsample_4096_chunk_1M \
--model output/h2o_bs_1M_step_200_lr_2e-5_128K_rope_30M  \
--seq-length 256000 \
--rope-theta 100000000 \
--parallel_mode zigzag_ring_attn


