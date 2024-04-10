# accelerate launch \
# --config_file  accelerate_configs/single_node.yaml \
# --main_process_port 12345 \
# train.py \
# --batch-size 2 \
# --gradient-accumulate-every 4 \
# --output-dir ./output/h2o_bs_1M_step_400_lr_2e-5_16K_rope_30K \
# --wandb EasyContext \
# --seed 2024 \
# --max-train-steps 400  \
# --learning-rate 2e-5  \
# --dataset mistral_data \
# --model output/h2o-danube2-1.8b-base  \
# --seq-length 16000 \
# --rope-theta 30000 \
# --parallel_mode data_parallel


# accelerate launch \
# --config_file  accelerate_configs/single_node.yaml \
# --main_process_port 12345 \
# train.py \
# --batch-size 1 \
# --gradient-accumulate-every 4 \
# --output-dir ./output/h2o_bs_1M_step_400_lr_2e-5_32K_rope_100K \
# --wandb EasyContext \
# --seed 2025 \
# --max-train-steps 400  \
# --learning-rate 2e-5  \
# --dataset mistral_data \
# --model output/h2o_bs_1M_step_400_lr_2e-5_16K_rope_30K  \
# --seq-length 32000 \
# --rope-theta 100000 \
# --parallel_mode data_parallel


# accelerate launch \
# --config_file  accelerate_configs/single_node.yaml \
# --main_process_port 12345 \
# train.py \
# --batch-size 1 \
# --gradient-accumulate-every 4 \
# --output-dir ./output/h2o_bs_1M_step_400_lr_2e-5_32K_rope_300K \
# --wandb EasyContext \
# --seed 2026 \
# --max-train-steps 400  \
# --learning-rate 2e-5  \
# --dataset mistral_data \
# --model output/h2o_bs_1M_step_400_lr_2e-5_32K_rope_100K  \
# --seq-length 32000 \
# --rope-theta 300000 \
# --parallel_mode data_parallel



# accelerate launch \
# --config_file  accelerate_configs/single_node.yaml \
# --main_process_port 12345 \
# train.py \
# --batch-size 4 \
# --gradient-accumulate-every 4 \
# --output-dir ./output/h2o_bs_1M_step_400_lr_2e-5_64K_rope_1M \
# --wandb EasyContext \
# --seed 2027 \
# --max-train-steps 400  \
# --learning-rate 2e-5  \
# --dataset mistral_data \
# --model output/h2o_bs_1M_step_400_lr_2e-5_32K_rope_300K  \
# --seq-length 64000 \
# --rope-theta 1000000 \
# --parallel_mode zigzag_ring_attn



# accelerate launch \
# --config_file  accelerate_configs/single_node.yaml \
# --main_process_port 12345 \
# train.py \
# --batch-size 4 \
# --gradient-accumulate-every 4 \
# --output-dir ./output/h2o_bs_1M_step_400_lr_2e-5_64K_rope_2.5M \
# --wandb EasyContext \
# --seed 2028 \
# --max-train-steps 400  \
# --learning-rate 2e-5  \
# --dataset mistral_data \
# --model output/h2o_bs_1M_step_400_lr_2e-5_64K_rope_1M  \
# --seq-length 64000 \
# --rope-theta 2500000 \
# --parallel_mode zigzag_ring_attn

# accelerate launch \
# --config_file  accelerate_configs/single_node.yaml \
# --main_process_port 12345 \
# train.py \
# --batch-size 4 \
# --gradient-accumulate-every 4 \
# --output-dir ./output/h2o_bs_1M_step_400_lr_2e-5_64K_rope_5M \
# --wandb EasyContext \
# --seed 2029 \
# --max-train-steps 400  \
# --learning-rate 2e-5  \
# --dataset mistral_data \
# --model output/h2o_bs_1M_step_400_lr_2e-5_64K_rope_2.5M  \
# --seq-length 64000 \
# --rope-theta 5000000 \
# --parallel_mode zigzag_ring_attn

accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 1 \
--gradient-accumulate-every 8 \
--output-dir ./output/h2o_bs_1M_step_300_lr_2e-5_128K_rope_10M \
--wandb EasyContext \
--seed 2030 \
--max-train-steps 300  \
--learning-rate 2e-5  \
--dataset mistral_data \
--model output/h2o_bs_1M_step_400_lr_2e-5_64K_rope_5M  \
--seq-length 128000 \
--rope-theta 10000000 \
--parallel_mode zigzag_ring_attn


accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 1 \
--gradient-accumulate-every 8 \
--output-dir ./output/h2o_bs_1M_step_200_lr_2e-5_128K_rope_30M \
--wandb EasyContext \
--seed 2031 \
--max-train-steps 200  \
--learning-rate 2e-5  \
--dataset mistral_data \
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
--dataset mistral_data_1M \
--model output/h2o_bs_1M_step_200_lr_2e-5_128K_rope_30M  \
--seq-length 256000 \
--rope-theta 100000000 \
--parallel_mode zigzag_ring_attn




accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 1 \
--gradient-accumulate-every 2 \
--output-dir ./output/h2o_bs_1M_step_150_lr_2e-5_512K_rope_300M \
--wandb EasyContext \
--seed 2033 \
--max-train-steps 150  \
--learning-rate 2e-5  \
--dataset mistral_data_1M \
--model output/h2o_bs_1M_step_200_lr_2e-5_256K_rope_100M  \
--seq-length 512000 \
--rope-theta 300000000 \
--parallel_mode zigzag_ring_attn


accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 1 \
--gradient-accumulate-every 1 \
--output-dir ./output/h2o_bs_1M_step_100_lr_2e-5_1M_rope_1B \
--wandb EasyContext \
--seed 2034 \
--max-train-steps 100  \
--learning-rate 2e-5  \
--dataset mistral_data_1M \
--model output/h2o_bs_1M_step_150_lr_2e-5_512K_rope_300M  \
--seq-length 1000000 \
--rope-theta 1000000000 \
--parallel_mode zigzag_ring_attn
