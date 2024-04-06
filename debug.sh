accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 1 \
--output-dir ./output/dist_attn_debug \
--wandb EasyContext \
--max-train-steps 400  \
--learning-rate 2e-5  \
--dataset yaofu_data \
--model meta-llama/Llama-2-7b-hf  \
--seq-length 64000 \
--rope-theta 1000000 \
--dist_flash_attention


ring flash attention:

| 1/1000 [00:16<4:42:58, 17.00s/it, loss=4.73
| 2/1000 [00:27<3:37:13, 13.06s/it, loss=3.86
  3/1000 [00:38<3:19:39, 12.02s/it, loss=3.71, ppl=40.8]
  4/1000 [00:48<3:08:04, 11.33s/it, loss=2.91, ppl=18.3]
  5/1000 [00:58<3:01:56, 10.97s/it, loss=3.01, ppl=20.
  6/1000 [01:11<3:13:18, 11.67s/it, loss=2.45, ppl=11.6]
  7/1000 [01:21<3:05:21, 11.20s/it, loss=1.84, ppl=6.3]

dist flash attention
 
 | 1/1000 [00:15<4:11:53, 15.13s/it, loss=3.76, ppl=42.9]
 | 2/1000 [00:23<3:08:18, 11.32s/it, loss=2.99, ppl=20]
  | 3/1000 [00:31<2:44:05,  9.88s/it, loss=2.1, ppl=8.17]
  | 4/1000 [00:40<2:36:40,  9.44s/it, loss=3.01, ppl=20.2]
  | 5/1000 [00:48<2:29:04,  8.99s/it, loss=2.37, ppl=10.7]
  | 6/1000 [00:57<2:24:46,  8.74s/it, loss=1.72, ppl=5.57]
  | 7/1000 [01:04<2:18:08,  8.35s/it, loss=1.32, ppl=3.76]

What???
Is my implementation correct?


Dist flash ring attention is much faster tho.