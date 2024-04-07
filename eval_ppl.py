import argparse
import datasets
import gc
import sys
import torch
import warnings
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from accelerate import Accelerator
from flash_attn.losses.cross_entropy import CrossEntropyLoss

from easy_context import (
    prepare_seq_parallel_inputs,
    apply_seq_parallel_monkey_patch,
)
apply_seq_parallel_monkey_patch("zigzag_ring_attn", "llama")

def compute_perplexity(
    encodings,
    model,
    tokenizer,
    add_start_token: bool = True,
    accelerator=None,
    max_length=None,
    sliding_window=256,
    truncate=False,
    aggressive_memory=False,
    hide_progress=False,
):

    device = accelerator.device
    if add_start_token:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    if max_length and truncate:
        encoded_texts = [x[0:max_tokenized_len] for x in encoded_texts]
        attn_masks = [x[0:max_tokenized_len] for x in attn_masks]
        sliding_window = max_tokenized_len
    loss_func = CrossEntropyLoss()
    pbar = tqdm(total=len(encoded_texts), disable=not accelerator.is_local_main_process)
    nlls = []
    for encoding_index in range(0, len(encoded_texts)):

        labels = torch.tensor(encoded_texts[encoding_index : encoding_index + 1])
        seq_len = labels.size(1)

        prev_end_loc = 0
        for begin_loc in range(0, seq_len, sliding_window):

            end_loc = min(begin_loc + max_tokenized_len, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = labels[:, begin_loc:end_loc]

            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[tokenizer.bos_token_id]] * input_ids.size(dim=0)
                )
                input_ids = torch.cat([bos_tokens_tensor, input_ids], dim=1)

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            # move target to the left by one (remember to add one new -100)
            target_ids = target_ids.roll(-1, dims=1)
            target_ids[:, -1] = -100
            position_ids = (
                torch.arange(target_ids.shape[1])
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )

            prepared = prepare_seq_parallel_inputs(
                "zigzag_ring_attn",
                input_ids,
                position_ids,
                target_ids,
                accelerator.process_index,
                accelerator.num_processes,
                accelerator.device,
            )
            local_input_ids = prepared["local_input_ids"]
            local_position_ids = prepared["local_position_ids"]
            local_target_ids = prepared["local_target_ids"]
            with torch.inference_mode():
                outputs = model(
                    local_input_ids,
                    position_ids=local_position_ids
                ).logits
                neg_log_likelihood = loss_func(
                    outputs.view(-1, outputs.shape[-1]), local_target_ids.view(-1)
                )
                neg_log_likelihood = accelerator.reduce(
                    neg_log_likelihood, reduction="mean"
                )
            if aggressive_memory:
                outputs = None
                input_ids = None
                target_ids = None
                gc.collect()
                torch.cuda.empty_cache()

            nlls.append(neg_log_likelihood)

            ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
            pbar.set_postfix(ppl=ppl)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        pbar.update(1)

    ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
    return {"mean_perplexity": ppl}


def main(args):
    models = [x[0] for x in args.model]
    tokenizer = AutoTokenizer.from_pretrained(
        models[0],
        model_max_length=sys.maxsize,
        trust_remote_code=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if args.tokenized:
        try:
            input_texts = datasets.load_from_disk(args.tokenized)
        except:
            input_texts = datasets.load_dataset(
                args.tokenized, name=args.subset, split=args.split
            )
    else:
        input_texts = datasets.load_dataset(
            args.dataset, name=args.subset, split=args.split
        )

        def tokenize(example):
            tokenized = tokenizer(
                example[args.feature],
                add_special_tokens=False,
                padding=True,
                truncation=False,
                max_length=sys.maxsize,
                return_attention_mask=True,
            )
            example["input_ids"] = tokenized["input_ids"]
            example["attention_mask"] = tokenized["attention_mask"]
            example["tokenized_len"] = len(tokenized["input_ids"])
            return example

        input_texts = input_texts.map(tokenize)
        if args.save_tokenized:
            input_texts.save_to_disk(args.save_tokenized)
            print(f"Saved tokenized dataset to {args.save_tokenized}")
            return

    if args.dataset_min_tokens:
        input_texts = input_texts.filter(
            lambda x: x["tokenized_len"] >= args.dataset_min_tokens,
            keep_in_memory=True,
            num_proc=64,
        )
    print("Dataset size after fildering:", len(input_texts))
    if args.samples:
        input_texts = input_texts[: args.samples]

    if args.tokens_step:
        tokens = [
            x for x in range(args.min_tokens, args.max_tokens + 1, args.tokens_step)
        ]
    else:
        tokens = [args.min_tokens]
        while args.min_tokens < args.max_tokens:
            point = tokens[-1] * 2
            if point <= args.max_tokens:
                tokens.append(point)
            else:
                break

    results = []
    accelerator = Accelerator(
        mixed_precision="bf16",
    )
    for model in tqdm(models, desc="Model", leave=False, disable=args.hide_progress):
        torch.cuda.empty_cache()
        loaded = LlamaForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            device_map=accelerator.device,
        )
        loaded = accelerator.prepare(loaded)
        loaded.gradient_checkpointing_enable()
        result = []
        for max_length in tokens:
            ppl = compute_perplexity(
                model=loaded,
                tokenizer=tokenizer,
                accelerator=accelerator,
                encodings=input_texts,
                add_start_token=tokenizer.bos_token is not None,
                max_length=max_length,
                sliding_window=args.sliding_window,
                truncate=args.truncate,
                aggressive_memory=args.aggressive_memory,
                hide_progress=args.hide_progress,
            )["mean_perplexity"]
            if accelerator.is_local_main_process:
                print(f"{model}: {max_length}={ppl}")
            result.append(ppl)

        result.insert(0, model)
        results.append(result)

    if args.output_file and accelerator.is_local_main_process:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(f",{','.join([str(x) for x in tokens])}\n")
            for result in results:
                f.write(f"{','.join([str(x) for x in result])}\n")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", action="append", nargs="+")
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-s", "--subset", type=str)
    parser.add_argument("-f", "--feature", type=str)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--dataset-min-tokens", type=int)
    parser.add_argument("--tokens-step", type=int)
    parser.add_argument("--sliding-window", type=int, default=256)
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--save-tokenized", type=str)
    parser.add_argument("--tokenized", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--aggressive-memory", action="store_true")
    parser.add_argument("--hide-progress", action="store_true")
    main(parser.parse_args())
