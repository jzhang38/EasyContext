from .dist_flash_attn.prepare_input import prepare_dist_flash_attn_inputs
from .dist_flash_attn.monkey_patch import apply_dist_flash_attn_monkey_patch_llama
from .zigzag_ring_attn.prepare_inputs import prepare_zigzag_ring_attn_inputs    
from .zigzag_ring_attn.monkey_patch import apply_zigzag_ring_attn_monkey_patch_llama    
from .zigzag_ring_attn.monkey_patch import apply_zigzag_ring_attn_monkey_patch_mistral
from .unsloth_offloaded_gradient_checkpoint.monkey_patch import apply_unsloth_offloaded_gradient_checkpoint_monkey_patch
from .ulysses_attn.prepare_inputs import prepare_ulysses_attn_inputs  
from .ulysses_attn.monkey_patch import apply_ulysses_attn_monkey_patch_llama 

def prepare_seq_parallel_inputs(
    seq_algo, input_ids, position_ids, target_ids, rank, world_size, device
):
    if seq_algo == "zigzag_ring_attn":
        return prepare_zigzag_ring_attn_inputs(
            input_ids, position_ids, target_ids, rank, world_size, device
        )
    elif seq_algo == "dist_flash_attn":
        return prepare_dist_flash_attn_inputs(
            input_ids, position_ids, target_ids, rank, world_size, device
        )
    elif seq_algo == "ulysses_attn":
        return prepare_ulysses_attn_inputs(
            input_ids, position_ids, target_ids, rank, world_size, device
        )
    elif seq_algo == "data_parallel":
        return {
            "local_input_ids": input_ids.to(device),
            "local_position_ids": position_ids.to(device),
            "local_target_ids": target_ids.to(device),
        }
    else:
        raise ValueError(f"Invalid seq_algo: {seq_algo}")
    
def apply_seq_parallel_monkey_patch(
    seq_algo, model
):
    assert seq_algo in ["zigzag_ring_attn", "dist_flash_attn", "ulysses_attn", "data_parallel"], f"Invalid seq_algo: {seq_algo}"
    assert model in ["llama", "mistral"], f"Invalid model: {model}"
    if seq_algo == "data_parallel":
        return
    elif seq_algo == "zigzag_ring_attn" and model == "llama":
        apply_zigzag_ring_attn_monkey_patch_llama()
    elif seq_algo == "zigzag_ring_attn" and model == "mistral":
        apply_zigzag_ring_attn_monkey_patch_mistral()
    elif seq_algo == "dist_flash_attn" and model == "llama":
        apply_dist_flash_attn_monkey_patch_llama()
    elif seq_algo == "ulysses_attn" and model == "llama":
        apply_ulysses_attn_monkey_patch_llama()
    else:
        raise ValueError(f"Invalid seq_algo: {seq_algo} or model: {model}")
        
def prepare_dataloader(seq_algo, dataloader, acclerator):
    if seq_algo == "data_parallel":
        return acclerator.prepare(dataloader)
    else:
        return dataloader