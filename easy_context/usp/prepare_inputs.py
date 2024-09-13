import torch

from yunchang.comm import zigzag_extract_local
from yunchang import set_seq_parallel_pg

def prepare_usp_attn_inputs(
    input_ids, position_ids, target_ids, rank, world_size, ring_degree, ulysses_degree, device
):
    f"""
    prepare input for USP attention

    USP: A Unified Sequence Parallelism Approach for Long Context Generative AI
    https://arxiv.org/abs/2405.07719
    
    input_ids: (batch_size, seq_len)
    position_ids: (batch_size, seq_len)
    target_ids: (batch_size, seq_len)
    rank: int
    world_size: int
    ring_degree: int
    ulysses_degree: int
    device: torch.device
    """

    set_seq_parallel_pg(usp_attn, ulysses_degree, ring_degree, rank, world_size)

    local_input_ids = zigzag_extract_local(
        input_ids,
        rank,
        world_size,
        device,
        ring_degree,
        ulysses_degree,
    )
    local_position_ids = zigzag_extract_local(
        position_ids,
        rank,
        world_size,
        device,
        ring_degree,
        ulysses_degree,
    )

    if target_ids is not None:
        local_target_ids = zigzag_extract_local(
            target_ids,
            rank,
            world_size,
            device,
            ring_degree,
            ulysses_degree,
        )
    else:
        local_target_ids = None
    return {
        "local_input_ids": local_input_ids,
        "local_position_ids": local_position_ids,
        "local_target_ids": local_target_ids,
    }
