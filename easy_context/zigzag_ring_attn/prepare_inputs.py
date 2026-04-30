import torch


def extract_local(value, rank, world_size, device, dim=1):
    """Extract local shard for this rank using zigzag pattern.

    ZigZag splits into 2*world_size equal chunks. Sequence length must be
    divisible by 2*world_size, otherwise chunk() produces unequal sizes and
    the backward pass fails with a size mismatch error (issue #47).
    """
    seq_len = value.shape[dim]
    chunk_size = 2 * world_size

    if seq_len % chunk_size != 0:
        raise ValueError(
            f"Sequence length {seq_len} (dim={dim}) is not divisible by "
            f"2 * world_size = {chunk_size}. This causes a tensor size mismatch "
            f"in the backward pass (see issue #47). "
            f"Pad your sequences to a multiple of {chunk_size}, "
            f"or reduce the number of GPUs."
        )

    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat(
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
    )
    return local_value.to(device)


def prepare_zigzag_ring_attn_inputs(
    input_ids, position_ids, target_ids, rank, world_size, device
):
    local_input_ids = extract_local(
        input_ids,
        rank,
        world_size,
        device,
    )
    local_position_ids = extract_local(
        position_ids,
        rank,
        world_size,
        device,
    )
    if target_ids is not None:
        local_target_ids = extract_local(
            target_ids,
            rank,
            world_size,
            device,
        )
    else:
        local_target_ids = None
    return {
        "local_input_ids": local_input_ids,
        "local_position_ids": local_position_ids,
        "local_target_ids": local_target_ids,
    }
