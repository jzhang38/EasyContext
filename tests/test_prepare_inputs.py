import importlib.util
import pytest
import torch

# Load prepare_inputs.py directly, bypassing easy_context/__init__.py
# which pulls in triton/flash-attn dependencies not available on Mac dev machines
_spec = importlib.util.spec_from_file_location(
    "prepare_inputs",
    "/Users/bhavyashah09/open_source/docs/contributions/easycontext/EasyContext"
    "/easy_context/zigzag_ring_attn/prepare_inputs.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

extract_local = _module.extract_local
prepare_zigzag_ring_attn_inputs = _module.prepare_zigzag_ring_attn_inputs


def make_test_inputs(seq_len, batch_size=2, vocab_size=1000):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    target_ids = torch.randint(-100, vocab_size, (batch_size, seq_len))
    return input_ids, position_ids, target_ids


class TestExtractLocal:

    def test_valid_sequence_length(self):
        world_size = 4
        seq_len = 256  # 256 % 8 = 0 ✓
        input_ids, _, _ = make_test_inputs(seq_len)
        result = extract_local(input_ids, rank=0, world_size=world_size, device="cpu")
        assert result.shape[0] == input_ids.shape[0]
        assert result.shape[1] == seq_len // world_size

    def test_invalid_sequence_length_raises(self):
        world_size = 4
        seq_len = 100  # 100 % 8 = 4 ✗
        input_ids, _, _ = make_test_inputs(seq_len)
        with pytest.raises(ValueError, match="not divisible by 2 \\* world_size"):
            extract_local(input_ids, rank=0, world_size=world_size, device="cpu")

    def test_error_message_is_helpful(self):
        world_size = 2
        seq_len = 7  # next valid = 8
        input_ids, _, _ = make_test_inputs(seq_len)
        with pytest.raises(ValueError) as exc_info:
            extract_local(input_ids, rank=0, world_size=world_size, device="cpu")
        assert "multiple of 4" in str(exc_info.value)
        assert "issue #47" in str(exc_info.value)

    def test_single_gpu_requires_even_length(self):
        input_ids, _, _ = make_test_inputs(7)  # odd
        with pytest.raises(ValueError, match="not divisible"):
            extract_local(input_ids, rank=0, world_size=1, device="cpu")

    def test_zigzag_pattern_assignment(self):
        world_size = 2
        input_ids = torch.arange(8).unsqueeze(0).expand(1, -1)
        # chunks: [0,1], [2,3], [4,5], [6,7]
        # rank=0 → chunks[0] + chunks[3] = [0,1,6,7]
        result = extract_local(input_ids, rank=0, world_size=world_size, device="cpu")
        assert torch.equal(result, torch.tensor([[0, 1, 6, 7]]))
        # rank=1 → chunks[1] + chunks[2] = [2,3,4,5]
        result = extract_local(input_ids, rank=1, world_size=world_size, device="cpu")
        assert torch.equal(result, torch.tensor([[2, 3, 4, 5]]))


class TestPrepareZigzagRingAttnInputs:

    def test_valid_inputs_returns_dict(self):
        world_size = 2
        input_ids, position_ids, target_ids = make_test_inputs(64)
        result = prepare_zigzag_ring_attn_inputs(
            input_ids, position_ids, target_ids, rank=0, world_size=world_size, device="cpu"
        )
        assert isinstance(result, dict)
        assert "local_input_ids" in result
        assert "local_position_ids" in result
        assert "local_target_ids" in result
        assert result["local_input_ids"] is not None
        assert result["local_target_ids"] is not None

    def test_none_target_ids_handled(self):
        world_size = 2
        input_ids, position_ids, _ = make_test_inputs(64)
        result = prepare_zigzag_ring_attn_inputs(
            input_ids, position_ids, None, rank=0, world_size=world_size, device="cpu"
        )
        assert result["local_target_ids"] is None
        assert result["local_input_ids"] is not None

    def test_invalid_length_raises(self):
        world_size = 2
        input_ids, position_ids, target_ids = make_test_inputs(7)
        with pytest.raises(ValueError, match="not divisible"):
            prepare_zigzag_ring_attn_inputs(
                input_ids, position_ids, target_ids, rank=0, world_size=world_size, device="cpu"
            )

    def test_mismatched_target_ids_seq_len_raises(self):
        world_size = 2
        seq_len = 8
        input_ids, position_ids, _ = make_test_inputs(seq_len)
        # target_ids has a different (invalid) seq_len — should fail when extract_local validates it
        target_ids = torch.randint(0, 1000, (2, 10))  # seq_len=10, 10 % 4 != 0
        with pytest.raises(ValueError, match="not divisible"):
            prepare_zigzag_ring_attn_inputs(
                input_ids, position_ids, target_ids, rank=0, world_size=world_size, device="cpu"
            )
