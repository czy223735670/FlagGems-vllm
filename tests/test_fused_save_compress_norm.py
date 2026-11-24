# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib

import pytest
import torch

import flaggems_vllm
from baseline.fused_save_compress_norm import (
    SHAPE_IDS,
    SHAPE_TRIPLES,
    build_inputs,
    torch_reference,
)
from flaggems_vllm.runtime import torch_device_fn

DEVICE = flaggems_vllm.device
GRAPH_TYPE = getattr(
    torch_device_fn,
    {"cuda": "CUDAGraph", "musa": "MUSAGraph", "npu": "NPUGraph"}.get(DEVICE),
    None,
)
GRAPH_SUPPORTED = (
    GRAPH_TYPE is not None
    and hasattr(torch_device_fn, "Stream")
    and hasattr(torch_device_fn, "stream")
    and hasattr(torch_device_fn, "graph")
)

pytestmark = [
    pytest.mark.fused_save_compress_norm,
    pytest.mark.skipif(
        torch_device_fn.device_count() < 1,
        reason="fused_save_compress_norm requires an accelerator",
    ),
]


def _make_case(
    compress_ratio,
    num_tokens,
    num_reqs,
    block_nums,
    *,
    invalid_slot_every=0,
):
    torch.manual_seed(42)
    args, _ = build_inputs(
        {
            "num_tokens": num_tokens,
            "num_reqs": num_reqs,
            "block_nums": block_nums,
            "compress_ratio": compress_ratio,
        },
        torch.float32,
        DEVICE,
        invalid_slot_every=invalid_slot_every,
    )
    args[-1].fill_(float("nan"))
    return args


def _run_and_compare(args):
    kv_before = args[0].clone()
    expected_state, expected_latent = torch_reference(*args)
    flaggems_vllm.fused_save_compress_norm(*args)
    _assert_results(args, expected_state, expected_latent, kv_before)


def _assert_results(args, expected_state, expected_latent, kv_before):
    kv_score, _, state, _, _, _, _, _, _, latent = args
    assert torch.equal(kv_score, kv_before)
    if state is not None:
        assert torch.equal(state, expected_state)
    assert torch.equal(latent.isnan(), expected_latent.isnan())
    valid = ~expected_latent.isnan()
    torch.testing.assert_close(
        latent[valid].float(),
        expected_latent[valid].float(),
        rtol=0.004,
        atol=1e-6,
    )


class _MemoryLibCache:
    """Small in-memory replacement for the persistent tuner caches."""

    def __init__(self):
        self.configs = {}
        self.benchmarks = {}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.configs.setdefault(key, {})
        return self.benchmarks.setdefault(key, {})


@pytest.mark.fused_save_compress_norm
@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize(
    "num_tokens,num_reqs,block_nums",
    SHAPE_TRIPLES,
    ids=SHAPE_IDS,
)
@torch.inference_mode()
def test_matches_pytorch_reference(
    compress_ratio,
    use_graph,
    num_tokens,
    num_reqs,
    block_nums,
):
    if use_graph and not GRAPH_SUPPORTED:
        pytest.skip(f"{DEVICE} graph capture API is unavailable")

    args = _make_case(compress_ratio, num_tokens, num_reqs, block_nums)
    kv_score, _, state, _, _, _, _, _, _, latent = args

    def run():
        flaggems_vllm.fused_save_compress_norm(*args)

    run()

    graph = None
    if use_graph:
        stream = torch_device_fn.Stream()
        stream.wait_stream(torch_device_fn.current_stream())
        with torch_device_fn.stream(stream):
            run()
        torch_device_fn.current_stream().wait_stream(stream)
        graph = GRAPH_TYPE()
        with torch_device_fn.graph(graph):
            run()

    for _ in range(3):
        kv_score.normal_()
        if state is not None:
            state.normal_()
        latent.fill_(float("nan"))
        kv_before = kv_score.clone()
        expected_state, expected_latent = torch_reference(*args)
        if graph is None:
            run()
        else:
            graph.replay()
        _assert_results(args, expected_state, expected_latent, kv_before)


@pytest.mark.parametrize("compress_ratio", [1, 2])
@torch.inference_mode()
def test_invalid_slots_match_reference(compress_ratio):
    args = _make_case(
        compress_ratio,
        64,
        16,
        32,
        invalid_slot_every=3,
    )
    _run_and_compare(args)


@torch.inference_mode()
def test_autotune_preserves_first_call_state(monkeypatch):
    """Every autotune candidate must leave the original CR2 history intact."""
    torch.manual_seed(17)
    args, _ = build_inputs(
        {
            "num_tokens": 77,
            "num_reqs": 7,
            "block_nums": 14,
            "compress_ratio": 2,
        },
        torch.float32,
        DEVICE,
    )
    kv_score, positions, state, slots, query_start_loc, _, _, _, _, latent = args
    capacity = state.shape[1]

    # Shift every request by one position so its first token closes a group
    # using the history row restored by the autotuner.
    for req in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[req].item())
        end = int(query_start_loc[req + 1].item())
        block = int(slots[start].item()) // capacity
        req_positions = positions[start:end] + 1
        positions[start:end] = req_positions
        slots[start:end] = block * capacity + req_positions % capacity

    history = state.clone()
    expected_state, expected_latent = torch_reference(*args)
    latent.fill_(float("nan"))
    kv_before = kv_score.clone()

    op_module = importlib.import_module(
        "flaggems_vllm.ops.fused_save_compress_norm"
    )
    libentry_module = importlib.import_module("flaggems_vllm.utils.libentry")
    tuner = op_module._fused_save_compress_norm_grouped_kernel.fn
    if len(tuner.configs) <= 1:
        pytest.skip("backend does not provide a multi-config autotune space")

    # Use fresh in-memory caches so this test always executes every candidate
    # and cannot pass by accidentally loading a persistent tuning result.
    memory_cache = _MemoryLibCache()
    monkeypatch.setattr(libentry_module, "libcache", memory_cache)
    monkeypatch.setattr(
        tuner,
        "cache",
        memory_cache[tuner.config_table_name],
    )

    original_bench = tuner._bench
    candidate_configs = []

    def checked_bench(*bench_args, config, **meta):
        assert torch.equal(state, history)
        result = original_bench(*bench_args, config=config, **meta)
        assert torch.equal(state, history)
        candidate_configs.append(config)
        return result

    monkeypatch.setattr(tuner, "_bench", checked_bench)

    assert torch.equal(state, history)
    flaggems_vllm.fused_save_compress_norm(*args)
    assert len(candidate_configs) == len(tuner.configs)
    assert len(set(candidate_configs)) == len(tuner.configs)
    assert len(tuner.cache) == 1
    _assert_results(args, expected_state, expected_latent, kv_before)

    # Restore the same history and verify the cached tuning path.
    tuned_calls = len(candidate_configs)
    state.copy_(history)
    latent.fill_(float("nan"))
    expected_state, expected_latent = torch_reference(*args)
    kv_before = kv_score.clone()
    flaggems_vllm.fused_save_compress_norm(*args)
    assert len(candidate_configs) == tuned_calls
    _assert_results(args, expected_state, expected_latent, kv_before)


@pytest.mark.parametrize("compress_ratio", [1, 2])
@torch.inference_mode()
def test_empty_mapping_is_noop(compress_ratio):
    num_tokens = 4
    kv_score = torch.randn(
        num_tokens,
        512 * compress_ratio,
        dtype=torch.float32,
        device=DEVICE,
    )
    positions = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    slots = torch.empty(0, dtype=torch.int64, device=DEVICE)
    weight = torch.randn(512, dtype=torch.bfloat16, device=DEVICE)
    latent = torch.zeros(num_tokens, 512, dtype=torch.bfloat16, device=DEVICE)
    latent_before = latent.clone()
    kv_before = kv_score.clone()
    if compress_ratio == 2:
        state = torch.randn(1, 8, 1024, dtype=torch.float32, device=DEVICE)
        state_before = state.clone()
        qsl = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)
        req_ids = torch.empty(0, dtype=torch.int32, device=DEVICE)
    else:
        state = state_before = qsl = req_ids = None

    flaggems_vllm.fused_save_compress_norm(
        kv_score,
        positions,
        state,
        slots,
        qsl,
        req_ids,
        weight,
        1e-20,
        compress_ratio,
        latent,
    )

    assert torch.equal(kv_score, kv_before)
    assert torch.equal(latent, latent_before)
    if state is not None:
        assert torch.equal(state, state_before)


@torch.inference_mode()
def test_rejects_unsupported_compress_ratio():
    with pytest.raises(AssertionError):
        flaggems_vllm.fused_save_compress_norm(
            torch.empty(1, 1536, dtype=torch.float32, device=DEVICE),
            torch.zeros(1, dtype=torch.int64, device=DEVICE),
            None,
            torch.zeros(1, dtype=torch.int64, device=DEVICE),
            None,
            None,
            torch.empty(512, dtype=torch.bfloat16, device=DEVICE),
            1e-20,
            3,
            torch.empty(1, 512, dtype=torch.bfloat16, device=DEVICE),
        )


@torch.inference_mode()
def test_baseline_input_contract():
    args, _ = build_inputs(
        {
            "num_tokens": 16,
            "num_reqs": 4,
            "block_nums": 8,
            "compress_ratio": 2,
        },
        torch.float32,
        DEVICE,
    )
    repeated_args, _ = build_inputs(
        {
            "num_tokens": 16,
            "num_reqs": 4,
            "block_nums": 8,
            "compress_ratio": 2,
        },
        torch.float32,
        DEVICE,
    )
    assert args[0].shape == (16, 1024)
    assert args[2].shape == (8, 8, 1024)
    query_start_loc = args[4]
    lengths = query_start_loc[1:] - query_start_loc[:-1]
    assert torch.equal(query_start_loc, repeated_args[4])
    assert torch.equal(args[5], repeated_args[5])
    assert torch.all(lengths > 0)
    assert int(lengths.sum()) == 16
    assert int(lengths.min()) < int(lengths.max())
