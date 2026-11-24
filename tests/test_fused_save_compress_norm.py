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


@pytest.mark.parametrize(
    "lengths,start_positions",
    [
        ((7, 8, 8), (7, 8, 13)),
        ((9, 10, 11), (8, 9, 10)),
    ],
)
@torch.inference_mode()
def test_ratio2_fused_state_store_ring_edges(lengths, start_positions):
    """Exercise endpoint ownership and predecessor-slot collisions."""
    torch.manual_seed(23)
    capacity = 8
    actual_tokens = sum(lengths)
    positions = torch.empty(actual_tokens, dtype=torch.int64, device=DEVICE)
    slots = torch.empty_like(positions)
    query_start_loc = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=DEVICE,
    )
    req_ids = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.int32, device=DEVICE),
        torch.tensor(lengths, dtype=torch.int64, device=DEVICE),
    )
    blocks = (2, 0, 1)
    cursor = 0
    for req, (length, position) in enumerate(zip(lengths, start_positions)):
        rows = slice(cursor, cursor + length)
        req_positions = torch.arange(
            position,
            position + length,
            dtype=torch.int64,
            device=DEVICE,
        )
        positions[rows] = req_positions
        slots[rows] = blocks[req] * capacity + req_positions % capacity
        cursor += length

    state = torch.randn(
        3,
        capacity,
        1024,
        dtype=torch.float32,
        device=DEVICE,
    )
    args = (
        torch.randn(actual_tokens, 1024, dtype=torch.float32, device=DEVICE),
        positions,
        state,
        slots,
        query_start_loc,
        req_ids,
        torch.randn(512, dtype=torch.bfloat16, device=DEVICE),
        1e-20,
        2,
        torch.empty(actual_tokens, 512, dtype=torch.bfloat16, device=DEVICE),
    )
    args[-1].fill_(float("nan"))
    _run_and_compare(args)


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
