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

"""Shared workload contract and Torch reference for fused_save_compress_norm."""

import torch

OP_NAME = "fused_save_compress_norm"
DTYPES = [torch.float32]
IS_INPLACE = True

BASE_SHAPES = [
    (1, 1),
    (8, 8),
    (512, 128),
    (1024, 64),
    (2048, 512),
    (4096, 1),
    (8192, 64),
    (16384, 256),
    (32768, 512),
]

SHAPE_TRIPLES = [
    (num_tokens, num_reqs, 2 * num_reqs) for num_tokens, num_reqs in BASE_SHAPES
]

SHAPE_IDS = [
    f"t{num_tokens}_r{num_reqs}_b{block_nums}"
    for num_tokens, num_reqs, block_nums in SHAPE_TRIPLES
]

def native():
    """Load the vLLM Python operator lazily as the performance baseline."""
    try:
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
            fused_save_compress_norm,
        )
    except Exception:
        return None
    return fused_save_compress_norm


def _pool_rows(rows):
    stacked = torch.stack(rows)
    kv = stacked[:, :512]
    score = stacked[:, 512:]
    return (kv * torch.softmax(score, dim=0)).sum(dim=0)


def _normalize(pooled, weight, eps):
    variance = pooled.square().mean()
    return pooled * torch.rsqrt(variance + eps) * weight.float()


def torch_reference(
    kv_score,
    positions,
    state_cache,
    slot_mapping,
    query_start_loc,
    token_to_req_indices,
    rms_norm_weight,
    rms_norm_eps,
    compress_ratio,
    latent_out,
):
    """Reference implementation of ``fused_save_compress_norm``.

    Returns ``(expected_state_cache, expected_latent_out)``. ``latent_out`` is
    used only for its shape, dtype and device; the supplied tensors are not
    modified.
    """
    expected_latent = torch.full(
        latent_out.shape,
        float("nan"),
        dtype=torch.float32,
        device=latent_out.device,
    )
    expected_state = state_cache.clone() if state_cache is not None else None
    num_slots = slot_mapping.numel()

    if compress_ratio == 1:
        for token in range(num_slots):
            if slot_mapping[token].item() < 0:
                continue
            pooled = kv_score[token, :512]
            expected_latent[token] = _normalize(pooled, rms_norm_weight, rms_norm_eps)
        return expected_state, expected_latent

    history = state_cache
    capacity = history.shape[1]
    num_reqs = query_start_loc.numel() - 1
    actual_tokens = int(query_start_loc[-1].item())

    # Request programs close a group crossing the chunk boundary, then retain
    # only the newest capacity rows in each request's ring block.
    for req in range(num_reqs):
        start = int(query_start_loc[req].item())
        end = int(query_start_loc[req + 1].item())
        if start >= end or slot_mapping[start].item() < 0:
            continue

        position = int(positions[start].item())
        if (position + 1) % 2 == 0:
            current_slot = int(slot_mapping[start].item())
            block = current_slot // capacity
            previous = history[block, (position - 1) % capacity]
            pooled = _pool_rows([previous, kv_score[start]])
            expected_latent[start] = _normalize(pooled, rms_norm_weight, rms_norm_eps)

        num_rows = min(end - start, capacity)
        for token in range(end - num_rows, end):
            slot = int(slot_mapping[token].item())
            if slot < 0:
                continue
            block = slot // capacity
            offset = slot % capacity
            expected_state[block, offset] = kv_score[token]

    # Group programs handle pairs wholly inside the current chunk. The first
    # token of each request chunk belongs to its request program.
    for base in range(0, actual_tokens, compress_ratio):
        token = base + (int(positions[base].item()) + 1) % compress_ratio
        if token >= actual_tokens or slot_mapping[token].item() < 0:
            continue
        req = int(token_to_req_indices[token].item())
        if token == int(query_start_loc[req].item()):
            continue
        pooled = _pool_rows([kv_score[token - 1], kv_score[token]])
        expected_latent[token] = _normalize(pooled, rms_norm_weight, rms_norm_eps)

    return expected_state, expected_latent


def grid():
    return [
        {
            "num_tokens": num_tokens,
            "num_reqs": num_reqs,
            "block_nums": block_nums,
            "compress_ratio": compress_ratio,
        }
        for num_tokens, num_reqs, block_nums in SHAPE_TRIPLES
        for compress_ratio in (1, 2)
    ]


def _packed_requests(num_tokens, num_reqs):
    # Uniformly sample a positive composition of T with R parts. Deriving the
    # seed from T/R keeps the functional and benchmark workloads reproducible.
    if num_reqs == 1:
        boundaries = [0, num_tokens]
    else:
        generator = torch.Generator(device="cpu").manual_seed(
            num_tokens * 1009 + num_reqs * 9176
        )
        cuts = torch.randperm(num_tokens - 1, generator=generator)[
            : num_reqs - 1
        ]
        boundaries = [
            0,
            *(int(cut) + 1 for cut in cuts.sort().values),
            num_tokens,
        ]
    lengths = [
        boundaries[index + 1] - boundaries[index]
        for index in range(num_reqs)
    ]

    query_start_loc = torch.tensor(boundaries, dtype=torch.int32, device="cpu")
    token_to_req = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int32),
        torch.tensor(lengths, dtype=torch.int64),
    )
    return lengths, query_start_loc, token_to_req


def build_inputs(binding, dtype, device, *, invalid_slot_every=0):
    """Create positional arguments shared by functional and performance tests."""
    del dtype  # kv_score is fixed FP32 by the vLLM operator contract.
    num_tokens = binding["num_tokens"]
    num_reqs = binding["num_reqs"]
    block_nums = binding.get("block_nums", num_reqs)
    compress_ratio = binding["compress_ratio"]
    capacity = binding.get("capacity", 8)
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")
    if not 1 <= num_reqs <= num_tokens:
        raise ValueError("num_reqs must be in [1, num_tokens]")
    if block_nums < num_reqs:
        raise ValueError("block_nums must be at least num_reqs")

    lengths, query_start_loc_cpu, token_to_req_cpu = _packed_requests(
        num_tokens,
        num_reqs,
    )
    query_start_loc = query_start_loc_cpu.to(device)
    token_to_req = token_to_req_cpu.to(device)
    positions = torch.empty(num_tokens, dtype=torch.int64, device=device)
    slot_mapping = torch.empty(num_tokens, dtype=torch.int64, device=device)
    request_blocks = torch.randperm(block_nums, device=device)[:num_reqs]

    cursor = 0
    for req, length in enumerate(lengths):
        req_positions = torch.arange(
            length,
            dtype=torch.int64,
            device=device,
        )
        rows = slice(cursor, cursor + length)
        positions[rows] = req_positions
        slot_mapping[rows] = request_blocks[req] * capacity + req_positions % capacity
        cursor += length

    if invalid_slot_every > 0:
        slot_mapping[::invalid_slot_every] = -1

    kv_score = torch.randn(
        num_tokens,
        512 * compress_ratio,
        dtype=torch.float32,
        device=device,
    )
    rms_norm_weight = torch.randn(512, dtype=torch.bfloat16, device=device)
    latent_out = torch.empty(
        num_tokens,
        512,
        dtype=torch.bfloat16,
        device=device,
    )

    if compress_ratio == 2:
        state_cache = torch.randn(
            block_nums,
            capacity,
            1024,
            dtype=torch.float32,
            device=device,
        )
        args = (
            kv_score,
            positions,
            state_cache,
            slot_mapping,
            query_start_loc,
            token_to_req,
            rms_norm_weight,
            1e-20,
            compress_ratio,
            latent_out,
        )
    else:
        args = (
            kv_score,
            positions,
            None,
            slot_mapping,
            None,
            None,
            rms_norm_weight,
            1e-20,
            compress_ratio,
            latent_out,
        )
    return args, {}


def key_shape(binding):
    return (
        f"tokens{binding['num_tokens']}_reqs{binding['num_reqs']}_"
        f"blocks{binding.get('block_nums', binding['num_reqs'])}_"
        f"cr{binding['compress_ratio']}"
    )


def config(binding, dtype):
    compress_ratio = binding["compress_ratio"]
    num_tokens = binding["num_tokens"]
    num_reqs = binding["num_reqs"]
    block_nums = binding.get("block_nums", num_reqs)
    outputs = {
        "latent_out": {
            "shape": [num_tokens, 512],
            "dtype": "torch.bfloat16",
        }
    }
    if compress_ratio == 2:
        outputs["state_cache"] = {
            "shape": [block_nums, binding.get("capacity", 8), 1024],
            "dtype": "torch.float32",
        }
    return {
        "inputs": {
            "kv_score": {
                "shape": [num_tokens, 512 * compress_ratio],
                "dtype": str(dtype),
            },
            "positions": {"shape": [num_tokens], "dtype": "torch.int64"},
            "slot_mapping": {"shape": [num_tokens], "dtype": "torch.int64"},
            "rms_norm_weight": {"shape": [512], "dtype": "torch.bfloat16"},
            "rms_norm_eps": {"scalar": 1e-20},
            "compress_ratio": {"scalar": compress_ratio},
        },
        "outputs": outputs,
        "dims": dict(binding),
    }
