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

"""DeepSeek V4.1 state saving, KV compression, and RMSNorm."""

import inspect
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import device, torch_device_fn
from flaggems_vllm.utils import libentry


# The ring and raw rows a ratio-2 request program pools are not adjacent, so
# reading them as one tile means joining two pointers. ROCm's Triton fails to
# legalize `tt.join` on pointers, so there the two rows are loaded separately.
_JOIN_ROW_PTRS = device.vendor_name != "amd"
try:
    _LAUNCH_PDL_SUPPORTED = (
        "launch_pdl"
        in inspect.signature(triton.runtime.jit.JITFunction.run).parameters
    )
except Exception:
    _LAUNCH_PDL_SUPPORTED = False
_HEURISTICS = runtime.get_heuristic_config("fused_save_compress_norm")
_LAUNCH_KWARGS = (
    {"launch_pdl": False}
    if device.vendor_name == "nvidia" and _LAUNCH_PDL_SUPPORTED
    else {}
)


def _heuristic_args(num_tokens, num_reqs, compress_ratio, state_block):
    return {
        "num_tokens": num_tokens,
        "num_reqs": num_reqs,
        "COMPRESS_RATIO": compress_ratio,
        "STATE_BLOCK": state_block,
    }


def _launch_config(args):
    if not _HEURISTICS or "LAUNCH_CONFIG" not in _HEURISTICS:
        raise NotImplementedError(
            "No fused_save_compress_norm heuristic config for this backend"
        )
    config = _HEURISTICS["LAUNCH_CONFIG"](args)
    if len(config) != 3:
        raise ValueError(
            "fused_save_compress_norm heuristic must return "
            "(ITEMS_PER_PROGRAM, num_warps, num_stages)"
        )
    return config


def _use_group_state(args):
    if not _HEURISTICS or "USE_GROUP_STATE" not in _HEURISTICS:
        raise NotImplementedError(
            "No fused_save_compress_norm USE_GROUP_STATE heuristic for "
            "this backend"
        )
    use_group_state = _HEURISTICS["USE_GROUP_STATE"](args)
    if not isinstance(use_group_state, bool):
        raise ValueError(
            "fused_save_compress_norm USE_GROUP_STATE heuristic must return bool"
        )
    if args["COMPRESS_RATIO"] != 2 and use_group_state:
        raise ValueError(
            "fused_save_compress_norm USE_GROUP_STATE requires compress_ratio=2"
        )
    return use_group_state


def fused_save_compress_norm(
    kv_score: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    token_to_req_indices: torch.Tensor | None,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    compress_ratio: int,
    latent_out: torch.Tensor,
) -> None:
    """Pool each closed group into a normalized BF16 latent; save FP32 states.

    The latent feeds the main-cache insert and the indexer K path, which the
    attention layer schedules on separate streams.

    Ratio 2 keeps one ring block per request holding the open group's rows:
    position ``p`` lives in row ``p % capacity`` and ``slot_mapping`` encodes
    ``block * capacity + p % capacity``. The grid has one program per request
    followed by one per ``ITEMS_PER_PROGRAM`` packed groups. A request program
    handles the group that the chunk's first token closes with its
    predecessor's ring row. It also owns the chunk's first, last, and
    predecessor-collision tail rows. Group programs store the remaining tail
    rows from the pair they already load for pooling, so the raw rows are not
    read a second time. Tiny CR2 inputs use a request-only tail-copy path to
    avoid metadata overhead when the grid cannot fill the GPU. Ratio 1 has no
    ring and each program handles up to ``ITEMS_PER_PROGRAM`` valid tokens;
    ``slot_mapping`` then only marks valid tokens.

    Args:
        kv_score: FP32 [tokens, 512] for CR1, [tokens, 1024] for CR2.
        positions: Absolute positions of the packed request tokens.
        state_cache: Ring FP32 [blocks, capacity, 1024] KV/score states (CR2).
        slot_mapping: Ring slots (CR2) or main-cache slots (CR1).
        query_start_loc: [num_reqs + 1] token offsets of each request's chunk.
        token_to_req_indices: Request indices for the packed token rows.
        rms_norm_weight: BF16 [512] normalization weight.
        rms_norm_eps: RMSNorm epsilon.
        compress_ratio: Group size, either 1 or 2.
        latent_out: BF16 [tokens, 512], written only at valid group boundaries.
    """
    assert compress_ratio in (1, 2)
    assert kv_score.dtype == torch.float32
    assert kv_score.shape[1] == 512 * compress_ratio and kv_score.stride(1) == 1
    assert latent_out.shape == (kv_score.shape[0], 512)
    assert latent_out.is_contiguous() and latent_out.dtype == torch.bfloat16
    assert positions.is_contiguous() and slot_mapping.is_contiguous()
    # Rows stay 64-byte aligned, as the kernel's tl.multiple_of hint promises.
    assert kv_score.stride(0) % 16 == 0
    if compress_ratio == 2:
        assert state_cache is not None and query_start_loc is not None
        assert token_to_req_indices is not None
        assert query_start_loc.is_contiguous()
        assert token_to_req_indices.is_contiguous()
        assert state_cache.dtype == torch.float32
        assert state_cache.shape[2] == 1024 and state_cache.stride(2) == 1
        assert state_cache.stride(1) % 16 == 0
        state_stride, state_row_stride, state_block = (
            state_cache.stride(0),
            state_cache.stride(1),
            state_cache.shape[1],
        )
        num_reqs = query_start_loc.numel() - 1
    else:
        state_cache = query_start_loc = token_to_req_indices = None
        state_stride = state_row_stride = state_block = 1
        num_reqs = 0
    num_tokens = slot_mapping.numel()
    assert num_tokens <= min(kv_score.shape[0], positions.numel())
    if num_tokens == 0:
        return
    heuristic_args = _heuristic_args(
        num_tokens,
        num_reqs,
        compress_ratio,
        state_block,
    )
    items_per_program, num_warps, num_stages = _launch_config(heuristic_args)
    use_group_state = _use_group_state(heuristic_args)

    grid = num_reqs + triton.cdiv(
        num_tokens,
        compress_ratio * items_per_program,
    )

    device_context = (
        nullcontext()
        if torch_device_fn.current_device() == kv_score.device.index
        else torch_device_fn.device(kv_score.device)
    )
    with device_context:
        _fused_save_compress_norm_grouped_kernel[(grid,)](
            kv_score,
            positions,
            state_cache,
            slot_mapping,
            query_start_loc,
            token_to_req_indices,
            rms_norm_weight,
            latent_out,
            num_tokens,
            num_reqs,
            RAW_STRIDE=kv_score.stride(0),
            STATE_STRIDE=state_stride,
            STATE_ROW_STRIDE=state_row_stride,
            STATE_BLOCK=state_block,
            COMPRESS_RATIO=compress_ratio,
            EPS=rms_norm_eps,
            JOIN_ROW_PTRS=_JOIN_ROW_PTRS,
            USE_GROUP_STATE=use_group_state,
            ITEMS_PER_PROGRAM=items_per_program,
            num_warps=num_warps,
            num_stages=num_stages,
            **_LAUNCH_KWARGS,
        )


@libentry()
@triton.jit
def _fused_save_compress_norm_grouped_kernel(
    raw,
    positions,
    state,
    state_slots,
    query_start_loc,
    req_ids,
    norm_weight,
    latent,
    num_tokens,
    num_reqs,
    RAW_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_ROW_STRIDE: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    EPS: tl.constexpr,
    JOIN_ROW_PTRS: tl.constexpr,
    USE_GROUP_STATE: tl.constexpr,
    ITEMS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    d = tl.arange(0, 512)

    if COMPRESS_RATIO == 2:  # noqa: SIM102 (constexpr branch, then runtime)
        if pid < num_reqs:
            # Request program: the group closed by the chunk's first token,
            # then the ring tail store.
            start = tl.load(query_start_loc + pid)
            end = tl.load(query_start_loc + pid + 1)
            if start >= end:
                return
            state_slot = tl.load(state_slots + start)
            if state_slot < 0:
                return
            position = tl.load(positions + start)
            if (position + 1) % 2 == 0:
                ring = state + (state_slot // STATE_BLOCK).to(tl.int64) * STATE_STRIDE
                prev = ring + ((position - 1) % STATE_BLOCK) * STATE_ROW_STRIDE
                current = raw + start.to(tl.int64) * RAW_STRIDE
                pooled = _pool_with_previous(
                    prev,
                    current,
                    d,
                    JOIN_ROW_PTRS,
                )
                _store_latent(pooled, start, norm_weight, latent, d, EPS)

            if USE_GROUP_STATE:
                _store_request_tail(
                    raw,
                    state,
                    state_slots,
                    start,
                    end,
                    d,
                    RAW_STRIDE,
                    STATE_STRIDE,
                    STATE_ROW_STRIDE,
                    STATE_BLOCK,
                )
            else:
                num_rows = tl.minimum(end - start, STATE_BLOCK)
                state_d = tl.arange(0, 1024)
                for k in tl.range(0, num_rows):
                    token = end - num_rows + k
                    slot = tl.load(state_slots + token)
                    if slot >= 0:
                        row = _state_row_ptr(
                            state,
                            slot,
                            STATE_STRIDE,
                            STATE_ROW_STRIDE,
                            STATE_BLOCK,
                        )
                        src = raw + token.to(tl.int64) * RAW_STRIDE
                        tl.store(row + state_d, tl.load(src + state_d))
            return

    if COMPRESS_RATIO == 1:
        first_row = (pid * ITEMS_PER_PROGRAM).to(tl.int64)
        rows_idx = first_row + tl.arange(0, ITEMS_PER_PROGRAM)
        row_mask = rows_idx < num_tokens
        state_slots_vec = tl.load(
            state_slots + rows_idx,
            mask=row_mask,
            other=-1,
        )
        valid = row_mask & (state_slots_vec >= 0)
        raw_block = tl.make_block_ptr(
            base=raw + first_row * RAW_STRIDE,
            shape=(num_tokens, 512),
            strides=(RAW_STRIDE, 1),
            offsets=(0, 0),
            block_shape=(ITEMS_PER_PROGRAM, 512),
            order=(1, 0),
        )
        values = tl.load(
            raw_block,
            boundary_check=(0,),
            padding_option="zero",
        )
        values = tl.where(valid[:, None], values, 0.0)
        weight = tl.load(norm_weight + d).to(tl.float32)
        variance = tl.sum(values * values, axis=1) * (1.0 / 512.0)
        normed = values * tl.rsqrt(variance + EPS)[:, None] * weight[None, :]
        tl.store(
            latent + rows_idx[:, None] * 512 + d[None, :],
            normed.to(tl.bfloat16),
            mask=valid[:, None],
        )
    else:
        # Several pairs per program. The request program owns each chunk's
        # first token, last token, and predecessor-ring collision token. This
        # program stores the remaining tail rows from the pair it already
        # loaded for pooling.
        first_group = (
            (pid - num_reqs) * COMPRESS_RATIO * ITEMS_PER_PROGRAM
        ).to(tl.int64)
        for group in tl.static_range(0, ITEMS_PER_PROGRAM):
            base_t = first_group + group * COMPRESS_RATIO
            if base_t < num_tokens:
                t = base_t + (tl.load(positions + base_t) + 1).to(tl.int64) % 2
                if t < num_tokens:
                    req = tl.load(req_ids + t)
                    start = tl.load(query_start_loc + req).to(tl.int64)
                    state_slot = tl.load(state_slots + t)
                    if t != start:
                        rows = (
                            raw
                            + (t - 1 + tl.arange(0, 2)[:, None]) * RAW_STRIDE
                        )
                        kv = tl.load(rows + d[None, :])
                        score = tl.load(rows + 512 + d[None, :])
                        if state_slot >= 0:
                            pooled = tl.sum(kv * tl.softmax(score, 0), 0)
                            _store_latent(pooled, t, norm_weight, latent, d, EPS)
                        if USE_GROUP_STATE:
                            end = tl.load(query_start_loc + req + 1).to(tl.int64)
                            _store_group_state(
                                kv,
                                score,
                                t,
                                start,
                                end,
                                state,
                                state_slots,
                                d,
                                STATE_STRIDE,
                                STATE_ROW_STRIDE,
                                STATE_BLOCK,
                            )
@triton.jit
def _pool_with_previous(prev, current, d, JOIN_ROW_PTRS: tl.constexpr):
    if JOIN_ROW_PTRS:
        # Joining pointers hides row alignment from Triton; restate it.
        rows = tl.multiple_of(tl.join(prev, current)[:, None], (16, 16))
        kv = tl.load(rows + d[None, :])
        score = tl.load(rows + 512 + d[None, :])
        # Every lane has read the ring before the tail store below writes it.
        tl.debug_barrier()
        return tl.sum(kv * tl.softmax(score, 0), 0)

    kv_prev = tl.load(prev + d)
    score_prev = tl.load(prev + 512 + d)
    kv_current = tl.load(current + d)
    score_current = tl.load(current + 512 + d)
    # Every lane has read the ring before the tail store below writes it.
    tl.debug_barrier()
    peak = tl.maximum(score_prev, score_current)
    weight_prev = tl.exp(score_prev - peak)
    weight_current = tl.exp(score_current - peak)
    return (kv_prev * weight_prev + kv_current * weight_current) / (
        weight_prev + weight_current
    )


@triton.jit
def _state_row_ptr(
    state,
    slot,
    STATE_STRIDE: tl.constexpr,
    STATE_ROW_STRIDE: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
):
    return (
        state
        + (slot // STATE_BLOCK).to(tl.int64) * STATE_STRIDE
        + (slot % STATE_BLOCK) * STATE_ROW_STRIDE
    )


@triton.jit
def _collision_token(length, end, STATE_BLOCK: tl.constexpr):
    end_token = end - 1
    return tl.where(
        length >= STATE_BLOCK,
        end_token - length % STATE_BLOCK,
        -1,
    )


@triton.jit
def _store_request_tail(
    raw,
    state,
    state_slots,
    start,
    end,
    d,
    RAW_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_ROW_STRIDE: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
):
    length = end - start
    end_token = end - 1
    if length <= STATE_BLOCK:
        _copy_state_row(
            raw,
            state,
            state_slots,
            start,
            d,
            RAW_STRIDE,
            STATE_STRIDE,
            STATE_ROW_STRIDE,
            STATE_BLOCK,
        )
    if end_token != start:
        _copy_state_row(
            raw,
            state,
            state_slots,
            end_token,
            d,
            RAW_STRIDE,
            STATE_STRIDE,
            STATE_ROW_STRIDE,
            STATE_BLOCK,
        )
    if length >= STATE_BLOCK:
        collision = _collision_token(length, end, STATE_BLOCK)
        if collision != start and collision != end_token:
            _copy_state_row(
                raw,
                state,
                state_slots,
                collision,
                d,
                RAW_STRIDE,
                STATE_STRIDE,
                STATE_ROW_STRIDE,
                STATE_BLOCK,
            )


@triton.jit
def _store_group_state(
    kv,
    score,
    t,
    start,
    end,
    state,
    state_slots,
    d,
    STATE_STRIDE: tl.constexpr,
    STATE_ROW_STRIDE: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
):
    length = end - start
    tail_start = end - tl.minimum(length, STATE_BLOCK)
    pair_tokens = t - 1 + tl.arange(0, 2).to(tl.int64)
    pair_slots = tl.load(state_slots + pair_tokens)
    start_slot = tl.load(state_slots + start)
    end_token = end - 1
    collision = _collision_token(length, end, STATE_BLOCK)
    store_pair = (
        (pair_tokens >= tail_start)
        & (pair_tokens < end)
        & (pair_slots >= 0)
        & (start_slot >= 0)
        & (pair_tokens != start)
        & (pair_tokens != end_token)
        & (pair_tokens != collision)
    )
    state_rows = _state_row_ptr(
        state,
        pair_slots,
        STATE_STRIDE,
        STATE_ROW_STRIDE,
        STATE_BLOCK,
    )[:, None]
    tl.store(
        state_rows + d[None, :],
        kv,
        mask=store_pair[:, None],
    )
    tl.store(
        state_rows + 512 + d[None, :],
        score,
        mask=store_pair[:, None],
    )


@triton.jit
def _copy_state_row(
    raw,
    state,
    state_slots,
    token,
    d,
    RAW_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_ROW_STRIDE: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
):
    slot = tl.load(state_slots + token)
    if slot >= 0:
        row = _state_row_ptr(
            state,
            slot,
            STATE_STRIDE,
            STATE_ROW_STRIDE,
            STATE_BLOCK,
        )
        src = raw + token.to(tl.int64) * RAW_STRIDE
        tl.store(row + d, tl.load(src + d))
        tl.store(row + 512 + d, tl.load(src + 512 + d))


@triton.jit
def _store_latent(pooled, t, norm_weight, latent, d, EPS: tl.constexpr):
    weight = tl.load(norm_weight + d).to(tl.float32)
    variance = tl.sum(pooled * pooled, 0) / 512
    normed = pooled * tl.rsqrt(variance + EPS) * weight
    tl.store(latent + t.to(tl.int64) * 512 + d, normed.to(tl.bfloat16))
