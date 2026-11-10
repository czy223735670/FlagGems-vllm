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

"""Triton implementation of the mHC stream expansion operation."""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_expand_kernel(
    x,
    out,
    numel,
    H: tl.constexpr,
    N: tl.constexpr,
    STRIDE_M: tl.constexpr,
    STRIDE_H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = index < numel
    row = index // H
    col = index - row * H
    value = tl.load(x + row * STRIDE_M + col * STRIDE_H, mask=mask, other=0)
    for stream in tl.static_range(N):
        tl.store(out + (row * N + stream) * H + col, value, mask=mask)


def _launch_config(numel, dtype):
    if numel < (1 << 20):
        return 256, 2
    if numel < (1 << 21):
        return (1024, 4) if dtype == torch.bfloat16 else (512, 2)
    if numel < (1 << 22):
        return 1024, 4
    if numel < (1 << 24):
        return 4096, 8
    return 1024, 8


@lru_cache(maxsize=None)
def _supports_cta_clusters(device_index):
    return torch.cuda.get_device_capability(device_index)[0] >= 9


def _launch_hc_expand_kernel(x, out, hc_mult, kernel=_hc_expand_kernel):
    if not out.numel():
        return
    num_tokens, hidden_size = x.shape
    numel = num_tokens * hidden_size

    # Small inputs are launch-latency bound. Avoid the generic launch-config
    # path and use their fixed, pre-tuned configuration directly.
    if numel < (1 << 20):
        kernel[(triton.cdiv(numel, 256),)](
            x,
            out,
            numel,
            hidden_size,
            hc_mult,
            *x.stride(),
            BLOCK=256,
            num_warps=2,
            num_stages=2,
        )
        return

    block, num_warps = _launch_config(numel, x.dtype)
    launch_options = {
        "num_stages": 3,
    }
    if (
        numel >= (1 << 24)
        and x.device.type == "cuda"
        and torch.version.hip is None
    ):
        device_index = x.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        if _supports_cta_clusters(device_index):
            launch_options["num_ctas"] = 2
    kernel[(triton.cdiv(num_tokens * hidden_size, block),)](
        x,
        out,
        num_tokens * hidden_size,
        hidden_size,
        hc_mult,
        *x.stride(),
        BLOCK=block,
        num_warps=num_warps,
        **launch_options,
    )


def hc_expand(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """``x.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()``.

    Each input element is loaded once and written to all ``hc_mult`` output
    streams. ``x`` may be a non-contiguous rank-2 tensor.
    """
    m, h = x.shape
    out = x.new_empty(m, hc_mult, h)
    if out.numel():
        _launch_hc_expand_kernel(x, out, hc_mult)
    return out
