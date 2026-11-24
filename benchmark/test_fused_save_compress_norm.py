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

import flaggems_vllm
from baseline.fused_save_compress_norm import (
    DTYPES,
    IS_INPLACE,
    OP_NAME,
    build_inputs,
    native,
)

from . import base

VLLM_REF = native()

pytestmark = [
    pytest.mark.fused_save_compress_norm,
    pytest.mark.skipif(VLLM_REF is None, reason="vLLM native op unavailable"),
]


class FusedSaveCompressNormBenchmark(base.Benchmark):
    op_name = OP_NAME

    def __init__(self):
        super().__init__(OP_NAME, VLLM_REF, DTYPES, is_inplace=IS_INPLACE)
        self.set_gems(flaggems_vllm.fused_save_compress_norm)
        self.shape_desc = "num_tokens, num_reqs, block_nums, compress_ratio"

    def get_input_iter(self, dtype):
        for num_tokens, num_reqs, block_nums, compress_ratio in self.shapes:
            binding = {
                "num_tokens": int(num_tokens),
                "num_reqs": int(num_reqs),
                "block_nums": int(block_nums),
                "compress_ratio": int(compress_ratio),
            }
            args, _ = build_inputs(binding, dtype, self.device)
            yield args


@pytest.mark.fused_save_compress_norm
def test_fused_save_compress_norm():
    FusedSaveCompressNormBenchmark().run()
