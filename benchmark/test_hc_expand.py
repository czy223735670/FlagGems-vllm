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

import statistics
from dataclasses import asdict

import pytest
import torch

import flaggems_vllm
from baseline.hc_expand import (
    DTYPES,
    OP_NAME,
    build_inputs,
    native,
)

from . import base, conftest as cfg
from .consts import BenchmarkMetrics, BenchmarkResult

VLLM_NATIVE = native()
MEASUREMENT_ROUNDS = 4

pytestmark = [
    pytest.mark.hc_expand,
    pytest.mark.skipif(
        VLLM_NATIVE is None,
        reason="vLLM native hc_expand unavailable",
    ),
]


def _assert_bits_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if actual.numel():
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


class HCExpandBenchmark(base.Benchmark):
    def __init__(self, dtype):
        super().__init__(OP_NAME, VLLM_NATIVE, [dtype], is_inplace=False)
        self.set_gems(flaggems_vllm.hc_expand)

    def get_input_iter(self, dtype):
        for num_tokens, hidden_size, hc_mult in self.shapes:
            binding = {
                "num_tokens": num_tokens,
                "hidden_size": hidden_size,
                "hc_mult": hc_mult,
            }
            args, _ = build_inputs(
                binding,
                dtype,
                self.device,
                special_values=False,
            )
            yield args

    def _measure_pair(self, base_fn, gems_fn):
        base_samples = []
        gems_samples = []
        for round_index in range(MEASUREMENT_ROUNDS):
            order = (
                (("base", base_fn), ("gems", gems_fn))
                if round_index % 2 == 0
                else (("gems", gems_fn), ("base", base_fn))
            )
            samples = {}
            for name, fn in order:
                samples[name] = float(self.get_latency(fn))
            base_samples.append(samples["base"])
            gems_samples.append(samples["gems"])

        base_latency = statistics.median(base_samples)
        gems_latency = statistics.median(gems_samples)
        return base_latency, gems_latency, base_latency / gems_latency

    def _metric(self, x, hc_mult, base_fn, gems_fn):
        base_latency, gems_latency, speedup = self._measure_pair(base_fn, gems_fn)
        payload_bytes = x.numel() * (1 + hc_mult) * x.element_size()
        gbps_base = (
            payload_bytes / (base_latency * 1e-3) / 1e9
            if base_latency
            else None
        )
        gbps = (
            payload_bytes / (gems_latency * 1e-3) / 1e9
            if gems_latency
            else None
        )
        return BenchmarkMetrics(
            shape_detail=["wrapper", "contiguous", tuple(x.shape), hc_mult],
            category="wrapper",
            latency_base=base_latency,
            latency=gems_latency,
            speedup=speedup,
            gbps_base=gbps_base,
            gbps=gbps,
        )

    def run(self):
        if cfg.Config.query:
            return super().run()

        self.init_user_config()
        for dtype in self.to_bench_dtypes:
            metrics = []
            for args in self.get_input_iter(dtype):
                x, hc_mult = args

                wrapper_base = lambda: self.torch_op(x, hc_mult)
                wrapper_gems = lambda: self.gems_op(x, hc_mult)
                _assert_bits_equal(wrapper_gems(), wrapper_base())
                metrics.append(
                    self._metric(
                        x,
                        hc_mult,
                        wrapper_base,
                        wrapper_gems,
                    )
                )

            result = BenchmarkResult(
                level=cfg.Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=cfg.Config.mode.value,
                result=metrics,
            )
            print(result)
            cfg.update_result(self.op_name, asdict(result))
            cfg.emit_record_logger(result.to_json())


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_hc_expand(dtype):
    HCExpandBenchmark(dtype).run()
