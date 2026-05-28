# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight per-phase timing for speculative decoding.

Captures fine-grained wall-clock timing of the target/draft forward passes,
the autoregressive draft loop, input prep and sampling, and logs periodic
per-phase means from the worker process.

This is meant for debugging (e.g. understanding where time goes when running a
separate draft model). It is fully gated by ``VLLM_SPEC_DECODE_TIMING`` and adds
zero overhead when disabled (``time()`` returns a ``nullcontext``). When enabled,
the only host<->device synchronization happens once per logging interval, not
per step: CUDA events are recorded and buffered, then reduced together at flush.

Modeled on the aggregation/logging structure of
:class:`vllm.v1.spec_decode.metrics.SpecDecodingLogging`.
"""

import contextlib
from collections import defaultdict
from contextlib import AbstractContextManager

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


class SpecTimer:
    """Env-gated CUDA-event / perf_counter phase timer + periodic logger."""

    def __init__(self):
        self.enabled: bool = envs.VLLM_SPEC_DECODE_TIMING
        self.interval: int = max(1, envs.VLLM_SPEC_DECODE_TIMING_INTERVAL)
        self.detail: bool = envs.VLLM_SPEC_DECODE_TIMING_DETAIL
        # CUDA events are only usable when CUDA is available; otherwise fall
        # back to perf_counter timing for every phase.
        self._use_cuda: bool = self.enabled and torch.cuda.is_available()
        self._step_count: int = 0
        self._reset()

    def _reset(self) -> None:
        # phase name -> list of (start_event, end_event) recorded this interval
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
            defaultdict(list)
        )
        # phase name -> list of elapsed milliseconds (perf_counter path)
        self._cpu_ms: dict[str, list[float]] = defaultdict(list)

    def time(self, name: str, *, cuda: bool = True) -> AbstractContextManager:
        """Time a phase. Returns a no-op context manager when disabled.

        Args:
            name: phase label, also used as the profiler scope name.
            cuda: time GPU work with CUDA events (default). Set ``False`` for
                predominantly CPU phases (e.g. input prep) to use perf_counter.
        """
        if not self.enabled:
            return contextlib.nullcontext()
        if cuda and self._use_cuda:
            return self._cuda_timer(name)
        return self._cpu_timer(name)

    @contextlib.contextmanager
    def _cuda_timer(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._events[name].append((start, end))

    @contextlib.contextmanager
    def _cpu_timer(self, name: str):
        from time import perf_counter

        start = perf_counter()
        try:
            yield
        finally:
            self._cpu_ms[name].append((perf_counter() - start) * 1e3)

    def step(self) -> None:
        """Mark the end of a model step and log on the configured interval."""
        if not self.enabled:
            return
        self._step_count += 1
        if self._step_count % self.interval == 0:
            self._log()

    def _log(self) -> None:
        if not self._events and not self._cpu_ms:
            return

        # Reduce all buffered CUDA event pairs with a single synchronize.
        durations: dict[str, list[float]] = defaultdict(list)
        if self._events:
            torch.cuda.synchronize()
            for name, pairs in self._events.items():
                durations[name].extend(s.elapsed_time(e) for s, e in pairs)
        for name, ms_list in self._cpu_ms.items():
            durations[name].extend(ms_list)

        # Each TP/PP rank runs its own timer in its own process; prefix every
        # line with the global rank so interleaved per-rank blocks stay legible
        # (and per-rank differences, e.g. stragglers, are visible).
        prefix = f"[rank {_global_rank()}] "
        lines = [
            f"{prefix}SpecDecode timing over {self.interval} steps (mean ms, count):"
        ]
        for name, ms_list in durations.items():
            arr = np.asarray(ms_list)
            lines.append(
                f"{prefix}  {name:<22} mean={arr.mean():7.3f}  "
                f"total={arr.sum():9.2f}  n={arr.size}"
            )
        logger.info("\n".join(lines))
        self._reset()


def _global_rank() -> int:
    """Best-effort global distributed rank; 0 when not running distributed."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


_SPEC_TIMER: SpecTimer | None = None


def get_spec_timer() -> SpecTimer:
    """Return the process-wide :class:`SpecTimer` singleton."""
    global _SPEC_TIMER
    if _SPEC_TIMER is None:
        _SPEC_TIMER = SpecTimer()
    return _SPEC_TIMER
