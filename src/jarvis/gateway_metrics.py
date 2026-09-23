"""
gateway_metrics.py
Provides memory and timing utilities for the Jarvis Smart Gateway.
Import and use in gateway.py when trace is requested.
"""

import os
import time
import psutil


def get_memory_mb() -> float:
    """Current process RSS memory in MB."""
    try:
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / 1024 / 1024, 1)
    except Exception:
        return 0.0


def get_system_memory() -> dict:
    """System-wide memory stats."""
    try:
        vm = psutil.virtual_memory()
        return {
            "total_gb":     round(vm.total     / 1024**3, 1),
            "available_gb": round(vm.available / 1024**3, 1),
            "used_gb":      round(vm.used      / 1024**3, 1),
            "percent":      vm.percent,
        }
    except Exception:
        return {}


class StepTimer:
    """
    Tracks timing and memory for each pipeline step.

    Usage:
        tracker = StepTimer()
        tracker.start("enhance")
        ... do work ...
        tracker.end("enhance")
        print(tracker.summary())
    """

    def __init__(self):
        self.steps: list[dict] = []
        self._current: dict = {}
        self.mem_start = get_memory_mb()
        self.wall_start = time.time()

    def start(self, name: str):
        self._current = {
            "name":       name,
            "t_start":    time.time(),
            "mem_before": get_memory_mb(),
        }

    def end(self, name: str):
        if self._current.get("name") != name:
            return
        t_end   = time.time()
        mem_end = get_memory_mb()
        self.steps.append({
            "name":       name,
            "duration_s": round(t_end - self._current["t_start"], 2),
            "mem_before": self._current["mem_before"],
            "mem_after":  mem_end,
            "mem_delta":  round(mem_end - self._current["mem_before"], 1),
        })
        self._current = {}

    def total_time(self) -> float:
        return round(time.time() - self.wall_start, 1)

    def total_mem_delta(self) -> float:
        return round(get_memory_mb() - self.mem_start, 1)

    def summary(self) -> str:
        """Returns a markdown-formatted trace block."""
        sys_mem = get_system_memory()
        lines = [
            "**Performance trace:**",
            f"- Total time: `{self.total_time()}s`",
            f"- Process memory delta: `{self.total_mem_delta()} MB`",
            f"- System RAM: `{sys_mem.get('used_gb')} GB used / {sys_mem.get('total_gb')} GB total ({sys_mem.get('percent')}%)`",
            "",
            "| Step | Time | Mem Before | Mem After | Delta |",
            "|------|------|------------|-----------|-------|",
        ]
        for s in self.steps:
            lines.append(
                f"| {s['name']} | {s['duration_s']}s "
                f"| {s['mem_before']} MB | {s['mem_after']} MB "
                f"| {'+' if s['mem_delta'] >= 0 else ''}{s['mem_delta']} MB |"
            )
        return "\n".join(lines)
