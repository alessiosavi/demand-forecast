"""Describe the device actually used by a run."""

import platform
from pathlib import Path

import torch


def describe_device(device) -> dict:
    device = torch.device(device)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        kind = "rocm" if torch.version.hip else "cuda"
    else:
        kind = "cpu"
        name = platform.processor() or platform.machine()
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text().splitlines():
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
    return {
        "device": kind,
        "name": name,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda_available": torch.cuda.is_available(),
    }
