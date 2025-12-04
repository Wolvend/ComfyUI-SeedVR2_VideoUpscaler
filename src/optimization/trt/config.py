"""
TensorRT configuration for SeedVR2.

Defines shape buckets and engine naming helpers. These buckets are kept small
to avoid exploding build time/VRAM. Anything outside these buckets should
fallback to the torch path.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class EngineProfile:
    name: str              # human-readable name, used in filenames
    batch: int
    height: int
    width: int


# Default fixed profiles (batch=1) covering common landscape/portrait.
DEFAULT_PROFILES: List[EngineProfile] = [
    EngineProfile("720p_landscape", batch=1, height=720, width=1280),
    EngineProfile("1080p_landscape", batch=1, height=1080, width=1920),
    EngineProfile("720p_portrait", batch=1, height=1280, width=720),
    EngineProfile("1080p_portrait", batch=1, height=1920, width=1080),
]

# Optional higher-res profiles; enable only if VRAM allows.
OPTIONAL_PROFILES: List[EngineProfile] = [
    EngineProfile("1440p_landscape", batch=1, height=1440, width=2560),
    EngineProfile("2160p_landscape", batch=1, height=2160, width=3840),
]


def engine_filename(model_name: str, profile: EngineProfile, precision: str, trt_version: str) -> str:
    """
    Build a deterministic engine filename.
    """
    return f"{model_name}_{precision}_{profile.batch}x3x{profile.height}x{profile.width}_{trt_version}.plan"
