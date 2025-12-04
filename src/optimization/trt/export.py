"""
ONNX export and TensorRT build helpers for SeedVR2.

These helpers are shape-bucketed and conservative: they export fixed-shape ONNX
for DiT and VAE, then optionally build FP16 TensorRT engines. If TensorRT/ONNX
are unavailable, they raise a friendly error.

NOTE: The current exports use stub input signatures and may need adjustment to
match the real model forward signatures before being production ready.
"""

import os
import torch
from typing import Optional

try:
    import onnx  # noqa: F401
    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt  # noqa: F401
    TRT_AVAILABLE = True
    TRT_VERSION = trt.__version__
except Exception:
    TRT_AVAILABLE = False
    TRT_VERSION = "unknown"

from .config import EngineProfile, engine_filename


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def export_stub_onnx(model: torch.nn.Module, profile: EngineProfile, onnx_dir: str, precision: str, prefix: str) -> str:
    """
    Placeholder exporter: exports a single-input single-output ONNX for a given model.

    NOTE: This uses a minimal dummy input and may need adjustment for real input signatures.
    """
    if not ONNX_AVAILABLE:
        raise RuntimeError("ONNX is not installed. Please install onnx to export models.")
    _ensure_dir(onnx_dir)
    model.eval()
    dtype = torch.float16 if precision == "fp16" else torch.float32
    dummy = torch.zeros(profile.batch, 3, profile.height, profile.width, device="cpu", dtype=dtype)
    onnx_path = os.path.join(onnx_dir, f"{prefix}_{engine_filename(model.__class__.__name__, profile, precision, 'onnx')}.onnx")
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        export_params=True,
        opset_version=19,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    return onnx_path


def build_trt_engine(onnx_path: str, profile: EngineProfile, precision: str, engine_dir: str) -> Optional[str]:
    """
    Build a TensorRT engine for a fixed profile. Returns the engine path.
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT not installed; cannot build engine.")
    _ensure_dir(engine_dir)
    engine_path = os.path.join(engine_dir, engine_filename(os.path.splitext(os.path.basename(onnx_path))[0], profile, precision, TRT_VERSION))
    if os.path.exists(engine_path):
        return engine_path
    import tensorrt as trt  # noqa: WPS433
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("Failed to parse ONNX for TensorRT.")
    config = builder.create_builder_config()
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    profile_cfg = builder.create_optimization_profile()
    min_shape = (profile.batch, 3, profile.height, profile.width)
    profile_cfg.set_shape("input", min_shape, min_shape, min_shape)
    config.add_optimization_profile(profile_cfg)
    engine = builder.build_engine(network, config)
    with open(engine_path, "wb") as f:
        f.write(bytearray(engine.serialize()))
    return engine_path
