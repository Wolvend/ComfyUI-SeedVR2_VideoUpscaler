"""
SeedVR2 TensorRT export/build helper (experimental scaffold).

Exports stub ONNX and builds FP16 TensorRT engines for fixed profiles. Engines
are stored under models/SEEDVR2/trt/. Shapes outside these profiles should
fallback to the torch path at runtime.

Usage:
  python scripts/build_trt.py --model_dir ./models/SEEDVR2 --precision fp16
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.optimization.trt.config import DEFAULT_PROFILES  # noqa: E402
from src.optimization.trt.export import export_stub_onnx, build_trt_engine  # noqa: E402
from src.core.model_loader import prepare_model_structure  # noqa: E402
from src.utils.constants import SEEDVR2_FOLDER_NAME  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT engines for SeedVR2 fixed profiles (experimental).")
    parser.add_argument("--model_dir", type=str, default=None, help="Model directory (default: ./models/SEEDVR2)")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"], help="Engine precision")
    parser.add_argument("--profiles", type=str, nargs="*", default=None, help="Profile names to build (default: all DEFAULT_PROFILES)")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_dir or os.path.join(".", "models", SEEDVR2_FOLDER_NAME)

    # Load model structures (weights must exist in model_dir)
    dit, vae = prepare_model_structure(model_dir=model_dir, debug=None)

    selected_profiles = [p for p in DEFAULT_PROFILES if (not args.profiles) or p.name in args.profiles]
    if not selected_profiles:
        print("No matching profiles to build. Exiting.")
        return

    onnx_dir = os.path.join(model_dir, "onnx")
    engine_dir = os.path.join(model_dir, "trt")

    for profile in selected_profiles:
        print(f"Exporting ONNX for profile {profile.name}...")
        try:
            dit_onnx = export_stub_onnx(dit, profile, onnx_dir, args.precision, prefix="dit")
            vae_onnx = export_stub_onnx(vae, profile, onnx_dir, args.precision, prefix="vae")
        except Exception as e:  # noqa: BLE001
            print(f"ONNX export failed for {profile.name}: {e}")
            continue

        print(f"Building TensorRT engines for {profile.name}...")
        for onnx_path in (dit_onnx, vae_onnx):
            try:
                engine_path = build_trt_engine(onnx_path, profile, args.precision, engine_dir)
                print(f"Built engine: {engine_path}")
            except Exception as e:  # noqa: BLE001
                print(f"TRT build failed for {onnx_path}: {e}")


if __name__ == "__main__":
    main()
