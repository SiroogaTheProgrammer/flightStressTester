import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import yolo_fastest as yf
import yolo_quant as yq

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["host_test.cpp", "yolo_engine.cpp", "yolo_decode.cpp", "yolo_model_data.cpp"]


def build(exe):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise SystemExit("no host C++ compiler found on PATH")
    command = [compiler, "-O2", "-std=c++17", "-DYOLO_HOST_TEST", "-I", str(ROOT / "yolo")]
    command += [str(ROOT / "yolo" / name) for name in SOURCES] + ["-o", str(exe)]
    subprocess.run(command, check=True)


def load_rgb(path, size):
    import cv2

    bgr = cv2.resize(cv2.imread(str(path)), (size, size), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(bgr[:, :, ::-1])


def main():
    parser = argparse.ArgumentParser(description="Compare the firmware engine (host build) with the Python int8 reference.")
    parser.add_argument("images", nargs="*", type=Path)
    args = parser.parse_args()
    images = args.images or [Path(tempfile.gettempdir()) / "pico_yolo_samples" / n for n in ("bus.jpg", "zidane.jpg")]

    g = yq.load_ref(ROOT / "yolo" / "yolo_model_ref.npz")
    exe = Path(tempfile.gettempdir()) / "pico_yolo_cache" / "yolo_host_test.exe"
    build(exe)
    names = yf.class_names()
    failures = 0
    for path in images:
        rgb = load_rgb(path, g.size)
        raw = Path(tempfile.gettempdir()) / "pico_yolo_cache" / "input.rgb"
        raw.write_bytes(rgb.tobytes())
        trace = []
        arena = yq.run_int(g, rgb, trace)
        expected_sums = trace
        expected_out = yq.output_checksum(g, arena)
        expected = yq.detections_from_arena(g, arena)

        output = subprocess.run([str(exe), str(raw)], check=True, capture_output=True, text=True).stdout.splitlines()
        sums = [int(v) for v in output[0].split(",")[1:]]
        out = int(output[1].split(",")[1])
        dets = [line.split(",")[1:] for line in output if line.startswith("DET,")]

        bad = [i for i, (a, b) in enumerate(zip(sums, expected_sums)) if a != b]
        print(f"{path.name}: layers mismatching={len(bad)} first={bad[:3]} output_checksum_match={out == expected_out}")
        print("  engine :", ", ".join(f"{names[int(c)]}:{int(s) / 1000:.2f}" for c, s, *_ in dets))
        print("  python :", ", ".join(f"{names[int(d[5])]}:{d[4]:.2f}" for d in expected))
        failures += bool(bad) or out != expected_out
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
