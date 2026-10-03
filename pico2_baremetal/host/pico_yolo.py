import argparse
import queue
import statistics
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

SAMPLES = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PICO_VID = 0x2E8A


def find_port(requested):
    from serial.tools import list_ports

    if requested:
        return requested
    ports = [p for p in list_ports.comports() if p.vid == PICO_VID]
    if len(ports) == 1:
        return ports[0].device
    listing = ", ".join(f"{p.device} ({p.description})" for p in list_ports.comports()) or "none"
    raise SystemExit(f"Specify --port; found {len(ports)} Pico ports. Available: {listing}")


def download_samples():
    folder = Path(tempfile.gettempdir()) / "pico_yolo_samples"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in SAMPLES.items():
        path = folder / name
        if not path.exists():
            print(f"Downloading {name}")
            urllib.request.urlretrieve(url, path)
        paths.append(path)
    return paths


def collect_images(args):
    if args.samples:
        return download_samples()
    paths = list(args.images or [])
    for folder in args.dir or []:
        paths += sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit("Choose images with --images, --dir or --samples")
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise SystemExit("Image not found: " + ", ".join(map(str, missing)))
    return [Path(p) for p in paths]


class Link:
    def __init__(self, port_name):
        import serial

        self.port = serial.Serial(port_name, 115200, timeout=0.1, write_timeout=30)
        self.lines = queue.Queue()
        self.stop = threading.Event()
        self.port.reset_input_buffer()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        buffer = b""
        while not self.stop.is_set():
            try:
                chunk = self.port.read(4096)
            except Exception as error:
                self.lines.put(f"LINKERR,{error}")
                return
            if not chunk:
                continue
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                text = line.decode("ascii", errors="replace").strip()
                if text:
                    self.lines.put(text)

    def send(self, data):
        self.port.write(data)

    def wait(self, prefix, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if line.startswith(prefix):
                return line
        raise TimeoutError(f"no {prefix} reply from the Pico")

    def close(self):
        self.stop.set()
        self.reader.join(timeout=1)
        self.port.close()


def preprocess(path, size):
    import cv2

    original = cv2.imread(str(path))
    if original is None:
        raise SystemExit(f"Cannot read image {path}")
    resized = cv2.resize(original, (size, size), interpolation=cv2.INTER_LINEAR)
    return original, np.ascontiguousarray(resized[:, :, ::-1])


def annotate(original, detections, size, names, out_path):
    import cv2

    scale_x, scale_y = original.shape[1] / size, original.shape[0] / size
    canvas = original.copy()
    for cls, score, x1, y1, x2, y2 in detections:
        p1 = (int(x1 * scale_x), int(y1 * scale_y))
        p2 = (int(x2 * scale_x), int(y2 * scale_y))
        cv2.rectangle(canvas, p1, p2, (255, 200, 0), 2)
        cv2.putText(canvas, f"{names[cls]} {score / 1000:.2f}", (p1[0], max(p1[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))]


def load_reference():
    import yolo_quant as yq

    path = ROOT / "yolo" / "yolo_model_ref.npz"
    if not path.exists():
        return None, None
    return yq, yq.load_ref(path)


def check_layers(link, rgb, yq, graph, size):
    link.send(b"DEBUG,1\n")
    link.wait("OK,DEBUG", 5)
    link.send(b"IMG,0\n" + rgb.tobytes())
    link.wait("RESULT,0,", 60)
    link.send(b"SUMS\n")
    device = [int(v, 16) for v in link.wait("SUMS,", 10).split(",")[2:]]
    trace = []
    yq.run_int(graph, rgb, trace)
    bad = [i for i, (a, b) in enumerate(zip(device, trace)) if a != b]
    link.send(b"DEBUG,0\n")
    link.wait("OK,DEBUG", 5)
    if bad:
        op = graph.ops[bad[0]]
        print(f"LAYER MISMATCH: {len(bad)} of {len(trace)} layers differ, first is #{bad[0]} {op.name} ({op.kind})")
    else:
        print(f"All {len(trace)} layer checksums match the Python int8 reference.")
    return not bad


def print_profile(link, graph):
    link.send(b"PROFILE\n")
    values = [int(v) for v in link.wait("PROFILE,", 10).split(",")[2:]]
    total = sum(values)
    print(f"\nPer-layer time on the Pico (last image, total {total / 1000:.1f} ms):")
    if graph is None:
        print(" ", values)
        return
    by_kind = {}
    for op, us in zip(graph.ops, values):
        by_kind[op.kind] = by_kind.get(op.kind, 0) + us
    for kind, us in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:<10} {us / 1000:8.1f} ms  {100 * us / total:5.1f}%")
    print("  slowest layers:")
    for index in sorted(range(len(values)), key=lambda i: -values[i])[:6]:
        print(f"    #{index:<3} {graph.ops[index].name:<22} {values[index] / 1000:7.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Run chosen images through the YOLO detector running on the Pico 2 W.")
    parser.add_argument("--port", help="Pico serial port, auto-detected when omitted")
    parser.add_argument("--images", nargs="+", type=Path, help="Image files to run")
    parser.add_argument("--dir", nargs="+", type=Path, help="Folders whose images are run")
    parser.add_argument("--samples", action="store_true", help="Download and run the two sample images")
    parser.add_argument("--repeat", type=int, default=1, help="Run the whole list this many times")
    parser.add_argument("--seconds", type=float, default=0, help="Keep cycling the list for this long (stress test)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.4, help="NMS IoU threshold")
    parser.add_argument("--window", type=int, default=2, help="Images in flight; 1 disables USB/compute overlap")
    parser.add_argument("--verify", type=int, default=1, help="Check this many distinct images against the Python int8 reference")
    parser.add_argument("--check-layers", action="store_true", help="Compare every layer's checksum with the reference")
    parser.add_argument("--profile", action="store_true", help="Print the per-layer time breakdown at the end")
    parser.add_argument("--save-dir", type=Path, default=Path("runs/pico_yolo"), help="Where annotated images go")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1 or args.window < 1:
        parser.error("--repeat and --window must be at least 1")

    paths = collect_images(args)
    names = (ROOT / "yolo" / "coco.names").read_text().split("\n")
    yq, graph = load_reference()

    link = Link(find_port(args.port))
    try:
        link.send(b"HELLO\n")
        ready = link.wait("READY,", 15).split(",")
        size, arena, layers, mhz, slots = int(ready[2]), int(ready[3]), int(ready[4]), int(ready[5]), int(ready[6])
        print(f"Pico ready: YOLO {size}x{size}, {layers} layers, arena {arena / 1024:.0f} KB, {mhz} MHz")
        if graph is not None and graph.size != size:
            print("Reference file does not match the firmware size; verification disabled.")
            yq = graph = None
        link.send(f"CFG,{round(args.conf * 1000)},{round(args.iou * 1000)}\n".encode())
        link.wait("OK,CFG", 5)

        prepared = [preprocess(p, size) for p in paths]
        if args.check_layers:
            if graph is None:
                raise SystemExit("--check-layers needs yolo/yolo_model_ref.npz")
            if not check_layers(link, prepared[0][1], yq, graph, size):
                sys.exit(1)

        plan = [i % len(paths) for i in range(len(paths) * args.repeat)]
        deadline = time.monotonic() + args.seconds if args.seconds else None
        if deadline:
            plan = None

        expected = {}
        for index in range(min(args.verify, len(paths)) if graph is not None else 0):
            expected[index] = yq.output_checksum(graph, yq.run_int(graph, prepared[index][1]))
        seen_checksum = {}

        sent_at, dets, rows = {}, {}, []
        saved = set()
        next_id, completed, mismatches, started = 0, 0, 0, time.perf_counter()
        total = len(plan) if plan else None
        print(f"{'id':>4} {'image':<18} {'det':>3} {'pico ms':>8} {'decode':>7} {'usb rx':>7} {'e2e ms':>8}  result")
        while True:
            while len(sent_at) < args.window and (total is None and time.monotonic() < deadline or total is not None and next_id < total):
                image_index = plan[next_id] if plan else next_id % len(paths)
                sent_at[next_id] = (time.perf_counter(), image_index)
                link.send(f"IMG,{next_id}\n".encode() + prepared[image_index][1].tobytes())
                next_id += 1
            if not sent_at:
                break
            try:
                line = link.lines.get(timeout=60)
            except queue.Empty:
                raise SystemExit("Timed out waiting for the Pico")
            fields = line.split(",")
            if fields[0] == "DET":
                dets.setdefault(int(fields[1]), []).append(tuple(int(v) for v in fields[2:8]))
            elif fields[0] == "RESULT":
                rid = int(fields[1])
                t_sent, image_index = sent_at.pop(rid)
                e2e = (time.perf_counter() - t_sent) * 1000
                infer, decode, rx = int(fields[3]), int(fields[4]), int(fields[5])
                checksum = int(fields[10], 16)
                status = "ok"
                if image_index in expected:
                    status = "bit-exact" if checksum == expected[image_index] else "MISMATCH vs python"
                if image_index in seen_checksum and seen_checksum[image_index] != checksum:
                    status = "UNSTABLE (differs from earlier run)"
                seen_checksum.setdefault(image_index, checksum)
                mismatches += status.startswith(("MISMATCH", "UNSTABLE"))
                found = dets.pop(rid, [])
                rows.append((infer, decode, rx, e2e))
                completed += 1
                label = ", ".join(f"{names[c]} {s / 1000:.2f}" for c, s, *_ in found[:4]) + (" ..." if len(found) > 4 else "")
                print(f"{rid:>4} {paths[image_index].name[:18]:<18} {len(found):>3} {infer / 1000:>8.1f} {decode / 1000:>7.1f} "
                      f"{rx / 1000:>7.1f} {e2e:>8.0f}  {status}  {label}")
                if not args.no_save and image_index not in saved:
                    saved.add(image_index)
                    annotate(prepared[image_index][0], found, size, names, args.save_dir / f"{paths[image_index].stem}_pico.jpg")
            elif fields[0] in ("ERR", "LINKERR"):
                print("  ", line)
        elapsed = time.perf_counter() - started

        if rows:
            infer = [r[0] / 1000 for r in rows]
            print(f"\n{completed} images in {elapsed:.1f} s -> {completed / elapsed:.2f} images/s end-to-end")
            print(f"Pico inference: mean {statistics.mean(infer):.1f} ms, median {statistics.median(infer):.1f}, "
                  f"min {min(infer):.1f}, p95 {percentile(infer, 95):.1f}, max {max(infer):.1f}  "
                  f"(~{1000 / statistics.mean(infer):.2f} images/s compute-bound at {mhz} MHz)")
            print(f"Box decode + NMS: mean {statistics.mean(r[1] for r in rows) / 1000:.2f} ms; "
                  f"USB image receive: mean {statistics.mean(r[2] for r in rows) / 1000:.1f} ms; "
                  f"host end-to-end: mean {statistics.mean(r[3] for r in rows):.0f} ms")
            print("Integrity: " + ("all runs bit-identical" if not mismatches else f"{mismatches} PROBLEM(S) detected"))
            if not args.no_save:
                print(f"Annotated images: {args.save_dir.resolve()}")
        if args.profile:
            print_profile(link, graph)
    finally:
        link.close()


if __name__ == "__main__":
    main()
