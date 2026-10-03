import argparse
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

import yolo_fastest as yf
import yolo_graph as yg
import yolo_quant as yq

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
KIND_CODES = {"conv1pool": 0, "pw": 1, "dw": 2, "split": 3, "upcopy": 4}
CONV1_TAPS = 28


def calibration_files():
    cache = yf.CACHE_DIR
    archive = cache / "coco128.zip"
    if not archive.exists():
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(COCO128_URL, archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(cache)
    return sorted((cache / "coco128" / "images" / "train2017").glob("*.jpg"))


def load_rgb(path, size):
    import cv2

    bgr = cv2.resize(cv2.imread(str(path)), (size, size), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(bgr[:, :, ::-1])


def box_matches(a, b, iou_threshold=0.5):
    used, hits = set(), 0
    for box in a:
        for j, other in enumerate(b):
            if j in used or other[5] != box[5]:
                continue
            iw = min(box[2], other[2]) - max(box[0], other[0])
            ih = min(box[3], other[3]) - max(box[1], other[1])
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            union = (box[2] - box[0]) * (box[3] - box[1]) + (other[2] - other[0]) * (other[3] - other[1]) - inter
            if inter / union > iou_threshold:
                hits += 1
                used.add(j)
                break
    return hits


def validate(g, images):
    tp = fp = fn = 0
    for image in images:
        arrays = yg.run_float(g, image)
        heads = [tuple(arrays[h[label]] for label in ("reg", "obj", "cls")) for h in g.heads]
        reference = yf.nms(yf.decode_heads(heads, g.size), 0.3, 0.4)
        got = yq.detections_from_arena(g, yq.run_int(g, image))
        hits = box_matches(got, reference)
        tp += hits
        fp += len(got) - hits
        fn += len(reference) - hits
    return tp, fp, fn


def expand_rows(rows):
    count, cin = rows.shape
    quad = rows.reshape(count, cin // 4, 4).astype(np.int16)
    out = np.empty((count, cin // 4, 2, 2), np.int16)
    out[:, :, 0, 0], out[:, :, 0, 1] = quad[:, :, 0], quad[:, :, 2]
    out[:, :, 1, 0], out[:, :, 1, 1] = quad[:, :, 1], quad[:, :, 3]
    return out


def pack_pointwise(wq):
    cout, cin = wq.shape
    expanded = expand_rows(wq)
    blocks = cout // 4
    parts = []
    if blocks:
        grouped = expanded[:4 * blocks].reshape(blocks, 4, cin // 4, 2, 2).transpose(0, 2, 1, 3, 4)
        parts.append(grouped.ravel())
    if cout % 4:
        parts.append(expanded[4 * blocks:].ravel())
    return np.concatenate(parts).astype("<i2")


def pack_depthwise(wq):
    taps, channels = wq.shape
    quad = wq.reshape(taps, channels // 4, 4).astype(np.int16)
    out = np.empty((taps, channels // 4, 2, 2), np.int16)
    out[:, :, 0, 0], out[:, :, 0, 1] = quad[..., 0], quad[..., 2]
    out[:, :, 1, 0], out[:, :, 1, 1] = quad[..., 1], quad[..., 3]
    return out.astype("<i2").ravel()


def weight_bytes(op):
    if op.kind == "pw":
        return pack_pointwise(op.wq).tobytes()
    if op.kind == "dw":
        return pack_depthwise(op.wq).tobytes()
    if op.kind == "conv1pool":
        padded = np.zeros((op.wq.shape[0], CONV1_TAPS), np.int8)
        padded[:, :op.wq.shape[1]] = op.wq
        return pack_pointwise(padded).tobytes()
    return b""


def format_array(ctype, name, values, per_line):
    out = [f"alignas(16) const {ctype} {name}[] = {{"]
    values = list(values)
    for start in range(0, len(values), per_line):
        out.append("    " + ", ".join(str(int(v)) for v in values[start:start + per_line]) + ",")
    out.append("};")
    return "\n".join(out) + "\n"


def check_alignment(g, op):
    if op.kind not in ("pw", "dw"):
        return
    src = g.bufs[op.src.buf]
    assert op.src.c % 4 == 0 and src.c % 4 == 0 and (src.offset + op.src.coff) % 4 == 0, op.name


def layer_row(g, op, w_off, w_len, b_off):
    src, dst = g.bufs[op.src.buf], g.bufs[op.dst.buf]
    dst2 = g.bufs[op.dst2.buf] if op.dst2 is not None else None
    cout = op.dst.c
    cin = op.src.c
    mult = int(op.mult[0]) if op.kind in ("split", "upcopy") else 0
    out2_off = dst2.offset + op.dst2.coff if dst2 is not None else 0
    out2_stride = dst2.c if dst2 is not None else 0
    fields = [
        KIND_CODES[op.kind], op.k, op.stride, int(op.in_signed), int(dst.signed), int(op.identity), op.sh, 0,
        src.h, src.w, dst.h, dst.w, cin, cout, src.c, dst.c, out2_stride, 0,
        src.offset + op.src.coff, dst.offset + op.dst.coff, out2_off, w_off, w_len, b_off, mult,
    ]
    return "    {" + ", ".join(str(int(v)) for v in fields) + "},"


def generate(g, out_dir):
    weights = bytearray()
    biases, mults = [], []
    rows = []
    max_weights = 0
    for op in g.ops:
        check_alignment(g, op)
        blob = weight_bytes(op)
        w_off = len(weights)
        weights += blob
        weights += b"\0" * (-len(weights) % 16)
        b_off = len(biases)
        if op.wq is not None:
            biases.extend(op.bq.tolist())
            mults.extend(np.asarray(op.mult).tolist())
        max_weights = max(max_weights, len(blob))
        rows.append(layer_row(g, op, w_off, len(blob), b_off))

    img = g.bufs[0]
    header = f"""#pragma once

#include "yolo_layer.h"

namespace yolo {{

constexpr int kInputSize = {g.size};
constexpr int kInputBytes = {img.size};
constexpr uint32_t kInputOffset = {img.offset};
constexpr uint32_t kArenaSize = {g.arena_size};
constexpr int kLayerCount = {len(g.ops)};
constexpr uint32_t kMaxWeightBytes = {max_weights};
constexpr int kNumClasses = {yf.NUM_CLASSES};
constexpr int kNumAnchors = {yf.NUM_ANCHORS};

extern const Layer kLayers[kLayerCount];
extern const Head kHeads[2];
extern const int8_t kWeights[];
extern const int32_t kBias[];
extern const int32_t kMult[];

}}  // namespace yolo
"""
    head_rows = []
    for index, names in enumerate(g.heads):
        reg, obj, cls = (g.bufs[names[label]] for label in ("reg", "obj", "cls"))
        anchors = ", ".join(f"{v:.2f}f" for v in yf.ANCHORS[index].ravel())
        head_rows.append(
            f"    {{{reg.offset}, {obj.offset}, {cls.offset}, {reg.scale!r}f, {obj.scale!r}f, {cls.scale!r}f, "
            f"{reg.h}, {reg.w}, {{{anchors}}}}},"
        )
    source = '#include "yolo_model_data.h"\n\nnamespace yolo {\n\n'
    source += "const Layer kLayers[kLayerCount] = {\n" + "\n".join(rows) + "\n};\n\n"
    source += "const Head kHeads[2] = {\n" + "\n".join(head_rows) + "\n};\n\n"
    source += format_array("int8_t", "kWeights", np.frombuffer(bytes(weights), np.int8), 32) + "\n"
    source += format_array("int32_t", "kBias", biases, 12) + "\n"
    source += format_array("int32_t", "kMult", mults, 8) + "\n}  // namespace yolo\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "yolo_model_data.h").write_text(header, encoding="ascii", newline="\n")
    (out_dir / "yolo_model_data.cpp").write_text(source, encoding="ascii", newline="\n")
    return len(weights)


def mac_summary(g):
    totals = {}
    for op in g.ops:
        src, dst = g.bufs[op.src.buf], g.bufs[op.dst.buf]
        if op.kind == "pw":
            macs = src.h * src.w * op.src.c * op.dst.c
        elif op.kind == "dw":
            macs = dst.h * dst.w * op.dst.c * op.k * op.k
        elif op.kind == "conv1pool":
            macs = (src.h // 2) * (src.w // 2) * 24 * 27
        else:
            macs = 0
        totals[op.kind] = totals.get(op.kind, 0) + macs
    return totals


def main():
    parser = argparse.ArgumentParser(description="Quantize Yolo-FastestV2 to int8 and generate Pico 2 W firmware data.")
    parser.add_argument("--size", type=int, default=224, help="Square input size, multiple of 32")
    parser.add_argument("--percentile", type=float, default=99.999, help="Activation clipping percentile")
    parser.add_argument("--calib", type=int, default=96, help="Calibration images taken from coco128")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "yolo")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    files = calibration_files()
    images = [load_rgb(path, args.size) for path in files]
    calib, held_out = images[:args.calib], images[args.calib:]

    model = yf.load_float_model()
    g = yg.build_graph(model.state_dict(), args.size)
    stats = yq.calibrate(g, calib, args.percentile)
    yq.assign_scales(g, stats)
    yq.quantize_graph(g)
    yq.plan_memory(g)

    if not args.skip_validate and held_out:
        tp, fp, fn = validate(g, held_out)
        print(f"int8 vs float on {len(held_out)} held-out images: tp={tp} fp={fp} fn={fn} "
              f"precision={tp / max(tp + fp, 1):.3f} recall={tp / max(tp + fn, 1):.3f}")

    weight_total = generate(g, args.out_dir)
    yq.save_ref(g, args.out_dir / "yolo_model_ref.npz")
    macs = mac_summary(g)
    print(f"size={g.size} layers={len(g.ops)} arena={g.arena_size} bytes weights={weight_total} bytes")
    print("MACs: " + ", ".join(f"{k}={v / 1e6:.2f}M" for k, v in macs.items()) + f", total={sum(macs.values()) / 1e6:.2f}M")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
