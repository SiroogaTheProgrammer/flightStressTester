import json

import numpy as np

import yolo_fastest as yf
import yolo_graph as yg

ARENA_ALIGN = 16
SIGNED_RANGE = (-128, 127)
UNSIGNED_RANGE = (0, 255)


def quantize_multiplier(real):
    real = np.atleast_1d(np.asarray(real, dtype=np.float64))
    exponent = int(np.frexp(real.max())[1])
    shift = 31 - exponent
    if not 1 <= shift <= 62:
        raise ValueError("multiplier out of range")
    mult = np.minimum(np.round(real * 2.0 ** shift), (1 << 31) - 1).astype(np.int32)
    return mult, shift


def requantize(acc, mult, shift, lo, hi):
    acc = acc.astype(np.int64) * mult.astype(np.int64)
    acc = (acc + (np.int64(1) << (shift - 1))) >> shift
    return np.clip(acc, lo, hi)


def calibrate(g, images_rgb, percentile):
    rng = np.random.default_rng(0)
    pooled = {b.id: [] for b in g.bufs}
    maxima = {b.id: 0.0 for b in g.bufs}
    for image in images_rgb:
        arrays = yg.run_float(g, image)
        for b in g.bufs:
            values = np.abs(arrays[b.id]).ravel()
            maxima[b.id] = max(maxima[b.id], float(values.max()))
            pooled[b.id].append(values[rng.integers(0, values.size, 20000)])
    stats = {}
    for b in g.bufs:
        use_max = b.keep or percentile >= 100.0
        values = np.concatenate(pooled[b.id])
        stats[b.id] = maxima[b.id] if use_max else float(np.percentile(values, percentile))
    return stats


def assign_scales(g, stats):
    for b in g.bufs:
        if b.external:
            continue
        limit = max(stats[b.id], 1e-6)
        b.scale = limit / (127.0 if b.signed else 255.0)
    for op in g.ops:
        if op.kind == "split":
            g.bufs[op.dst2.buf].scale = g.bufs[op.src.buf].scale


def quantize_op(g, op):
    src, dst = g.bufs[op.src.buf], g.bufs[op.dst.buf]
    lo, hi = SIGNED_RANGE if dst.signed else UNSIGNED_RANGE
    op.lo, op.hi = lo, hi
    op.in_signed = src.signed
    if op.kind in ("split", "upcopy"):
        real = src.scale / dst.scale
        mult, shift = quantize_multiplier(real)
        op.mult, op.sh = mult, shift
        op.identity = bool(mult[0] == (1 << 30) and shift == 30)
        return
    assert dst.signed == (not op.relu), op.name
    if op.kind == "dw":
        per_channel = op.wf.reshape(-1, op.wf.shape[-1])
        max_abs = np.abs(per_channel).max(axis=0)
        channels = per_channel.shape[1]
    else:
        per_channel = op.wf.reshape(op.wf.shape[0], -1)
        max_abs = np.abs(per_channel).max(axis=1)
        channels = per_channel.shape[0]
    floor = np.abs(op.bf) / (src.scale * (1 << 22))
    tiny = 2.0 ** -30 * dst.scale / src.scale
    sw = np.maximum(np.maximum(max_abs / 127.0, floor), tiny)
    if op.kind == "dw":
        wq = np.round(per_channel / sw[None, :])
    else:
        wq = np.round(per_channel / sw[:, None])
    op.wq = np.clip(wq, -127, 127).astype(np.int8)
    op.bq = np.clip(np.round(op.bf / (src.scale * sw)), -(1 << 30), 1 << 30).astype(np.int32)
    op.mult, op.sh = quantize_multiplier(src.scale * sw / dst.scale)
    if op.sh < 33:
        raise ValueError(f"{op.name}: shift {op.sh} below the firmware fast path")
    assert op.bq.shape[0] == channels


def quantize_graph(g):
    for op in g.ops:
        quantize_op(g, op)


def plan_memory(g):
    first, last = {}, {}
    for index, op in enumerate(g.ops):
        for view in (op.src, op.dst, op.dst2):
            if view is None:
                continue
            first.setdefault(view.buf, index)
            first[view.buf] = min(first[view.buf], index)
            last[view.buf] = max(last.get(view.buf, index), index)
    end = len(g.ops)
    for b in g.bufs:
        if b.keep:
            last[b.id] = end
    placed = []
    order = sorted(g.bufs, key=lambda b: (-b.size, b.id))
    for b in order:
        busy = []
        for other in placed:
            if first[other.id] <= last[b.id] and first[b.id] <= last[other.id]:
                busy.append((other.offset, other.offset + other.size))
        busy.sort()
        offset = 0
        for start, stop in busy:
            if offset + b.size <= start:
                break
            offset = max(offset, (stop + ARENA_ALIGN - 1) // ARENA_ALIGN * ARENA_ALIGN)
        b.offset = offset
        placed.append(b)
    g.arena_size = max(b.offset + b.size for b in placed)
    live = 0
    for index in range(end + 1):
        live = max(live, sum(b.size for b in placed if first[b.id] <= index <= last[b.id]))
    g.arena_lower_bound = live
    g.lifetimes = {b.id: (first[b.id], last[b.id]) for b in placed}


def view_array(g, arena, view):
    b = g.bufs[view.buf]
    dtype = np.int8 if b.signed else np.uint8
    itemsize = 1
    return np.ndarray(
        (b.h, b.w, view.c), dtype, buffer=arena, offset=b.offset + view.coff,
        strides=(b.w * b.c * itemsize, b.c * itemsize, itemsize),
    )


def checksum(values):
    data = np.ascontiguousarray(values).astype(np.uint8).ravel().astype(np.uint64)
    n = data.size
    s1 = int(data.sum()) & 0xFFFFFFFF
    s2 = int(((np.arange(n, 0, -1, dtype=np.uint64)) * data).sum()) & 0xFFFFFFFF
    return (s2 + 0x9E3779B1 * s1) & 0xFFFFFFFF


def pad_zero(x, pad):
    return np.pad(x, ((pad, pad), (pad, pad), (0, 0)))


def run_int(g, image_rgb_u8, trace=None):
    arena = np.zeros(g.arena_size + 64, np.uint8)
    view_array(g, arena, g.full(g.bufs[0]))[...] = np.ascontiguousarray(image_rgb_u8, dtype=np.uint8)
    for index, op in enumerate(g.ops):
        x = view_array(g, arena, op.src).astype(np.int64)
        dst = view_array(g, arena, op.dst)
        h, w, c = x.shape
        if op.kind == "pw":
            acc = (x.reshape(-1, c).astype(np.float64) @ op.wq.T.astype(np.float64)).astype(np.int64) + op.bq
            out = requantize(acc, op.mult, op.sh, op.lo, op.hi).reshape(h, w, -1)
        elif op.kind == "dw":
            pad = op.k // 2
            xp = pad_zero(x, pad)
            oh, ow = (h - 1) // op.stride + 1, (w - 1) // op.stride + 1
            wq = op.wq.reshape(op.k, op.k, c).astype(np.int64)
            acc = np.zeros((oh, ow, c), np.int64)
            for ky in range(op.k):
                for kx in range(op.k):
                    tap = xp[ky:ky + op.stride * oh:op.stride, kx:kx + op.stride * ow:op.stride, :]
                    acc += tap * wq[ky, kx]
            acc += op.bq
            out = requantize(acc, op.mult, op.sh, op.lo, op.hi)
        elif op.kind == "conv1pool":
            xp = pad_zero(x, 1)
            oh, ow = h // 2, w // 2
            wq = op.wq[:, :27].reshape(24, 3, 3, 3).astype(np.int64)
            acc = np.zeros((oh, ow, 24), np.int64)
            for ky in range(3):
                for kx in range(3):
                    tap = xp[ky:ky + 2 * oh:2, kx:kx + 2 * ow:2, :]
                    acc += tap @ wq[:, ky, kx, :].T
            acc += op.bq
            conv = requantize(acc, op.mult, op.sh, op.lo, op.hi)
            cp = np.pad(conv, ((1, 1), (1, 1), (0, 0)), constant_values=0)
            ph, pw_ = oh // 2, ow // 2
            out = np.zeros((ph, pw_, 24), np.int64)
            for ky in range(3):
                for kx in range(3):
                    out = np.maximum(out, cp[ky:ky + 2 * ph:2, kx:kx + 2 * pw_:2, :])
        elif op.kind == "split":
            even = x[:, :, 0::2]
            out = requantize(even, op.mult, op.sh, op.lo, op.hi)
            dst2 = view_array(g, arena, op.dst2)
            dst2[...] = x[:, :, 1::2].astype(dst2.dtype)
        elif op.kind == "upcopy":
            up = x.repeat(2, axis=0).repeat(2, axis=1)
            out = requantize(up, op.mult, op.sh, op.lo, op.hi)
        else:
            raise ValueError(op.kind)
        dst[...] = out.astype(dst.dtype)
        if trace is not None:
            value = checksum(dst)
            if op.dst2 is not None:
                value = (value * 1000003 + checksum(view_array(g, arena, op.dst2))) & 0xFFFFFFFF
            trace.append(value)
    return arena


def head_arrays(g, arena):
    heads = []
    for names in g.heads:
        parts = []
        for label in ("reg", "obj", "cls"):
            b = g.bufs[names[label]]
            raw = view_array(g, arena, yg.View(b.id, 0, b.c)).astype(np.float32)
            parts.append(raw * np.float32(b.scale))
        heads.append(tuple(parts))
    return heads


def output_checksum(g, arena):
    value = 0
    for names in g.heads:
        for label in ("reg", "obj", "cls"):
            b = g.bufs[names[label]]
            value = (value * 1000003 + checksum(view_array(g, arena, yg.View(b.id, 0, b.c)))) & 0xFFFFFFFF
    return value


def detections_from_arena(g, arena, conf=0.3, iou=0.4):
    return yf.nms(yf.decode_heads(head_arrays(g, arena), g.size), conf, iou)


def _view_json(view):
    return None if view is None else [view.buf, view.coff, view.c]


def _view_load(value):
    return None if value is None else yg.View(*value)


def save_ref(g, path):
    meta = {
        "size": g.size,
        "arena_size": g.arena_size,
        "heads": g.heads,
        "bufs": [
            {"id": b.id, "name": b.name, "h": b.h, "w": b.w, "c": b.c, "signed": b.signed,
             "scale": b.scale, "external": b.external, "keep": b.keep, "offset": b.offset}
            for b in g.bufs
        ],
        "ops": [
            {"kind": op.kind, "name": op.name, "src": _view_json(op.src), "dst": _view_json(op.dst),
             "dst2": _view_json(op.dst2), "k": op.k, "stride": op.stride, "relu": op.relu, "sh": op.sh,
             "identity": op.identity, "lo": op.lo, "hi": op.hi, "in_signed": op.in_signed}
            for op in g.ops
        ],
    }
    arrays = {}
    for index, op in enumerate(g.ops):
        arrays[f"mult{index}"] = np.asarray(op.mult)
        if op.wq is not None:
            arrays[f"wq{index}"] = op.wq
            arrays[f"bq{index}"] = op.bq
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load_ref(path):
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    g = yg.Graph(meta["size"])
    g.arena_size = meta["arena_size"]
    g.heads = meta["heads"]
    for b in meta["bufs"]:
        g.bufs.append(yg.Buf(**b))
    for index, o in enumerate(meta["ops"]):
        op = yg.Op(o["kind"], o["name"], _view_load(o["src"]), _view_load(o["dst"]), _view_load(o["dst2"]),
                   k=o["k"], stride=o["stride"], relu=o["relu"])
        op.sh, op.identity, op.lo, op.hi, op.in_signed = o["sh"], o["identity"], o["lo"], o["hi"], o["in_signed"]
        op.mult = data[f"mult{index}"]
        if f"wq{index}" in data:
            op.wq, op.bq = data[f"wq{index}"], data[f"bq{index}"]
        g.ops.append(op)
    return g
