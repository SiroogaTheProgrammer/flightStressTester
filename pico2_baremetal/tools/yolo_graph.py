from dataclasses import dataclass

import numpy as np

BN_EPS = 1e-5
STAGE_REPEATS = (4, 8, 4)
STAGE_CHANNELS = (24, 48, 96, 192)
HEAD_CHANNELS = 72


@dataclass
class Buf:
    id: int
    name: str
    h: int
    w: int
    c: int
    signed: bool = False
    scale: float = 1.0
    external: bool = False
    keep: bool = False
    offset: int = -1

    @property
    def size(self):
        return self.h * self.w * self.c


@dataclass
class View:
    buf: int
    coff: int
    c: int


@dataclass
class Op:
    kind: str
    name: str
    src: View
    dst: View
    dst2: View | None = None
    k: int = 1
    stride: int = 1
    relu: bool = False
    wf: np.ndarray | None = None
    bf: np.ndarray | None = None
    wq: np.ndarray | None = None
    bq: np.ndarray | None = None
    mult: np.ndarray | None = None
    sh: np.ndarray | None = None
    identity: bool = False
    lo: int = 0
    hi: int = 255
    in_signed: bool = False


class Graph:
    def __init__(self, size):
        self.size = size
        self.bufs = []
        self.ops = []
        self.heads = []

    def add_buf(self, name, h, w, c, signed=False, external=False, keep=False):
        buf = Buf(len(self.bufs), name, h, w, c, signed=signed, external=external, keep=keep)
        self.bufs.append(buf)
        return buf

    def full(self, buf):
        return View(buf.id, 0, buf.c)

    def add_op(self, op):
        self.ops.append(op)
        return op

    def pw(self, name, src, dst, wf, bf, relu):
        wf = wf.reshape(wf.shape[0], -1)
        return self.add_op(Op("pw", name, src, dst, wf=wf, bf=bf, relu=relu))

    def dw(self, name, src, dst, wf, bf, k, stride, relu):
        wf = np.ascontiguousarray(wf.reshape(wf.shape[0], k, k).transpose(1, 2, 0))
        return self.add_op(Op("dw", name, src, dst, k=k, stride=stride, wf=wf, bf=bf, relu=relu))


def fold_conv(sd, conv_key, bn_key=None):
    weight = sd[conv_key + ".weight"].numpy().astype(np.float64)
    if bn_key is None:
        bias = sd[conv_key + ".bias"].numpy().astype(np.float64)
        return weight.astype(np.float32), bias.astype(np.float32)
    gamma = sd[bn_key + ".weight"].numpy().astype(np.float64)
    beta = sd[bn_key + ".bias"].numpy().astype(np.float64)
    mean = sd[bn_key + ".running_mean"].numpy().astype(np.float64)
    var = sd[bn_key + ".running_var"].numpy().astype(np.float64)
    scale = gamma / np.sqrt(var + BN_EPS)
    weight = weight * scale[:, None, None, None]
    return weight.astype(np.float32), (beta - mean * scale).astype(np.float32)


def shuffle_unit(g, sd, key, cur, inp, oup, stride, out=None):
    src = g.bufs[cur.buf]
    height, width = src.h, src.w
    mid = oup // 2
    count = oup - inp
    if stride == 2:
        oh, ow = height // 2, width // 2
        y = out or g.full(g.add_buf(key + ".y", oh, ow, oup))
        m1 = g.add_buf(key + ".m1", height, width, mid)
        wf, bf = fold_conv(sd, key + ".branch_main.0", key + ".branch_main.1")
        g.pw(key + ".m.pw1", cur, g.full(m1), wf, bf, True)
        m2 = g.add_buf(key + ".m2", oh, ow, mid, signed=True)
        wf, bf = fold_conv(sd, key + ".branch_main.3", key + ".branch_main.4")
        g.dw(key + ".m.dw", g.full(m1), g.full(m2), wf, bf, 3, 2, False)
        wf, bf = fold_conv(sd, key + ".branch_main.5", key + ".branch_main.6")
        g.pw(key + ".m.pw2", g.full(m2), View(y.buf, y.coff + inp, count), wf, bf, True)
        p1 = g.add_buf(key + ".p1", oh, ow, inp, signed=True)
        wf, bf = fold_conv(sd, key + ".branch_proj.0", key + ".branch_proj.1")
        g.dw(key + ".p.dw", cur, g.full(p1), wf, bf, 3, 2, False)
        wf, bf = fold_conv(sd, key + ".branch_proj.2", key + ".branch_proj.3")
        g.pw(key + ".p.pw", g.full(p1), View(y.buf, y.coff, inp), wf, bf, True)
        return y

    half = oup // 2
    y = out or g.full(g.add_buf(key + ".y", height, width, oup))
    xo = g.add_buf(key + ".xo", height, width, half, signed=src.signed)
    g.add_op(Op("split", key + ".split", cur, View(y.buf, y.coff, half), dst2=g.full(xo)))
    m1 = g.add_buf(key + ".m1", height, width, half)
    wf, bf = fold_conv(sd, key + ".branch_main.0", key + ".branch_main.1")
    g.pw(key + ".m.pw1", g.full(xo), g.full(m1), wf, bf, True)
    m2 = g.add_buf(key + ".m2", height, width, half, signed=True)
    wf, bf = fold_conv(sd, key + ".branch_main.3", key + ".branch_main.4")
    g.dw(key + ".m.dw", g.full(m1), g.full(m2), wf, bf, 3, 1, False)
    wf, bf = fold_conv(sd, key + ".branch_main.5", key + ".branch_main.6")
    g.pw(key + ".m.pw2", g.full(m2), View(y.buf, y.coff + half, half), wf, bf, True)
    return y


def head_block(g, sd, key, name, src):
    h, w = g.bufs[src.buf].h, g.bufs[src.buf].w
    c = HEAD_CHANNELS
    a = g.add_buf(name + ".a", h, w, c)
    wf, bf = fold_conv(sd, key + ".block.0", key + ".block.1")
    g.dw(name + ".dw1", src, g.full(a), wf, bf, 5, 1, True)
    b = g.add_buf(name + ".b", h, w, c, signed=True)
    wf, bf = fold_conv(sd, key + ".block.3", key + ".block.4")
    g.pw(name + ".pw1", g.full(a), g.full(b), wf, bf, False)
    d = g.add_buf(name + ".c", h, w, c)
    wf, bf = fold_conv(sd, key + ".block.5", key + ".block.6")
    g.dw(name + ".dw2", g.full(b), g.full(d), wf, bf, 5, 1, True)
    e = g.add_buf(name + ".d", h, w, c, signed=True)
    wf, bf = fold_conv(sd, key + ".block.8", key + ".block.9")
    g.pw(name + ".pw2", g.full(d), g.full(e), wf, bf, False)
    return g.full(e)


def detection_outputs(g, sd, name, cls_feat, reg_feat):
    h, w = g.bufs[cls_feat.buf].h, g.bufs[cls_feat.buf].w
    outputs = {}
    for label, channels, feat in (("reg", 12, reg_feat), ("obj", 3, cls_feat), ("cls", 80, cls_feat)):
        buf = g.add_buf(f"{name}.{label}", h, w, channels, signed=True, keep=True)
        wf, bf = fold_conv(sd, f"output_{label}_layers")
        g.pw(f"{name}.{label}", feat, g.full(buf), wf, bf, False)
        outputs[label] = buf.id
    return outputs


def build_graph(sd, size):
    assert size % 32 == 0
    g = Graph(size)
    img = g.add_buf("img", size, size, 3, external=True)
    img.scale = 1.0 / 255.0
    x0 = g.add_buf("x0", size // 4, size // 4, STAGE_CHANNELS[0])
    wf, bf = fold_conv(sd, "backbone.first_conv.0", "backbone.first_conv.1")
    wf = np.ascontiguousarray(wf[:, ::-1].transpose(0, 2, 3, 1))
    g.add_op(Op("conv1pool", "conv1pool", g.full(img), g.full(x0), k=3, stride=2, relu=True, wf=wf, bf=bf))

    cur = g.full(x0)
    inp = STAGE_CHANNELS[0]
    p2cat = None
    for stage, repeats in enumerate(STAGE_REPEATS):
        oup = STAGE_CHANNELS[stage + 1]
        for i in range(repeats):
            key = f"backbone.stage{stage + 2}.{i}"
            last_of_c2 = stage == 1 and i == repeats - 1
            out = None
            if last_of_c2:
                half = size // 16
                p2cat = g.add_buf("p2cat", half, half, STAGE_CHANNELS[3] + oup)
                out = View(p2cat.id, STAGE_CHANNELS[3], oup)
            if i == 0:
                cur = shuffle_unit(g, sd, key, cur, inp, oup, 2, out)
            else:
                cur = shuffle_unit(g, sd, key, cur, oup // 2, oup, 1, out)
        inp = oup
    c3 = cur

    s3 = g.add_buf("s3", size // 32, size // 32, HEAD_CHANNELS)
    wf, bf = fold_conv(sd, "fpn.conv1x1_3.0", "fpn.conv1x1_3.1")
    g.pw("fpn.s3", c3, g.full(s3), wf, bf, True)
    cls3 = head_block(g, sd, "fpn.cls_head_3", "cls3", g.full(s3))
    reg3 = head_block(g, sd, "fpn.reg_head_3", "reg3", g.full(s3))
    heads3 = detection_outputs(g, sd, "det3", cls3, reg3)

    g.add_op(Op("upcopy", "fpn.up", c3, View(p2cat.id, 0, STAGE_CHANNELS[3])))
    s2 = g.add_buf("s2", size // 16, size // 16, HEAD_CHANNELS)
    wf, bf = fold_conv(sd, "fpn.conv1x1_2.0", "fpn.conv1x1_2.1")
    g.pw("fpn.s2", g.full(p2cat), g.full(s2), wf, bf, True)
    cls2 = head_block(g, sd, "fpn.cls_head_2", "cls2", g.full(s2))
    reg2 = head_block(g, sd, "fpn.reg_head_2", "reg2", g.full(s2))
    heads2 = detection_outputs(g, sd, "det2", cls2, reg2)
    g.heads = [heads2, heads3]
    return g


def run_float(g, image_rgb_u8):
    import torch
    import torch.nn.functional as F

    arrays = {b.id: np.zeros((b.h, b.w, b.c), np.float32) for b in g.bufs}
    arrays[0][:] = image_rgb_u8.astype(np.float32) / 255.0
    for op in g.ops:
        x = arrays[op.src.buf][:, :, op.src.coff:op.src.coff + op.src.c]
        h, w, c = x.shape
        dst = arrays[op.dst.buf]
        if op.kind == "conv1pool":
            t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None]
            wt = torch.from_numpy(op.wf).permute(0, 3, 1, 2)
            y = F.relu(F.conv2d(t, wt, torch.from_numpy(op.bf), stride=2, padding=1))
            y = F.max_pool2d(y, 3, 2, 1)
            out = y[0].permute(1, 2, 0).numpy()
        elif op.kind == "pw":
            out = (x.reshape(-1, c) @ op.wf.T + op.bf).reshape(h, w, -1)
            if op.relu:
                out = np.maximum(out, 0)
        elif op.kind == "dw":
            t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None]
            wt = torch.from_numpy(np.ascontiguousarray(op.wf.transpose(2, 0, 1)[:, None]))
            y = F.conv2d(t, wt, torch.from_numpy(op.bf), stride=op.stride, padding=op.k // 2, groups=c)
            if op.relu:
                y = F.relu(y)
            out = y[0].permute(1, 2, 0).numpy()
        elif op.kind == "split":
            out = x[:, :, 0::2]
            arrays[op.dst2.buf][:, :, op.dst2.coff:op.dst2.coff + op.dst2.c] = x[:, :, 1::2]
        elif op.kind == "upcopy":
            out = x.repeat(2, axis=0).repeat(2, axis=1)
        else:
            raise ValueError(op.kind)
        dst[:, :, op.dst.coff:op.dst.coff + op.dst.c] = out
    return arrays
