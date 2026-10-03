import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

REPO_URL = "https://github.com/dog-qiuqiu/Yolo-FastestV2.git"
CACHE_DIR = Path(tempfile.gettempdir()) / "pico_yolo_cache"
WEIGHTS_FILE = "modelzoo/coco2017-0.241078ap-model.pth"
NUM_CLASSES = 80
NUM_ANCHORS = 3
TRAIN_SIZE = 352
ANCHORS = np.array(
    [12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 126.91, 78.23, 131.57, 214.55, 279.92, 258.87],
    dtype=np.float32,
).reshape(2, NUM_ANCHORS, 2)


def ensure_repo():
    repo = CACHE_DIR / "Yolo-FastestV2"
    if not repo.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=True)
    return repo


def class_names():
    return (ensure_repo() / "data" / "coco.names").read_text().split("\n")[:NUM_CLASSES]


def load_float_model():
    import torch

    repo = ensure_repo()
    if "torchsummary" not in sys.modules:
        stub = types.ModuleType("torchsummary")
        stub.summary = lambda *args, **kwargs: None
        sys.modules["torchsummary"] = stub
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from model.detector import Detector

    model = Detector(NUM_CLASSES, NUM_ANCHORS, True)
    model.load_state_dict(torch.load(repo / WEIGHTS_FILE, map_location="cpu"))
    return model.eval()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_heads(heads, size):
    """heads: list of (reg[H,W,12], obj[H,W,3], cls[H,W,80]) float arrays per scale, coarse to fine order as in the model."""
    candidates = []
    for scale, (reg, obj, cls) in enumerate(heads):
        height, width = reg.shape[:2]
        stride = size / height
        reg = reg.reshape(height, width, NUM_ANCHORS, 4)
        grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
        probs = np.exp(cls - cls.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        best_class = probs.argmax(axis=-1)
        best_prob = probs.max(axis=-1)
        for anchor in range(NUM_ANCHORS):
            score = sigmoid(obj[..., anchor]) * best_prob
            cx = (sigmoid(reg[..., anchor, 0]) * 2.0 - 0.5 + grid_x) * stride
            cy = (sigmoid(reg[..., anchor, 1]) * 2.0 - 0.5 + grid_y) * stride
            bw = (sigmoid(reg[..., anchor, 2]) * 2.0) ** 2 * ANCHORS[scale, anchor, 0]
            bh = (sigmoid(reg[..., anchor, 3]) * 2.0) ** 2 * ANCHORS[scale, anchor, 1]
            boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, score, best_class], axis=-1)
            candidates.append(boxes.reshape(-1, 6))
    return np.concatenate(candidates, axis=0)


def nms(candidates, conf_threshold, iou_threshold, max_detections=100):
    candidates = candidates[candidates[:, 4] > conf_threshold]
    order = np.argsort(-candidates[:, 4], kind="stable")
    candidates = candidates[order]
    kept = []
    for box in candidates:
        suppressed = False
        for other in kept:
            if other[5] != box[5]:
                continue
            iw = min(box[2], other[2]) - max(box[0], other[0])
            ih = min(box[3], other[3]) - max(box[1], other[1])
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            union = (box[2] - box[0]) * (box[3] - box[1]) + (other[2] - other[0]) * (other[3] - other[1]) - inter
            if inter / union > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(box)
            if len(kept) >= max_detections:
                break
    return np.array(kept, dtype=np.float32).reshape(-1, 6)


def float_heads(model, image_bgr_resized):
    import torch

    x = torch.from_numpy(image_bgr_resized.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    with torch.no_grad():
        out = model(x)
    heads = []
    for i in range(2):
        reg, obj, cls = (out[3 * i + k][0].permute(1, 2, 0).numpy() for k in range(3))
        heads.append((reg, obj, cls))
    return heads
