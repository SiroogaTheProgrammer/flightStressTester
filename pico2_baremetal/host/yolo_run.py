import argparse
import tempfile
import time
import urllib.request
from pathlib import Path


SAMPLE_IMAGES = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
}


def download_samples():
    sample_dir = Path(tempfile.gettempdir()) / "pico_yolo_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, url in SAMPLE_IMAGES.items():
        image_path = sample_dir / filename
        if not image_path.exists():
            print(f"Downloading {filename}")
            urllib.request.urlretrieve(url, image_path)
        paths.append(image_path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Run real YOLO object detection on selected images on this PC.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", nargs="+", type=Path, help="Image files to run")
    inputs.add_argument("--download-samples", action="store_true", help="Download and run two sample images")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model or weights path")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size; larger sizes use more compute")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum detection confidence")
    parser.add_argument("--device", default="cpu", help="Inference device, for example cpu or 0")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/yolo_pc"))
    args = parser.parse_args()
    if args.imgsz < 32 or not 0.0 <= args.conf <= 1.0:
        parser.error("--imgsz must be at least 32 and --conf must be between 0 and 1")

    image_paths = download_samples() if args.download_samples else args.images
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        parser.error("image file not found: " + ", ".join(str(path) for path in missing))

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Install YOLO dependencies with: python -m pip install -r requirements-yolo.txt") from error

    model_started = time.perf_counter()
    model = YOLO(args.model)
    model_load_seconds = time.perf_counter() - model_started
    run_started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        stream=True,
        save=True,
        project=str(output_dir),
        name="predict",
        exist_ok=True,
        verbose=False,
    )

    image_count = 0
    detection_count = 0
    for result in results:
        image_count += 1
        boxes = result.boxes
        box_count = 0 if boxes is None else len(boxes)
        detection_count += box_count
        print(f"IMAGE,{result.path},{box_count}")
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
                print(
                    f"DETECTION,{result.names[class_id]},{confidence:.4f},"
                    f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}"
                )
        print(
            f"YOLO_SPEED_MS_PER_IMAGE,{result.speed['preprocess']:.2f},"
            f"{result.speed['inference']:.2f},{result.speed['postprocess']:.2f}"
        )

    elapsed = time.perf_counter() - run_started
    print(f"MODEL_LOAD_SECONDS,{model_load_seconds:.2f}")
    print(f"TOTAL_IMAGES,{image_count}")
    print(f"TOTAL_DETECTIONS,{detection_count}")
    if elapsed > 0 and image_count:
        print(f"HOST_IMAGES_PER_SECOND,{image_count / elapsed:.2f}")
    print(f"ANNOTATED_IMAGES,{output_dir / 'predict'}")


if __name__ == "__main__":
    main()