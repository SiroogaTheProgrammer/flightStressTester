import argparse
import gzip
import struct
import tempfile
import time
import urllib.request
from pathlib import Path

import serial
from PIL import Image, ImageOps


MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_IMAGES = "t10k-images-idx3-ubyte.gz"
MNIST_LABELS = "t10k-labels-idx1-ubyte.gz"


def encode_image(path):
    with Image.open(path) as source:
        grayscale = ImageOps.invert(source.convert("L"))
        fitted = ImageOps.contain(grayscale, (28, 28), method=Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(fitted, ((28 - fitted.width) // 2, (28 - fitted.height) // 2))
        return canvas.tobytes()


def load_mnist_test(cache_dir, start_index, count):
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_path = cache_dir / MNIST_IMAGES
    label_path = cache_dir / MNIST_LABELS
    for filename, path in ((MNIST_IMAGES, image_path), (MNIST_LABELS, label_path)):
        if not path.exists():
            print(f"Downloading {filename}")
            urllib.request.urlretrieve(MNIST_URL + filename, path)

    with gzip.open(image_path, "rb") as dataset:
        magic, image_count, rows, columns = struct.unpack(">IIII", dataset.read(16))
        image_data = dataset.read()
    if magic != 2051 or rows != 28 or columns != 28:
        raise ValueError("Downloaded MNIST image file has an unexpected header.")

    with gzip.open(label_path, "rb") as dataset:
        label_magic, label_count = struct.unpack(">II", dataset.read(8))
        labels = dataset.read()
    if label_magic != 2049 or image_count != label_count or len(labels) != label_count:
        raise ValueError("Downloaded MNIST label file has an unexpected header or length.")
    if start_index < 0 or count < 1 or start_index + count > image_count:
        raise ValueError(f"Choose a range within the {image_count} MNIST test images.")

    samples = []
    for index in range(start_index, start_index + count):
        offset = index * rows * columns
        pixels = image_data[offset:offset + rows * columns]
        samples.append((pixels, labels[index]))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Stream digit images to the Pico 2 W over USB serial.")
    parser.add_argument("image", nargs="?", type=Path, help="Optional handwritten image file")
    parser.add_argument("--port", required=True, help="USB serial port, for example COM5")
    parser.add_argument("--repeat", type=int, default=1, help="Number of inferences; use 0 to run continuously")
    parser.add_argument("--mnist-test", action="store_true", help="Download and stream labeled MNIST test images")
    parser.add_argument("--count", type=int, default=100, help="Number of MNIST test images to stream")
    parser.add_argument("--start-index", type=int, default=0, help="First MNIST test image index (0-9999)")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "pico_digit_mnist",
        help="Directory for cached MNIST files",
    )
    args = parser.parse_args()
    if args.mnist_test == (args.image is not None):
        parser.error("provide an image path or use --mnist-test")
    if args.repeat < 0 or args.count < 1:
        parser.error("--repeat must be zero or greater")

    samples = load_mnist_test(args.cache_dir, args.start_index, args.count) if args.mnist_test else None
    pixels = None if args.mnist_test else encode_image(args.image)
    started = time.perf_counter()
    received = 0
    correct = 0
    with serial.Serial(args.port, 115200, timeout=10, write_timeout=10) as port:
        port.write(b"HELLO\n")
        port.flush()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            line = port.readline().decode("ascii", errors="replace").strip()
            if line.startswith("READY,"):
                break
        else:
            raise TimeoutError("Pico did not report READY; check the USB connection and firmware.")

        print("seq,predicted,expected,correct,confidence,latency_us,average_us,max_us,total")
        try:
            sample_index = 0
            while (args.mnist_test and sample_index < len(samples)) or (
                not args.mnist_test and (args.repeat == 0 or received < args.repeat)
            ):
                expected = None
                if args.mnist_test:
                    pixels, expected = samples[sample_index]
                    sample_index += 1
                port.write(b"IMG\n" + pixels)
                port.flush()
                while True:
                    line = port.readline().decode("ascii", errors="replace").strip()
                    if line.startswith(("RESULT,", "DROP,", "ERROR,")):
                        if line.startswith("RESULT,"):
                            fields = line.split(",")
                            prediction = int(fields[2])
                            is_correct = expected is not None and prediction == expected
                            correct += is_correct
                            expected_text = str(expected) if expected is not None else ""
                            correct_text = "yes" if is_correct else ("no" if expected is not None else "")
                            print(
                                f"{fields[1]},{prediction},{expected_text},{correct_text},"
                                f"{fields[3]},{fields[4]},{fields[5]},{fields[6]},{fields[7]}"
                            )
                            received += 1
                        else:
                            print(line)
                        break
                    if not line:
                        raise TimeoutError("Timed out waiting for a Pico result.")
        except KeyboardInterrupt:
            pass

    elapsed = time.perf_counter() - started
    if received:
        print(f"Completed {received} inferences in {elapsed:.2f}s ({received / elapsed:.2f} results/s).")
        if args.mnist_test:
            print(f"MNIST accuracy: {correct}/{received} ({100 * correct / received:.2f}%).")


if __name__ == "__main__":
    main()