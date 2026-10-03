import argparse
import gzip
import struct
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier


MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


def download_dataset(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        destination = cache_dir / filename
        if not destination.exists():
            print(f"Downloading {filename}")
            urllib.request.urlretrieve(MNIST_URL + filename, destination)
    return cache_dir


def read_images(path):
    with gzip.open(path, "rb") as dataset:
        magic, count, rows, columns = struct.unpack(">IIII", dataset.read(16))
        if magic != 2051 or rows != 28 or columns != 28:
            raise ValueError(f"Unexpected image file header: {path}")
        return np.frombuffer(dataset.read(), dtype=np.uint8).reshape(count, rows * columns)


def read_labels(path):
    with gzip.open(path, "rb") as dataset:
        magic, count = struct.unpack(">II", dataset.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label file header: {path}")
        return np.frombuffer(dataset.read(), dtype=np.uint8).reshape(count)


def write_array(output, name, values):
    output.write(f"inline constexpr float {name}[] = {{\n")
    flat_values = np.asarray(values, dtype=np.float32).reshape(-1)
    for start in range(0, flat_values.size, 8):
        row = ", ".join(f"{value:.9g}f" for value in flat_values[start:start + 8])
        output.write(f"    {row},\n")
    output.write("};\n\n")


def main():
    parser = argparse.ArgumentParser(description="Train and export the Pico digit model from MNIST.")
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "pico_digit_mnist")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "digit_model.h")
    args = parser.parse_args()

    download_dataset(args.cache_dir)
    train_images = read_images(args.cache_dir / FILES[0]).astype(np.float32) / 255.0
    train_labels = read_labels(args.cache_dir / FILES[1])
    test_images = read_images(args.cache_dir / FILES[2]).astype(np.float32) / 255.0
    test_labels = read_labels(args.cache_dir / FILES[3])

    model = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        batch_size=256,
        max_iter=args.iterations,
        early_stopping=True,
        n_iter_no_change=4,
        random_state=7,
        verbose=True,
    )
    model.fit(train_images, train_labels)
    accuracy = model.score(test_images, test_labels)
    print(f"MNIST test accuracy: {accuracy:.4%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii", newline="\n") as output:
        output.write("#pragma once\n\n")
        write_array(output, "kLayer1Weights", model.coefs_[0].T)
        write_array(output, "kHiddenBiases", model.intercepts_[0])
        write_array(output, "kLayer2Weights", model.coefs_[1].T)
        write_array(output, "kOutputBiases", model.intercepts_[1])
    print(f"Wrote model weights to {args.output}")


if __name__ == "__main__":
    main()