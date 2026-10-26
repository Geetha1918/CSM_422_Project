"""Download TrashNet and create the train/test folders used by this project.

Source: Gary Thung and Mindy Yang's TrashNet dataset. The original six classes
are mapped to this project's three educational waste categories:
paper/cardboard -> biodegradable; glass/metal/plastic -> recyclable;
trash -> non_recyclable.
"""

import argparse
import random
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
SOURCE_TO_TARGET = {
    "cardboard": "biodegradable",
    "paper": "biodegradable",
    "glass": "recyclable",
    "metal": "recyclable",
    "plastic": "recyclable",
    "trash": "non_recyclable",
}


def find_source_folder(extracted_root: Path, source_name: str) -> Path:
    matches = [path for path in extracted_root.rglob(source_name) if path.is_dir()]
    if not matches:
        raise FileNotFoundError(f"Could not find the '{source_name}' folder in downloaded archive.")
    return matches[0]


def copy_split(files: list[Path], target_name: str, data_dir: Path, test_fraction: float) -> None:
    random.shuffle(files)
    test_count = max(1, round(len(files) * test_fraction))
    for split, subset in (("test", files[:test_count]), ("train", files[test_count:])):
        destination = data_dir / split / target_name
        destination.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(subset):
            # Prefixing avoids name collisions when two source categories merge.
            shutil.copy2(source, destination / f"{source.parent.name}_{index:04d}{source.suffix.lower()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare TrashNet for this project.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if not 0 < arguments.test_fraction < 1:
        raise ValueError("--test-fraction must be between 0 and 1.")

    random.seed(arguments.seed)
    data_dir = Path(arguments.data_dir)
    expected = [data_dir / split / target for split in ("train", "test") for target in set(SOURCE_TO_TARGET.values())]
    if any(path.exists() and any(path.iterdir()) for path in expected):
        raise FileExistsError("Dataset folders already contain files. Use an empty data directory to avoid overwriting images.")

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        archive_path = temporary_path / "trashnet.zip"
        print("Downloading TrashNet (about 41 MB)...")
        urllib.request.urlretrieve(DATASET_URL, archive_path)
        print("Extracting and creating reproducible train/test split...")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temporary_path / "extracted")
        extracted_root = temporary_path / "extracted"
        for source_name, target_name in SOURCE_TO_TARGET.items():
            source_folder = find_source_folder(extracted_root, source_name)
            images = [path for path in source_folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            copy_split(images, target_name, data_dir, arguments.test_fraction)

    print("Dataset ready.")
    for split in ("train", "test"):
        for target in sorted(set(SOURCE_TO_TARGET.values())):
            print(f"{split}/{target}: {len(list((data_dir / split / target).glob('*')))} images")


if __name__ == "__main__":
    main()
