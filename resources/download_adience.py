import csv
import zipfile
from pathlib import Path
from typing import List
import requests
from tqdm import tqdm

DATA_URL = "https://data.deepai.org/adience.zip"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"File {dest} already exists, skipping download")
        return
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists():
        print(f"Directory {out_dir} already exists, skipping extraction")
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def parse_fold_files(dataset_dir: Path) -> List[dict]:
    entries = []
    for fold_file in sorted(dataset_dir.glob("fold_*_data.txt")):
        with open(fold_file, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            print(fold_file.name)
            for row in reader:
                gender = row.get("gender", "").lower()
                if gender not in {"m", "f"}:
                    continue
                label = "male" if gender == "m" else "female"
                age = row.get("age", "")
                user_id = row.get("user_id")
                face_id = row.get("face_id")
                original_image = row.get("original_image")
                filename = f"aligned/{user_id}/landmark_aligned_face.{face_id}.{original_image}"
                entries.append({"filename": filename, "age": age, "label": label})
    return entries


def create_metadata_csv(entries: List[dict], csv_path: Path) -> None:
    if not entries:
        print("No entries parsed; metadata CSV will not be created")
        return
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["filename", "age", "label"])
        writer.writeheader()
        for row in entries:
            writer.writerow(row)


def main(out_dir: str = "adience") -> None:
    out_path = Path(out_dir)
    # zip_file = out_path / "adience.zip"
    # download_file(DATA_URL, zip_file)
    # extract_zip(zip_file, out_path)
    entries = parse_fold_files(out_path)
    create_metadata_csv(entries, out_path / "adience_metadata.csv")
    print("Done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download the Adience dataset and create metadata CSV")
    parser.add_argument("out_dir", nargs="?", default="adience", help="Output directory")
    args = parser.parse_args()
    main(args.out_dir)
