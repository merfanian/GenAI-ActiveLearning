import logging
import pandas as pd
from pathlib import Path

_image_dir_path = None
_metadata_df = None

def load_and_validate_dataset(image_dir_path: str, metadata_csv_path: str) -> dict:
    try:
        df = pd.read_csv(metadata_csv_path)
    except Exception as e:
        return {"error": f"Failed to read metadata CSV: {e}"}
    expected_columns = {"filename", "label"}
    if not expected_columns.issubset(df.columns):
        missing = expected_columns - set(df.columns)
        return {"error": f"Metadata CSV missing columns: {missing}"}
    filenames = df["filename"].tolist()
    missing_files = [f for f in filenames if not Path(image_dir_path, f).exists()]
    if missing_files:
        return {"error": f"Image files not found for filenames: {missing_files}"}
    global _image_dir_path, _metadata_df
    _image_dir_path = image_dir_path
    _metadata_df = df.copy()
    logging.info(f"Dataset loaded: directory={image_dir_path}, metadata_csv={metadata_csv_path}, records={len(_metadata_df)}")
    return {"message": "Dataset loaded successfully."}

def get_current_dataset_info() -> dict:
    if _metadata_df is None or _image_dir_path is None:
        return {"error": "Dataset not loaded."}
    df = _metadata_df
    attrs = [c for c in df.columns if c not in ("filename", "label")]
    return {"num_images": len(df), "attribute_columns": attrs, "num_unique_labels": int(df["label"].nunique())}

def add_augmented_data(generated_image_path: str, attributes: dict, llm_acquired_label: str):
    global _metadata_df
    if _metadata_df is None or _image_dir_path is None:
        raise ValueError("Dataset not loaded.")
    df = _metadata_df
    attrs = [c for c in df.columns if c not in ("filename", "label")]
    new_row = {c: attributes.get(c) for c in attrs}
    new_row["filename"] = generated_image_path
    new_row["label"] = llm_acquired_label
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _metadata_df = df

def get_metadata_df() -> pd.DataFrame:
    if _metadata_df is None:
        raise ValueError("Dataset not loaded.")
    return _metadata_df

def get_image_paths_and_labels() -> (list[str], list[str]):
    if _metadata_df is None or _image_dir_path is None:
        raise ValueError("Dataset not loaded.")
    df = _metadata_df
    paths = []
    for f in df["filename"].tolist():
        original = Path(_image_dir_path) / f
        if original.exists():
            paths.append(str(original))
        else:
            paths.append(str(Path(f)))
    labels = df["label"].tolist()
    return paths, labels