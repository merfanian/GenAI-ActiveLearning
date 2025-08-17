import csv
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import random

from utils.config import AUGMENTED_IMAGES_DIR

_text_column = "text"
_label_column = "label"
_metadata_df = None

_SPLIT_SEED = 42
_TEST_FRAC = 0.20
_HOLDOUT_FRAC = 0.20
_VAL_FRAC = 0.15
_MIN_TEST_SAMPLES_PER_GROUP = 0

AUGMENTED_TEXT_METADATA_CSV = AUGMENTED_IMAGES_DIR / "augmented_text_metadata.csv"


def load_and_validate_dataset(metadata_csv_path: str, target_attribute: str, fairness_attribute: str) -> dict:
    logging.debug(
        f"load_and_validate_dataset called with metadata_csv_path={metadata_csv_path}"
    )
    try:
        df = pd.read_csv(metadata_csv_path)
        logging.debug(
            f"Loaded metadata CSV with columns {df.columns.tolist()} and {len(df)} records"
        )
    except Exception as e:
        logging.debug(f"Failed to read metadata CSV: {e}", exc_info=True)
        return {"error": f"Failed to read metadata CSV: {e}"}

    expected_columns = {"text", target_attribute, fairness_attribute}
    if not expected_columns.issubset(df.columns):
        missing = expected_columns - set(df.columns)
        logging.debug(f"Metadata CSV missing columns: {missing}")
        return {"error": f"Metadata CSV missing columns: {missing}"}

    df = df.rename(columns={target_attribute: "label"})

    global _metadata_df
    logging.debug("Splitting dataset into train/val/holdout/test")
    df["split"] = ""

    # initial test split
    train_val_holdout_idx, test_idx = train_test_split(
        df.index, test_size=_TEST_FRAC, random_state=_SPLIT_SEED
    )

    # ensure a minimum number of test samples per attribute group
    attrs = [c for c in df.columns if c not in ("text", "label", "split")]
    if attrs:
        test_set = set(test_idx)
        remaining_set = set(train_val_holdout_idx)
        rng = random.Random(_SPLIT_SEED)
        for _, group_df in df.groupby(attrs):
            group_indices = set(group_df.index)
            current = test_set & group_indices
            missing = _MIN_TEST_SAMPLES_PER_GROUP - len(current)
            if missing > 0:
                available = list(group_indices - test_set)
                rng.shuffle(available)
                selected = available[:missing]
                test_set.update(selected)
                remaining_set.difference_update(selected)
        test_idx = list(test_set)
        train_val_holdout_idx = list(remaining_set)

    test_frac_actual = len(test_idx) / len(df)
    holdout_frac_rel = _HOLDOUT_FRAC / (1 - test_frac_actual)
    train_val_idx, holdout_idx = train_test_split(
        train_val_holdout_idx, test_size=holdout_frac_rel, random_state=_SPLIT_SEED
    )
    val_frac_rel = _VAL_FRAC / (1 - test_frac_actual - holdout_frac_rel)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_frac_rel, random_state=_SPLIT_SEED
    )
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[holdout_idx, "split"] = "holdout"
    df.loc[test_idx, "split"] = "test"
    logging.info(
        f"Dataset splits: train={len(train_idx)}, val={len(val_idx)}, holdout={len(holdout_idx)}, test={len(test_idx)}"
    )
    _metadata_df = df.copy()
    logging.debug(
        f"Metadata DataFrame shape: {_metadata_df.shape}, columns: {_metadata_df.columns.tolist()}"
    )
    logging.info(
        f"Dataset loaded: metadata_csv={metadata_csv_path}, records={len(_metadata_df)}"
    )
    return {"message": "Dataset loaded successfully."}


def get_current_dataset_info(target_attribute: str, fairness_attribute: str) -> dict:
    logging.debug("get_current_dataset_info called")
    if _metadata_df is None:
        return {"error": "Dataset not loaded."}
    df = _metadata_df
    attrs = [c for c in df.columns if c not in ("text", "label", "split")]
    return {
        "num_texts": len(df),
        "attribute_columns": attrs,
        "num_unique_labels": int(df["label"].nunique()),
        "target_attribute": target_attribute,
        "fairness_attribute": fairness_attribute,
    }


def append_to_augmented_metadata_csv(text: str, attributes: dict, label: str):
    """
    Append a row to the augmented metadata CSV with text, attributes, and label.
    """
    AUGMENTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["text"] + list(attributes.keys()) + ["label"]
    file_exists = AUGMENTED_TEXT_METADATA_CSV.exists()
    with open(AUGMENTED_TEXT_METADATA_CSV, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {"text": text, **attributes, "label": label}
        writer.writerow(row)


def load_augmented_metadata_csv() -> list[dict]:
    """
    Load all rows from the augmented metadata CSV as a list of dicts.
    """
    if not AUGMENTED_TEXT_METADATA_CSV.exists():
        return []
    with open(AUGMENTED_TEXT_METADATA_CSV, mode="r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def add_augmented_data(text: str, attributes: dict, llm_acquired_label: str):
    logging.debug(
        f"add_augmented_data called with generated_text='{text}', attributes={attributes}, llm_acquired_label={llm_acquired_label}"
    )
    append_to_augmented_metadata_csv(text, attributes, llm_acquired_label)


def get_metadata_df() -> pd.DataFrame:
    logging.debug("get_metadata_df called")
    if _metadata_df is None:
        raise ValueError("Dataset not loaded.")
    return _metadata_df


def get_texts_and_labels() -> (list[str], list[str]):
    logging.debug("get_texts_and_labels called")
    if _metadata_df is None:
        raise ValueError("Dataset not loaded.")
    df = _metadata_df
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    logging.debug(f"Returning {len(texts)} texts and labels")
    return texts, labels


def get_train_val_texts_and_labels(
    include_augmented: bool = False,
) -> (list[str], list[str]):
    logging.debug(
        f"get_train_val_texts_and_labels called with include_augmented={include_augmented}"
    )
    if _metadata_df is None:
        raise ValueError("Dataset not loaded.")
    df = _metadata_df[_metadata_df["split"].isin(["train", "val"])]
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    if include_augmented:
        aug_rows = load_augmented_metadata_csv()
        for row in aug_rows:
            texts.append(row["text"])
            labels.append(row["label"])
    logging.debug(
        f"Returning {len(texts)} train/val texts and labels (include_augmented={include_augmented})"
    )
    return texts, labels


def get_test_texts_and_labels() -> (list[str], list[str]):
    logging.debug("get_test_texts_and_labels called")
    if _metadata_df is None:
        raise ValueError("Dataset not loaded.")
    df = _metadata_df[_metadata_df["split"] == "test"]
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    logging.debug(f"Returning {len(texts)} test texts and labels")
    return texts, labels


def get_test_metadata_df() -> pd.DataFrame:
    logging.debug("get_test_metadata_df called")
    if _metadata_df is None:
        raise ValueError("Dataset not loaded.")
    return _metadata_df[_metadata_df["split"] == "test"].copy()


def get_holdout_texts_and_labels() -> (list[str], list[str]):
    logging.debug("get_holdout_texts_and_labels called")
    if _metadata_df is None:
        raise ValueError("Dataset not loaded.")
    df = _metadata_df[_metadata_df["split"] == "holdout"]
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    logging.debug(f"Returning {len(texts)} holdout texts and labels")
    return texts, labels


def get_holdout_metadata_df() -> pd.DataFrame:
    logging.debug("get_holdout_metadata_df called")
    if _metadata_df is None:
        raise ValueError("Dataset not loaded.")
    return _metadata_df[_metadata_df["split"] == "holdout"].copy()
