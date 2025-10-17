import pandas as pd
from sklearn.model_selection import train_test_split
import logging

_train_df = None
_val_df = None
_test_df = None
_augmented_df = None


def load_and_validate_dataset(config: dict):
    """
    Loads and validates the dataset based on the provided configuration.
    """
    global _train_df, _val_df, _test_df, _augmented_df
    
    metadata_csv_path = config["metadata_csv_path"]
    columns = config.get("columns")
    text_column = config["text_column"]
    target_attribute = config["target_attribute"]
    fairness_attribute = config["fairness_attribute"]
    label_mapping = config.get("label_mapping")
    target_fairness_equal = config["target_fairness_equal"]

    logging.info(f"Loading dataset from {metadata_csv_path}...")
    try:
        df = pd.read_csv(metadata_csv_path, header=None if columns else 'infer')
        if columns:
            df.columns = columns
        if target_fairness_equal:
            df = df[[text_column, target_attribute]]
            config[fairness_attribute] = "label"
        else:
            df = df[[text_column, target_attribute, fairness_attribute]]
        df.rename(columns={text_column: 'text', target_attribute: 'label'}, inplace=True)


    except Exception as e:
        logging.error(f"Failed to load or process CSV: {e}")
        return {"error": str(e)}

    if 'text' not in df.columns or 'label' not in df.columns:
        error_msg = f"Required columns 'text' and/or 'label' not in the processed DataFrame."
        logging.error(error_msg)
        return {"error": error_msg}

    # Split the data
    train_val_df, _test_df = train_test_split(df, test_size=0.2)
    _train_df, _val_df = train_test_split(train_val_df, test_size=0.1)
    
    _augmented_df = pd.DataFrame(columns=df.columns)
    
    logging.info(f"Dataset loaded: {_train_df.shape[0]} train, {_val_df.shape[0]} validation, {_test_df.shape[0]} test samples.")
    return {"message": "Dataset loaded and split successfully."}

def get_train_metadata_df():
    return _train_df.copy()

def get_train_val_texts_and_labels(include_augmented: bool = False):
    """
    Returns the training and validation texts and labels.
    Optionally includes augmented data.
    """
    df_to_use = pd.concat([_train_df, _val_df])
    if include_augmented and not _augmented_df.empty:
        df_to_use = pd.concat([_augmented_df, df_to_use])
    
    return df_to_use['text'].tolist(), df_to_use['label'].tolist()

def get_test_metadata_df():
    return _test_df.copy()

def get_test_texts_and_labels():
    """
    Returns the test texts and labels.
    """
    return _test_df['text'].tolist(), _test_df['label'].tolist()

def add_augmented_data(new_data: list[dict], config: dict):
    """
    Adds new augmented text data to the internal store.
    `new_data` is a list of dicts, e.g., [{'text': '...', 'label': ...}]
    """
    global _augmented_df
    new_df = pd.DataFrame(new_data)
    new_df.rename(columns={config["text_column"]: 'text', config["target_attribute"]: 'label'}, inplace=True)
    if config.get("label_mapping"):
        # Check if labels are already integers
        if new_df['label'].dtype != 'int':
            label_to_int = {v: k for k, v in config["label_mapping"].items()}
            new_df['label'] = new_df['label'].map(label_to_int)
    _augmented_df = pd.concat([_augmented_df, new_df], ignore_index=True)
    logging.info(f"Added {len(new_data)} augmented samples. Total augmented data: {_augmented_df.shape[0]}")

def remove_last_augmented_batch(batch_size: int):
    """
    Removes the last N samples from the augmented data store.
    """
    global _augmented_df
    if not _augmented_df.empty and len(_augmented_df) >= batch_size:
        _augmented_df = _augmented_df.iloc[:-batch_size]
        logging.info(f"Removed last {batch_size} augmented samples.")

def get_final_training_df():
    """
    Returns the final training dataframe including original and augmented data.
    """
    return pd.concat([_train_df, _val_df, _augmented_df], ignore_index=True)


def clear_augmented_data():
    """
    Clears all augmented data.
    """
    global _augmented_df
    _augmented_df = pd.DataFrame(columns=_train_df.columns)
    logging.info("Cleared all augmented data.")
