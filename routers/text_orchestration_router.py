import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import text_data_service, text_model_service, text_augmentation_service, fairness_service
from utils.config import TRAINED_MODELS_DIR

router = APIRouter()

def _serialize_group_performance(attrs_tuple, metrics):
    attributes = dict(attrs_tuple)
    return {"attributes": attributes, "accuracy": metrics.get("accuracy"), "count": metrics.get("count")}

def _serialize_worst_group(worst_raw):
    if not worst_raw:
        return {}
    return {"attributes": worst_raw.get("attributes", {}), "accuracy": worst_raw.get("accuracy"), "count": worst_raw.get("count")}

class SetupDatasetRequest(BaseModel):
    metadata_csv_path: str
    target_attribute: str
    fairness_attribute: str

@router.post("/text/setup_dataset")
def setup_dataset(req: SetupDatasetRequest):
    result = text_data_service.load_and_validate_dataset(req.metadata_csv_path, req.target_attribute, req.fairness_attribute)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

class TrainInitialModelRequest(BaseModel):
    use_augmented: bool = False

@router.post("/text/train_initial_model")
def train_initial_model(req: TrainInitialModelRequest):
    try:
        texts, labels = text_data_service.get_train_val_texts_and_labels(req.use_augmented)
        model_path = text_model_service.train_model(texts, labels)
        text_model_service.set_current_model_path(model_path)
        return {"message": "Initial text model trained.", "model_path": model_path}
    except Exception as e:
        logging.exception("Error training initial text model")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/text/evaluate_fairness")
def evaluate_fairness(model_name: str, fairness_attribute: str):
    try:
        model_file = TRAINED_MODELS_DIR / model_name
        if not model_file.exists():
            raise HTTPException(status_code=400, detail="Model not found.")
        
        df = text_data_service.get_test_metadata_df()
        texts, labels = text_data_service.get_test_texts_and_labels()
        
        # Note: fairness_service was designed for images, but predict is generic if the model service handles it.
        # We need a text-compatible version of calculate_group_performances.
        # For now, let's assume a generic fairness service or adapt it.
        # This is a placeholder for demonstration. A proper implementation would adapt fairness_service.
        
        # This is a simplified path for now.
        preds = text_model_service.predict(str(model_file), texts)
        df["predicted_label"] = [p["predicted_label"] for p in preds]
        
        group_perfs = {}
        groups = df.groupby(fairness_attribute) if fairness_attribute in df.columns else [((), df)]

        for group_vals, group_df in groups:
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            
            attrs = [fairness_attribute] if isinstance(fairness_attribute, str) else fairness_attribute
            key = tuple(zip(attrs, group_vals))
            correct = (group_df["predicted_label"].astype(str) == group_df["label"].astype(str)).sum()
            count = len(group_df)
            acc = correct / count if count else 0.0
            group_perfs[key] = {"accuracy": acc, "count": count}

        serialized_gp = [_serialize_group_performance(attrs, metrics) for attrs, metrics in group_perfs.items()]
        worst_raw = fairness_service.find_worst_performing_group(group_perfs)
        serialized_worst = _serialize_worst_group(worst_raw)

        return {"all_group_performances": serialized_gp, "worst_performing_group": serialized_worst}
    except Exception as e:
        logging.exception("Error evaluating text model fairness")
        raise HTTPException(status_code=500, detail=str(e))

class AugmentationCycleRequest(BaseModel):
    augmentation_budget_remaining: int
    accuracy_threshold: float
    augmentation_batch_size: int = 5
    fairness_attribute: str

@router.post("/text/run_iterative_augmentation_cycle")
def run_iterative_augmentation_cycle(req: AugmentationCycleRequest):
    try:
        model_path = text_model_service.get_current_model_path()
        if not model_path:
            raise HTTPException(status_code=400, detail="No text model trained.")

        # Simplified augmentation loop
        generated = text_augmentation_service.generate_text_and_get_label({}, num_texts=req.augmentation_batch_size)
        for item in generated:
            text_data_service.add_augmented_data(item["text"], item["attributes_used"], item["llm_acquired_label"])

        all_texts, all_labels = text_data_service.get_train_val_texts_and_labels(include_augmented=True)
        new_model_path = text_model_service.train_model(all_texts, all_labels, existing_model_path=model_path)
        text_model_service.set_current_model_path(new_model_path)

        return {"status": "Text augmentation cycle complete.", "new_model_path": new_model_path}
    except Exception as e:
        logging.exception("Error in text augmentation cycle")
        raise HTTPException(status_code=500, detail=str(e))
