import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import data_service, model_service, fairness_service, llm_augmentation_service
from utils.config import AUGMENTED_IMAGES_DIR


def _serialize_group_performance(attrs_tuple, metrics):
    # Convert numpy types to Python builtins and prepare JSON-friendly dict
    attributes = {}
    for key, val in attrs_tuple:
        if hasattr(val, "item"):
            val = val.item()
        attributes[key] = val
    acc = metrics.get("accuracy")
    cnt = metrics.get("count")
    if hasattr(acc, "item"):
        acc = acc.item()
    if hasattr(cnt, "item"):
        cnt = cnt.item()
    return {"attributes": attributes, "accuracy": acc, "count": cnt}


def _serialize_worst_group(worst_raw):
    if not worst_raw:
        return {}
    attributes = {}
    for key, val in worst_raw.get("attributes", {}).items():
        if hasattr(val, "item"):
            val = val.item()
        attributes[key] = val
    acc = worst_raw.get("accuracy")
    cnt = worst_raw.get("count")
    if hasattr(acc, "item"):
        acc = acc.item()
    if hasattr(cnt, "item"):
        cnt = cnt.item()
    return {"attributes": attributes, "accuracy": acc, "count": cnt}


router = APIRouter()


class SetupDatasetRequest(BaseModel):
    image_dir_path: str
    metadata_csv_path: str


@router.post("/setup_dataset")
def setup_dataset(req: SetupDatasetRequest):
    logging.debug(f"setup_dataset called with request={req.dict()}")
    result = data_service.load_and_validate_dataset(req.image_dir_path, req.metadata_csv_path)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    logging.debug(f"setup_dataset result: {result}")
    return result


class TrainInitialModelRequest(BaseModel):
    use_augmented: bool = False


@router.post("/train_initial_model")
def train_initial_model(req: TrainInitialModelRequest):
    logging.debug(f"train_initial_model called with use_augmented={req.use_augmented}")
    try:
        image_paths, labels = data_service.get_train_val_image_paths_and_labels(req.use_augmented)
        logging.debug(
            f"Retrieved train/val image paths and labels: {len(image_paths)} paths (include_augmented={req.use_augmented})")
        model_path = model_service.train_model(image_paths, labels)
        model_service.set_current_model_path(model_path)
        logging.debug(f"Initial model trained, model_path={model_path}")
        return {"message": "Initial model trained.", "model_path": model_path, "use_augmented": req.use_augmented}
    except Exception as e:
        logging.exception("Error training initial model")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluate_fairness")
def evaluate_fairness():
    logging.debug("evaluate_fairness called")
    try:
        model_path = model_service.get_current_model_path()
        logging.debug(f"Current model_path: {model_path}")
        if not model_path:
            raise HTTPException(status_code=400, detail="No model trained.")
        df = data_service.get_test_metadata_df()
        logging.debug(f"Test metadata DataFrame obtained: {len(df)} records")
        image_paths, _ = data_service.get_test_image_paths_and_labels()
        logging.debug(f"Test image paths count: {len(image_paths)}")
        gp_raw = fairness_service.calculate_group_performances(model_path, df, image_paths)
        logging.debug(f"Group performances raw: {gp_raw}")
        serialized_gp = [_serialize_group_performance(attrs, metrics) for attrs, metrics in gp_raw.items()]
        worst_raw = fairness_service.find_worst_performing_group(gp_raw)
        logging.debug(f"Worst performing group raw: {worst_raw}")
        serialized_worst = _serialize_worst_group(worst_raw)
        logging.debug(
            f"evaluate_fairness result: all_group_performances={serialized_gp}, worst_performing_group={serialized_worst}")
        return {"all_group_performances": serialized_gp, "worst_performing_group": serialized_worst}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error evaluating fairness")
        raise HTTPException(status_code=500, detail=str(e))


class AugmentationCycleRequest(BaseModel):
    augmentation_budget_remaining: int
    accuracy_threshold: float
    augmentation_batch_size: int = 5


@router.post("/run_iterative_augmentation_cycle")
def run_iterative_augmentation_cycle(req: AugmentationCycleRequest):
    logging.debug(f"run_iterative_augmentation_cycle called with request={req.dict()}")
    try:
        model_path = model_service.get_current_model_path()
        logging.debug(f"Current model_path: {model_path}")
        if not model_path:
            raise HTTPException(status_code=400, detail="No model trained.")
        df = data_service.get_test_metadata_df()
        logging.debug(f"Test metadata DataFrame obtained: {len(df)} records")
        image_paths, _ = data_service.get_test_image_paths_and_labels()
        logging.debug(f"Test image paths count: {len(image_paths)}")
        gp_raw = fairness_service.calculate_group_performances(model_path, df, image_paths)
        logging.debug(f"Group performances raw: {gp_raw}")
        worst_raw = fairness_service.find_worst_performing_group(gp_raw)
        logging.debug(f"Worst performing group raw: {worst_raw}")
        worst_acc = worst_raw.get("accuracy", 0.0).item() if hasattr(worst_raw.get("accuracy", 0.0),
                                                                     "item") else worst_raw.get("accuracy", 0.0)
        logging.debug(
            f"Worst accuracy: {worst_acc}, threshold: {req.accuracy_threshold}, budget_remaining: {req.augmentation_budget_remaining}")
        if worst_acc >= req.accuracy_threshold or req.augmentation_budget_remaining <= 0:
            logging.debug("Stopping augmentation cycle: condition met, no new augmentation")
            return {"status": "Stopping: Condition met."}
        num_to_generate = min(req.augmentation_batch_size, req.augmentation_budget_remaining)
        logging.debug(f"Number of images to generate: {num_to_generate}")
        generated = llm_augmentation_service.generate_image_and_get_label(worst_raw["attributes"], num_to_generate)
        logging.debug(f"Generated augmented items: {generated}")
        for item in generated:
            data_service.add_augmented_data(item["image_path"], item["attributes_used"], item["llm_acquired_label"])
        logging.debug("Added augmented data to dataset")
        remaining = req.augmentation_budget_remaining - num_to_generate
        all_image_paths, all_labels = data_service.get_train_val_image_paths_and_labels(include_augmented=True)
        logging.debug(f"Retraining model with augmented data: total images={len(all_image_paths)}")
        new_model_path = model_service.train_model(all_image_paths, all_labels, existing_model_path=model_path)
        logging.debug(f"New model trained at path: {new_model_path}")
        model_service.set_current_model_path(new_model_path)
        serialized_worst = _serialize_worst_group(worst_raw)
        logging.debug(
            f"run_iterative_augmentation_cycle result: worst_group_before_aug={serialized_worst}, augmented_images_generated={num_to_generate}, new_model_path={new_model_path}, augmentation_budget_remaining={remaining}")
        return {
            "status": "Iteration complete",
            "worst_group_before_aug": serialized_worst,
            "augmented_images_generated": num_to_generate,
            "new_model_path": new_model_path,
            "augmentation_budget_remaining": remaining
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in augmentation cycle")
        raise HTTPException(status_code=500, detail=str(e))
