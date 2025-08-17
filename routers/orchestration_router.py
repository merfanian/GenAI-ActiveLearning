import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import data_service, model_service, fairness_service, llm_augmentation_service
from utils.config import AUGMENTED_IMAGES_DIR, TRAINED_MODELS_DIR


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
    target_attribute: str
    fairness_attribute: str


@router.post("/setup_dataset")
def setup_dataset(req: SetupDatasetRequest):
    logging.debug(f"setup_dataset called with request={req.dict()}")
    result = data_service.load_and_validate_dataset(req.image_dir_path, req.metadata_csv_path, req.target_attribute, req.fairness_attribute)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    logging.debug(f"setup_dataset result: {result}")
    return result


class TrainInitialModelRequest(BaseModel):
    use_augmented: bool = False
    architecture: str = "resnet"


@router.post("/train_initial_model")
def train_initial_model(req: TrainInitialModelRequest):
    logging.debug(f"train_initial_model called with use_augmented={req.use_augmented}, architecture={req.architecture}")
    try:
        image_paths, labels = data_service.get_train_val_image_paths_and_labels(req.use_augmented)
        logging.debug(
            f"Retrieved train/val image paths and labels: {len(image_paths)} paths (include_augmented={req.use_augmented})")
        model_path = model_service.train_model(image_paths, labels, architecture=req.architecture)
        model_service.set_current_model_path(model_path)
        logging.debug(f"Initial model trained, model_path={model_path}")
        return {"message": "Initial model trained.", "model_path": model_path, "use_augmented": req.use_augmented, "architecture": req.architecture}
    except Exception as e:
        logging.exception("Error training initial model")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluate_fairness")
def evaluate_fairness(model_name: str, fairness_attribute: str):
    logging.debug(f"evaluate_fairness called with model_name={model_name}")
    try:
        model_file = TRAINED_MODELS_DIR / model_name
        logging.debug(f"Resolved model_path: {model_file}")
        if not model_file.exists():
            raise HTTPException(status_code=400, detail="Model not found.")
        model_path = str(model_file)
        df = data_service.get_test_metadata_df()
        logging.debug(f"Test metadata DataFrame obtained: {len(df)} records")
        image_paths, labels = data_service.get_test_image_paths_and_labels()
        logging.debug(f"Test image paths count: {len(image_paths)}")
        gp_raw = fairness_service.calculate_group_performances(model_path, df, image_paths, labels, fairness_attribute)
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
    exploration_steepness: float = 10.0
    target_attribute: str
    fairness_attribute: str


@router.post("/run_iterative_augmentation_cycle")
def run_iterative_augmentation_cycle(req: AugmentationCycleRequest):
    logging.debug(f"run_iterative_augmentation_cycle called with request={req.dict()}")
    try:
        model_path = model_service.get_current_model_path()
        if not model_path:
            raise HTTPException(status_code=400, detail="No model trained.")

        budget_remaining = req.augmentation_budget_remaining
        iteration_results = []

        while budget_remaining > 0:
            logging.debug(f"Starting augmentation iteration. Budget remaining: {budget_remaining}")

            df = data_service.get_test_metadata_df()
            image_paths, labels = data_service.get_test_image_paths_and_labels()
            gp_raw = fairness_service.calculate_group_performances(model_path, df, image_paths, labels, req.fairness_attribute)
            worst_raw = fairness_service.find_worst_performing_group(gp_raw)
            worst_acc = worst_raw.get("accuracy", 0.0)
            if hasattr(worst_acc, "item"):
                worst_acc = worst_acc.item()

            logging.debug(f"Worst accuracy: {worst_acc}, threshold: {req.accuracy_threshold}")

            if worst_acc >= req.accuracy_threshold:
                logging.info("Stopping augmentation cycle: accuracy threshold met.")
                break

            num_to_generate = min(req.augmentation_batch_size, budget_remaining)
            logging.debug(f"Number of images to generate: {num_to_generate}")

            generated = llm_augmentation_service.generate_image_and_get_label(
                worst_raw["attributes"], num_to_generate, exploration_steepness=req.exploration_steepness
            )
            for item in generated:
                data_service.add_augmented_data(item["filename"], item["attributes_used"], item["llm_acquired_label"])

            budget_remaining -= num_to_generate
            all_image_paths, all_labels = data_service.get_train_val_image_paths_and_labels(include_augmented=True)
            new_model_path = model_service.train_model(all_image_paths, all_labels, existing_model_path=model_path,
                                                       updated_model_path=f"augmented_model_iter_{len(iteration_results)}.pth")

            # Evaluate fairness for the new model
            new_gp_raw = fairness_service.calculate_group_performances(new_model_path, df, image_paths, labels, req.fairness_attribute)
            new_worst_raw = fairness_service.find_worst_performing_group(new_gp_raw)
            new_worst_acc = new_worst_raw.get("accuracy", 0.0)
            if hasattr(new_worst_acc, "item"):
                new_worst_acc = new_worst_acc.item()

            # Check for performance degradation and roll back if necessary
            if new_worst_acc < worst_acc:
                logging.warning(f"Performance degraded after augmentation (before: {worst_acc:.4f}, after: {new_worst_acc:.4f}). "
                                f"Discarding model and rolling back data.")
                data_service.remove_last_augmented_batch(num_to_generate)
                # Keep the old model for the next iteration, and log the old results
                iteration_model_path = model_path
                iteration_worst_raw = worst_raw
                iteration_gp_raw = gp_raw
            else:
                # Keep the new model
                model_service.set_current_model_path(new_model_path)
                model_path = new_model_path
                iteration_model_path = new_model_path
                iteration_worst_raw = new_worst_raw
                iteration_gp_raw = new_gp_raw

            iteration_results.append({
                "iteration": len(iteration_results) + 1,
                "worst_group_before_aug": _serialize_worst_group(worst_raw),
                "augmented_images_generated": num_to_generate,
                "new_model_path": iteration_model_path,
                "fairness_evaluation_after_aug": {
                    "all_group_performances": [_serialize_group_performance(attrs, metrics) for attrs, metrics in iteration_gp_raw.items()],
                    "worst_performing_group": _serialize_worst_group(iteration_worst_raw)
                }
            })

        return {
            "status": "Augmentation cycle complete.",
            "init": evaluate_fairness(model_name="model.pth", fairness_attribute=req.fairness_attribute),
            "iterations": iteration_results,
            "augmentation_budget_remaining": budget_remaining
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in augmentation cycle")
        raise HTTPException(status_code=500, detail=str(e))
