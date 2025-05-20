import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import data_service, model_service, fairness_service, llm_augmentation_service

router = APIRouter()

class SetupDatasetRequest(BaseModel):
    image_dir_path: str
    metadata_csv_path: str

@router.post("/setup_dataset")
def setup_dataset(req: SetupDatasetRequest):
    result = data_service.load_and_validate_dataset(req.image_dir_path, req.metadata_csv_path)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/train_initial_model")
def train_initial_model():
    try:
        image_paths, labels = data_service.get_image_paths_and_labels()
        model_path = model_service.train_model(image_paths, labels)
        model_service.set_current_model_path(model_path)
        return {"message": "Initial model trained.", "model_path": model_path}
    except Exception as e:
        logging.exception("Error training initial model")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluate_fairness")
def evaluate_fairness():
    try:
        model_path = model_service.get_current_model_path()
        if not model_path:
            raise HTTPException(status_code=400, detail="No model trained.")
        df = data_service.get_metadata_df()
        image_paths, _ = data_service.get_image_paths_and_labels()
        gp = fairness_service.calculate_group_performances(model_path, df, image_paths)
        worst = fairness_service.find_worst_performing_group(gp)
        return {"all_group_performances": gp, "worst_performing_group": worst}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error evaluating fairness")
        raise HTTPException(status_code=500, detail=str(e))

class AugmentationCycleRequest(BaseModel):
    augmentation_budget_remaining: int
    accuracy_threshold: float
    augmentation_batch_size: int

@router.post("/run_iterative_augmentation_cycle")
def run_iterative_augmentation_cycle(req: AugmentationCycleRequest):
    try:
        model_path = model_service.get_current_model_path()
        if not model_path:
            raise HTTPException(status_code=400, detail="No model trained.")
        df = data_service.get_metadata_df()
        image_paths, _ = data_service.get_image_paths_and_labels()
        gp = fairness_service.calculate_group_performances(model_path, df, image_paths)
        worst = fairness_service.find_worst_performing_group(gp)
        worst_acc = worst.get("accuracy", 0.0)
        if worst_acc >= req.accuracy_threshold or req.augmentation_budget_remaining <= 0:
            return {"status": "Stopping: Condition met."}
        num_to_generate = min(req.augmentation_batch_size, req.augmentation_budget_remaining)
        generated = llm_augmentation_service.generate_image_and_get_label(worst["attributes"], num_to_generate)
        for item in generated:
            data_service.add_augmented_data(item["image_path"], item["attributes_used"], item["llm_acquired_label"])
        remaining = req.augmentation_budget_remaining - num_to_generate
        all_image_paths, all_labels = data_service.get_image_paths_and_labels()
        new_model_path = model_service.train_model(all_image_paths, all_labels, existing_model_path=model_path)
        model_service.set_current_model_path(new_model_path)
        return {
            "status": "Iteration complete",
            "worst_group_before_aug": worst,
            "augmented_images_generated": num_to_generate,
            "new_model_path": new_model_path,
            "augmentation_budget_remaining": remaining
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in augmentation cycle")
        raise HTTPException(status_code=500, detail=str(e))