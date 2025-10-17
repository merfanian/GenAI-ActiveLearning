#!/usr/bin/env python3
"""
Test script to demonstrate the return_guide_images functionality.
This allows analyzing guide image selection effects without image generation.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv("/home/mahdi/Projects/GenAI-ActiveLearning/.env")

from services import data_service, model_service, fairness_service, llm_augmentation_service
from utils.config import AUGMENTED_IMAGES_DIR, TRAINED_MODELS_DIR
from utils.attribute_mappings import ADIENCE_GENDER_MAPPING, ADIENCE_GENDER_TARGET

# Configuration
IMAGE_DIR_PATH = "/home/mahdi/Projects/GenAI-ActiveLearning/resources/adience/"
METADATA_CSV_PATH = "/home/mahdi/Projects/GenAI-ActiveLearning/resources/adience/metadata_with_ethnicity_5000.csv"
TARGET_ATTRIBUTE = "label"
FAIRNESS_ATTRIBUTE = "ethnicity"
INITIAL_MODEL_NAME = "model.pth"
ARCHITECTURE = "resnet"
NUM_IMAGES = 5

ATTRIBUTE_MAPPING = ADIENCE_GENDER_MAPPING
TARGET_LABEL_MAPPING = ADIENCE_GENDER_TARGET

def test_guide_image_selection():
    """Test different sampling strategies by returning guide images directly"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load dataset
    logging.info("Loading dataset...")
    data_service.load_and_validate_dataset(IMAGE_DIR_PATH, METADATA_CSV_PATH, TARGET_ATTRIBUTE, FAIRNESS_ATTRIBUTE)
    
    # Load model
    model_path = TRAINED_MODELS_DIR / INITIAL_MODEL_NAME
    if not model_path.exists():
        logging.error(f"Model not found at {model_path}")
        return
    
    model_service.set_current_model_path(str(model_path))
    
    # Get test data for evaluation
    test_df = data_service.get_test_metadata_df()
    image_paths, labels = data_service.get_test_image_paths_and_labels()
    
    # Calculate initial fairness
    gp_raw = fairness_service.calculate_group_performances(str(model_path), test_df, image_paths, labels, FAIRNESS_ATTRIBUTE)
    worst_raw = fairness_service.find_worst_performing_group(gp_raw)
    
    logging.info(f"Initial worst group: {worst_raw['attributes']} with accuracy: {worst_raw['accuracy']:.4f}")
    
    # Test different sampling strategies
    strategies = ["ccds", "random", "confident_misclassifications", "uncertain_classifications"]
    results = {}
    
    for strategy in strategies:
        logging.info(f"\n--- Testing {strategy} sampling strategy ---")
        
        # Create output directory for this strategy
        output_dir = Path(f"guide_images_{strategy}")
        output_dir.mkdir(exist_ok=True)
        
        # Generate guide images (no actual generation, just selection)
        generated = llm_augmentation_service.generate_image_and_get_label(
            worst_raw["attributes"], 
            NUM_IMAGES,
            sampling_strategy=strategy,
            augmented_data_dir=output_dir,
            attribute_mapping=ATTRIBUTE_MAPPING,
            target_label_mapping=TARGET_LABEL_MAPPING,
            validate_quality=False,
            alpha=1.0,
            exploration_mode="exploitation_only",
            return_guide_images=True  # This is the key parameter!
        )
        
        # Analyze the selected guide images
        strategy_results = {
            "strategy": strategy,
            "selected_images": [],
            "confidence_stats": {
                "mean": 0.0,
                "std": 0.0,
                "min": 1.0,
                "max": 0.0
            },
            "prediction_accuracy": 0.0
        }
        
        confidences = []
        correct_predictions = 0
        
        for item in generated:
            confidence = item["model_confidence"]
            prediction = item["model_prediction"]
            true_label = item["true_label"]
            
            confidences.append(confidence)
            if prediction == true_label:
                correct_predictions += 1
            
            strategy_results["selected_images"].append({
                "filename": item["filename"],
                "guide_path": item["guide_image_path"],
                "model_confidence": confidence,
                "model_prediction": prediction,
                "true_label": true_label,
                "correct": prediction == true_label
            })
        
        # Calculate statistics
        if confidences:
            strategy_results["confidence_stats"] = {
                "mean": sum(confidences) / len(confidences),
                "std": (sum([(c - strategy_results["confidence_stats"]["mean"])**2 for c in confidences]) / len(confidences))**0.5,
                "min": min(confidences),
                "max": max(confidences)
            }
        
        strategy_results["prediction_accuracy"] = correct_predictions / len(generated) if generated else 0.0
        
        results[strategy] = strategy_results
        
        logging.info(f"Selected {len(generated)} guide images")
        logging.info(f"Mean confidence: {strategy_results['confidence_stats']['mean']:.4f}")
        logging.info(f"Prediction accuracy: {strategy_results['prediction_accuracy']:.4f}")
        logging.info(f"Confidence range: {strategy_results['confidence_stats']['min']:.4f} - {strategy_results['confidence_stats']['max']:.4f}")
    
    # Save results
    results_path = Path("guide_image_analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"\nResults saved to {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("GUIDE IMAGE SELECTION ANALYSIS SUMMARY")
    print("="*60)
    
    for strategy, data in results.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Mean Confidence: {data['confidence_stats']['mean']:.4f}")
        print(f"  Prediction Accuracy: {data['prediction_accuracy']:.4f}")
        print(f"  Confidence Range: {data['confidence_stats']['min']:.4f} - {data['confidence_stats']['max']:.4f}")
        print(f"  Images Selected: {len(data['selected_images'])}")

if __name__ == "__main__":
    test_guide_image_selection()
