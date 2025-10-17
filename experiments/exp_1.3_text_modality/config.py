import os

# --- General Experiment Configuration ---
NUM_ITERATIONS = 10
AUGMENTATION_BATCH_SIZE = 20
ACCURACY_THRESHOLD = 0.95
NUM_RUNS = 10
ARCHITECTURE = "distilbert"
RESULTS_DIR = "/home/mahdi/Projects/GenAI-ActiveLearning/experiments/exp_1.3_text_modality/results"

# --- Benchmark-Specific Configurations ---
BENCHMARKS = {
    "ag_news": {
        "name": "text_ag_news",
        "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/text/ag_news/tiny.csv",
        "columns": ["label", "title", "description"],
        "text_column": "description",
        "target_attribute": "label",
        "fairness_attribute": "label",
        "label_mapping": {
            1: "World",
            2: "Sports",
            3: "Business",
            4: "Sci/Tech"
        },
        "architecture": "distilbert-base-uncased",
        "hyperparameters": {
            "max_length": 128,
            "batch_size": 16,
            "epochs": 3,
            "learning_rate": 5e-5
        },
        "prompt_template_examples": (
            "Generate {num_samples} diverse, realistic-sounding news desciptions for the category: '{label_description}'. "
            "The new description should be stylistically similar to these examples, which were hard to classify, try to create variations of these examples + some new from your own creation:\n"
            "{examples_str}\n\n"
            "Return only the new descriptions, one per line."
        ),
        "prompt_template_no_examples": (
            "Generate {num_samples} diverse, realistic-sounding news headlines for the category: '{label_description}'. "
            "Each headline should be unique and typical of what you'd find in a news feed. "
            "Return only the headlines, one per line."
        )
    },
    "bias_in_bios": {
        "name": "text_bias_in_bios",
        "metadata_csv_path": "/home/mahdi/Projects/GenAI-ActiveLearning/resources/text/bias_in_bios/medium.csv",
        "text_column": "hard_text",
        "columns" : ["hard_text", "profession"],
        "target_attribute": "profession",
        "fairness_attribute": "profession",
        "label_mapping": {
            0: "accountant",
            1: "architect",
            2: "attorney",
            3: "chiropractor",
            4: "comedian",
            5: "composer",
            6: "dentist",
            7: "dietitian",
            8: "dj",
            9: "filmmaker",
            10: "interior_designer",
            11: "journalist",
            12: "model",
            13: "nurse",
            14: "painter",
            15: "paralegal",
            16: "pastor",
            17: "personal_trainer",
            18: "photographer",
            19: "physician",
            20: "poet",
            21: "professor",
            22: "psychologist",
            23: "rapper",
            24: "software_engineer",
            25: "surgeon",
            26: "teacher",
            27: "yoga_teacher"
        },
        "architecture": "distilbert-base-uncased",
        "hyperparameters": {
            "max_length": 256,
            "batch_size": 16,
            "epochs": 4,
            "learning_rate": 5e-5
        },
        "prompt_template_examples": (
            "Generate {num_samples} diverse, realistic-sounding biographies for a person with the profession: '{label_description}'. "
            "The new biographies should be stylistically similar to these examples, which were hard to classify, try to create variations of these examples + some new from your own creation:\n"
            "{examples_str}\n\n"
            "Return only the new biographies, one per line."
        ),
        "prompt_template_no_examples": (
            "Generate {num_samples} diverse, realistic-sounding biographies for a person with the profession: '{label_description}'. "
            "Each biography should be unique and reflect the typical language used to describe this profession. "
            "Return only the biographies, one per line."
        )
    },
}


def get_benchmark_config(benchmark_name: str):
    """
    Returns the configuration for a given benchmark.
    """
    if benchmark_name not in BENCHMARKS:
        raise ValueError(f"Benchmark '{benchmark_name}' not found.")
    return BENCHMARKS[benchmark_name]
