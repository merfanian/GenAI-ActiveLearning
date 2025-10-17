import logging
from services import text_data_service, text_model_service, text_augmentation_service

# Configure logging
logging.basicConfig(level=logging.INFO)

# 1. Load dataset and set model path
text_data_service.load_and_validate_dataset(
    '/home/mahdi/Projects/GenAI-ActiveLearning/resources/text/sample_10000.csv', 
    'label', 
    'label'
)
text_model_service.set_current_model_path('trained_models/text_model.pth')

# 2. Define the worst-performing group
worst_group_attributes = {'label': 0}
num_texts_to_generate = 5

# 3. Generate new text data
logging.info(f"Generating {num_texts_to_generate} new text samples for group: {worst_group_attributes}")
generated_data = text_augmentation_service.generate_text_and_get_label(
    attributes_for_generation=worst_group_attributes,
    num_texts=num_texts_to_generate
)

# 4. Add generated data to the augmented dataset
if generated_data:
    for item in generated_data:
        text_data_service.add_augmented_data(
            text=item["text"],
            attributes=item["attributes_used"],
            llm_acquired_label=str(item["llm_acquired_label"]) # Ensure label is a string
        )
    logging.info(f"Successfully generated and saved {len(generated_data)} new text samples.")
else:
    logging.warning("No text data was generated in this cycle.")

