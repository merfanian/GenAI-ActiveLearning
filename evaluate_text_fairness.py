import json
from services import text_data_service, text_model_service, fairness_service

# 1. Load dataset
text_data_service.load_and_validate_dataset(
    '/home/mahdi/Projects/GenAI-ActiveLearning/resources/text/sample_10000.csv', 
    'label', 
    'label'
)

# 2. Define model path and get test data
model_path = 'trained_models/text_model.pth'
df = text_data_service.get_test_metadata_df()
texts, labels = text_data_service.get_test_texts_and_labels() 

# 3. Get predictions
preds = text_model_service.predict(model_path, texts)
df['predicted_label'] = [p['predicted_label'] for p in preds]

# 4. Calculate group performances
group_perfs = {}
fairness_attribute = 'label'
groups = df.groupby(fairness_attribute)

for group_vals, group_df in groups:
    if not isinstance(group_vals, tuple):
        group_vals = (group_vals,)
    
    key = tuple(zip([fairness_attribute], group_vals))
    correct = (group_df['predicted_label'].astype(str) == group_df['label'].astype(str)).sum()
    count = len(group_df)
    acc = correct / count if count else 0.0
    group_perfs[key] = {"accuracy": acc, "count": count}

# 5. Find worst group and print results
worst_group = fairness_service.find_worst_performing_group(group_perfs)
output = {
    'all_group_performances': [metrics for metrics in group_perfs.values()],
    'worst_performing_group': worst_group
}
print(json.dumps(output, indent=4))
