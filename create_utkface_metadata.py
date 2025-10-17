import os
import pandas as pd
import numpy as np

# Age mapping based on Adience dataset
AGE_MAPPING = {
    (0, 2): 0,
    (4, 6): 1,
    (8, 13): 2,
    (15, 20): 3,
    (25, 32): 4,
    (38, 43): 5,
    (48, 53): 6,
    (60, 100): 7,
}

def get_age_group(age):
    for age_range, group in AGE_MAPPING.items():
        if age_range[0] <= age <= age_range[1]:
            return group
    return np.nan

def create_utkface_metadata(dataset_path, output_path):
    data = []
    for filename in os.listdir(os.path.join(dataset_path, 'UTKFace')):
        parts = filename.split('_')
        if len(parts) == 4:
            age, gender, race, _ = parts
            try:
                age, gender, race = int(age), int(gender), int(race)
                age_group = get_age_group(age)
                if not pd.isna(age_group):
                    data.append([filename, age, gender, race, int(age_group)])
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(data, columns=['filename', 'age', 'gender', 'race', 'age_group'])
    
    # Save the full metadata
    full_metadata_path = os.path.join(output_path, 'metadata.csv')
    df.to_csv(full_metadata_path, index=False)
    print(f"Full metadata saved to {full_metadata_path}")

    # Create and save a 5000-row sample
    sample_5000 = df.sample(n=5000, random_state=42)
    sample_5000_path = os.path.join(output_path, 'metadata_5000.csv')
    sample_5000.to_csv(sample_5000_path, index=False)
    print(f"5000-row sample saved to {sample_5000_path}")

    # Create and save a 1000-row sample
    sample_1000 = df.sample(n=1000, random_state=42)
    sample_1000_path = os.path.join(output_path, 'metadata_1000.csv')
    sample_1000.to_csv(sample_1000_path, index=False)
    print(f"1000-row sample saved to {sample_1000_path}")

if __name__ == '__main__':
    dataset_path = '/home/mahdi/Projects/GenAI-ActiveLearning/resources/utkface'
    output_path = '/home/mahdi/Projects/GenAI-ActiveLearning/resources/utkface'
    create_utkface_metadata(dataset_path, output_path)
