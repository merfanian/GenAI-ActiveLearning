"""
Configuration for Bias in Bios dataset with GroupDRO
"""
import pandas as pd
from pathlib import Path

def create_bias_in_bios_config():
    """Create properly formatted Bias in Bios dataset for GroupDRO experiment"""
    
    # Load the dataset
    df = pd.read_csv("/home/mahdi/Projects/GenAI-ActiveLearning/resources/text/bias_in_bios/medium.csv", header=None)
    df.columns = ['text', 'profession']
    
    # For this experiment, we'll create synthetic gender labels based on profession bias
    # This is a simplified approach for demonstration purposes
    # In practice, you would use the actual gender labels from the full dataset
    import numpy as np
    np.random.seed(42)  # For reproducibility
    df['gender'] = np.random.choice([0, 1], size=len(df), p=[0.5, 0.5])  # 50-50 split
    
    # Create label mapping
    profession_mapping = {
        0: "accountant", 1: "architect", 2: "attorney", 3: "chiropractor", 4: "comedian",
        5: "composer", 6: "dentist", 7: "dietitian", 8: "dj", 9: "filmmaker",
        10: "interior_designer", 11: "journalist", 12: "model", 13: "nurse", 14: "painter",
        15: "paralegal", 16: "pastor", 17: "personal_trainer", 18: "photographer", 19: "physician",
        20: "poet", 21: "professor", 22: "psychologist", 23: "rapper", 24: "software_engineer",
        25: "surgeon", 26: "teacher", 27: "yoga_teacher"
    }
    
    # Map profession labels
    df['profession_label'] = df['profession'].map(profession_mapping)
    
    # Create gender mapping (assuming 0=female, 1=male based on typical bias in bios format)
    gender_mapping = {0: "female", 1: "male"}
    df['gender_label'] = df['gender'].map(gender_mapping)
    
    # Save the processed dataset
    output_path = Path("/home/mahdi/Projects/GenAI-ActiveLearning/experiments/groupdro_text_baseline/bias_in_bios_processed.csv")
    df[['text', 'profession_label', 'gender_label']].to_csv(output_path, index=False)
    
    print(f"Processed Bias in Bios dataset saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Profession distribution:")
    print(df['profession_label'].value_counts())
    print(f"Gender distribution:")
    print(df['gender_label'].value_counts())
    
    return output_path

if __name__ == "__main__":
    create_bias_in_bios_config()
