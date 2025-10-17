"""
Configuration for AG News dataset with GroupDRO
"""
import pandas as pd
from pathlib import Path

def create_ag_news_config():
    """Create properly formatted AG News dataset for GroupDRO experiment"""
    
    # Load the dataset
    df = pd.read_csv("/home/mahdi/Projects/GenAI-ActiveLearning/resources/text/ag_news/medium.csv", header=None)
    df.columns = ['label', 'title', 'description']
    
    # Create label mapping
    label_mapping = {
        1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"
    }
    
    # Map labels
    df['label_name'] = df['label'].map(label_mapping)
    
    # For AG News, we'll use the label as both target and fairness attribute
    # This creates a scenario where we want to improve performance on underrepresented categories
    df['fairness_group'] = df['label_name']  # Same as target for this experiment
    
    # Save the processed dataset
    output_path = Path("/home/mahdi/Projects/GenAI-ActiveLearning/experiments/groupdro_text_baseline/ag_news_processed.csv")
    df[['description', 'label_name', 'fairness_group']].to_csv(output_path, index=False)
    
    print(f"Processed AG News dataset saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:")
    print(df['label_name'].value_counts())
    
    return output_path

if __name__ == "__main__":
    create_ag_news_config()
