import pandas as pd
from datasets import load_dataset
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_bias_in_bios_metadata():
    """
    Downloads the Bias in Bios dataset, processes it, and saves it as a CSV.
    """
    try:
        logging.info("Downloading Bias in Bios dataset from Hugging Face...")
        # The 'hard' configuration is the one used for the paper
        raw_dataset = load_dataset('mdzis/bias_in_bios', 'hard')
        logging.info("Dataset downloaded successfully.")

        # Combine all splits into a single DataFrame
        df = pd.concat([raw_dataset[split].to_pandas() for split in raw_dataset.keys()])

        # Rename columns for consistency with the experiment script
        # 'bio' -> 'text'
        # 'p' -> 'label' (profession)
        # 'g' -> 'gender' (fairness attribute)
        df.rename(columns={'bio': 'text', 'p': 'label', 'g': 'gender'}, inplace=True)
        
        # Select only the relevant columns
        df = df[['text', 'label', 'gender']]

        # Create the output directory
        output_dir = Path('/home/mahdi/Projects/GenAI-ActiveLearning/resources/text/bias_in_bios')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        output_path = output_dir / 'metadata.csv'
        df.to_csv(output_path, index=False)
        
        logging.info(f"Successfully processed and saved the dataset to {output_path}")
        logging.info(f"Dataset contains {len(df)} records.")
        logging.info(f"Columns: {df.columns.tolist()}")
        logging.info(f"Sample record:\n{df.head(1).to_string()}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    create_bias_in_bios_metadata()
