import pandas as pd
import os
import logging
import yaml

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Logging setup
log_dir = os.path.join(BASE_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("YAML file not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML: %s", e)
        raise

def load_data(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        logger.debug("Data loaded from %s", csv_path)
        return df
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug("Dataframe columns cleaned and renamed")
        return df
    except Exception as e:
        logger.error("Preprocessing failed: %s", e)
        raise

def save_raw_data(df: pd.DataFrame, output_dir: str) -> None:
    try:
        raw_path = os.path.join(output_dir, "raw")
        os.makedirs(raw_path, exist_ok=True)
        df.to_csv(os.path.join(raw_path, "spam_raw.csv"), index=False)
        logger.debug("Raw data saved to %s", raw_path)
    except Exception as e:
        logger.error("Saving raw data failed: %s", e)
        raise

def main():
    try:
        input_path = os.path.join(BASE_DIR, "experiments", "spam.csv")
        output_dir = os.path.join(BASE_DIR, "data")
        
        df = load_data(input_path)
        df_cleaned = preprocess_data(df)
        save_raw_data(df_cleaned, output_dir)

        logger.info("Data ingestion completed successfully.")
    except Exception as e:
        logger.error("Data ingestion failed: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
