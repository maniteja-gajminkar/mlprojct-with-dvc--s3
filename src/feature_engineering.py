import os
import logging
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Logging setup
log_dir = os.path.join(BASE_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Failed to load params: {e}")
        raise

def load_clean_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df.fillna("", inplace=True)
        logger.debug(f"Clean data loaded from {path}")
        return df
    except Exception as e:
        logger.error(f"Error loading clean data: {e}")
        raise

def save_csv(df: pd.DataFrame, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.debug(f"Saved CSV to {path}")
    except Exception as e:
        logger.error(f"Error saving CSV to {path}: {e}")
        raise

def save_pickle(obj, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        logger.debug(f"Saved pickle to {path}")
    except Exception as e:
        logger.error(f"Error saving pickle: {e}")
        raise

def main():
    try:
        # Load params
        params_path = os.path.join(BASE_DIR, "params.yaml")
        params = load_params(params_path)
        max_features = params['feature_engineering']['max_features']

        # Load cleaned data
        clean_data_path = os.path.join(BASE_DIR, "data", "processed", "spam_clean.csv")
        df = load_clean_data(clean_data_path)

        # Split
        X = df["text"]
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Save raw splits
        interim_dir = os.path.join(BASE_DIR, "data", "interim")
        save_csv(pd.DataFrame({'text': X_train, 'target': y_train}), os.path.join(interim_dir, "train_processed.csv"))
        save_csv(pd.DataFrame({'text': X_test, 'target': y_test}), os.path.join(interim_dir, "test_processed.csv"))

        # TF-IDF
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Save transformed data
        processed_dir = os.path.join(BASE_DIR, "data", "processed")
        save_csv(pd.DataFrame(X_train_tfidf.toarray()), os.path.join(processed_dir, "train_tfidf.csv"))
        save_csv(pd.DataFrame(X_test_tfidf.toarray()), os.path.join(processed_dir, "test_tfidf.csv"))

        # Save vectorizer
        transformer_path = os.path.join(BASE_DIR, "artifacts", "transformers", "tfidf_vectorizer.pkl")
        save_pickle(vectorizer, transformer_path)

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
