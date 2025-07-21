import os
import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Logger setup
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_training")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, "model_training.log"))

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Feature data loaded from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load feature data from {file_path}: {e}")
        raise


def load_vectorizer(vectorizer_path: str):
    try:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        logger.debug(f"Vectorizer loaded from {vectorizer_path}")
        return vectorizer
    except Exception as e:
        logger.error(f"Failed to load vectorizer: {e}")
        raise


def train_and_select_model(X_train, y_train, X_val, y_val):
    try:
        models = {
            "MultinomialNB": MultinomialNB(),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(),
            "SVC": SVC()
        }

        best_model = None
        best_score = 0
        best_name = ""

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = f1_score(y_val, preds)
            logger.debug(f"{name} F1 Score: {score:.4f}")
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name

        logger.info(f"Best model: {best_name} with F1 score: {best_score:.4f}")
        return best_model, best_name
    except Exception as e:
        logger.error(f"Model training error: {e}")
        raise


def save_model(model, save_path: str):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved at {save_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise


def main():
    try:
        train_path = os.path.join(BASE_DIR, "data", "interim", "train_processed.csv")
        df = load_data(train_path)

        # Fix: Fill NaN text values with empty strings to avoid vectorizer error
        X = df["text"].fillna("")
        y = df["target"]

        vectorizer_path = os.path.join(BASE_DIR, "artifacts", "transformers", "tfidf_vectorizer.pkl")
        vectorizer = load_vectorizer(vectorizer_path)

        X_vectorized = vectorizer.transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

        model, model_name = train_and_select_model(X_train, y_train, X_val, y_val)

        model_path = os.path.join(BASE_DIR, "artifacts", "models", "model.pkl")
        save_model(model, model_path)

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
