import os
import logging
import pickle
import json
import pandas as pd
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Logger setup
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.debug(f"Model loaded from {model_path}")
    return model


def load_vectorizer(vectorizer_path):
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    logger.debug(f"Vectorizer loaded from {vectorizer_path}")
    return vectorizer


def load_test_data(file_path):
    df = pd.read_csv(file_path)
    logger.debug(f"Test data loaded from {file_path}")
    return df


def evaluate_model(model, vectorizer, df):
    X_test = df["text"].fillna("")
    y_test = df["target"]

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return accuracy, precision, recall, f1  # Return metrics for writing


def main():
    try:
        model_path = os.path.join(BASE_DIR, "artifacts", "models", "model.pkl")
        vectorizer_path = os.path.join(BASE_DIR, "artifacts", "transformers", "tfidf_vectorizer.pkl")
        test_data_path = os.path.join(BASE_DIR, "data", "interim", "test_processed.csv")

        model = load_model(model_path)
        vectorizer = load_vectorizer(vectorizer_path)
        df_test = load_test_data(test_data_path)

        accuracy, precision, recall, f1 = evaluate_model(model, vectorizer, df_test)

        # ✅ Write metrics.json for DVC
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info("metrics.json file created.")

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
