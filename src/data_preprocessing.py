import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

# NLTK setup
nltk.download('stopwords')
nltk.download('punkt')

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Logging configuration
log_dir = os.path.join(BASE_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Text normalization function
def transform_text(text: str) -> str:
    try:
        ps = PorterStemmer()
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        tokens = [ps.stem(word) for word in tokens]
        return " ".join(tokens)
    except Exception as e:
        logger.error(f"Error transforming text: {e}")
        return ""

# DataFrame preprocessing
def preprocess_df(df: pd.DataFrame, text_column='text', target_column='target') -> pd.DataFrame:
    try:
        logger.debug("Starting preprocessing")

        # Encode target labels
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("Target labels encoded")

        # Drop duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        logger.debug(f"Removed {initial_count - len(df)} duplicate rows")

        # Apply text transformation
        df[text_column] = df[text_column].apply(transform_text)
        logger.debug("Text data transformed")

        return df
    except Exception as e:
        logger.error(f"Failed preprocessing: {e}")
        raise

def main():
    try:
        input_path = os.path.join(BASE_DIR, 'data', 'raw', 'spam_raw.csv')
        output_dir = os.path.join(BASE_DIR, 'data', 'processed')
        output_path = os.path.join(output_dir, 'spam_clean.csv')

        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(input_path)
        logger.debug(f"Loaded raw data from {input_path}")

        df_clean = preprocess_df(df)

        df_clean.to_csv(output_path, index=False)
        logger.debug(f"Saved cleaned data to {output_path}")

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
    except pd.errors.EmptyDataError as e:
        logger.error(f"CSV file is empty: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
