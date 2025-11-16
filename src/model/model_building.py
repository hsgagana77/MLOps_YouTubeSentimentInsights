import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

# MLflow
import mlflow
import mlflow.sklearn

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ------------------------------------------------------------
# helper functions
# ------------------------------------------------------------

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise


def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple):
    """Apply TF-IDF with ngrams to the data and save vectorizer."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        X_train_tfidf = vectorizer.fit_transform(X_train)

        logger.debug(f"TF-IDF transformation complete. Train shape: {X_train_tfidf.shape}")

        vect_path = os.path.join(get_root_directory(), "tfidf_vectorizer.pkl")
        with open(vect_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.debug("Vectorizer saved to %s", vect_path)

        return X_train_tfidf, y_train, vect_path

    except Exception as e:
        logger.error("Error during TF-IDF transformation: %s", e)
        raise


def train_lgbm(X_train, y_train, learning_rate, max_depth, n_estimators):
    """Train LightGBM model."""
    try:
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric="multi_logloss",
            is_unbalance=True,
            class_weight="balanced",
            reg_alpha=0.1,
            reg_lambda=0.1,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        model.fit(X_train, y_train)
        logger.debug("LightGBM model trained successfully.")
        return model
    except Exception as e:
        logger.error("Error during LightGBM training: %s", e)
        raise


def save_model(model, file_path: str):
    """Save trained model locally."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug("Model saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving model: %s", e)
        raise


def get_root_directory() -> str:
    """Get project root directory."""
    current = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current, "../../"))


# ------------------------------------------------------------
# MAIN TRAINING FLOW + MLFLOW LOGGING
# ------------------------------------------------------------

def main():
    try:
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, "params.yaml"))

        # hyperparameters from params.yaml
        max_features = params["model_building"]["max_features"]
        ngram_range = tuple(params["model_building"]["ngram_range"])
        learning_rate = params["model_building"]["learning_rate"]
        max_depth = params["model_building"]["max_depth"]
        n_estimators = params["model_building"]["n_estimators"]

        # MLflow setup
        mlflow.set_tracking_uri("http://ec2-44-213-77-221.compute-1.amazonaws.com:5000/")
        mlflow.set_experiment("youtube-sentiment-models")

        with mlflow.start_run():

            # 1. Load preprocessed data
            train_data_path = os.path.join(root_dir, "data/interim/train_processed.csv")
            train_data = load_data(train_data_path)

            # 2. TF-IDF vectorizer
            X_train_tfidf, y_train, vect_path = apply_tfidf(
                train_data, max_features, ngram_range
            )

            # 3. Train the model
            model = train_lgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators)

            # 4. Save model locally
            model_path = os.path.join(root_dir, "lgbm_model.pkl")
            save_model(model, model_path)

            # -------------------------------------------------------
            # 5. METRIC LOGGING (IMPORTANT!)
            # -------------------------------------------------------
            train_preds = model.predict(X_train_tfidf)
            f1 = f1_score(y_train, train_preds, average="weighted")
            mlflow.log_metric("f1_score", f1)
            logger.info(f"Logged f1_score = {f1}")

            # -------------------------------------------------------
            # 6. PARAMS + ARTIFACTS LOGGING
            # -------------------------------------------------------
            mlflow.log_param("max_features", max_features)
            mlflow.log_param("ngram_range", ngram_range)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("n_estimators", n_estimators)

            mlflow.log_artifact(model_path)
            mlflow.log_artifact(vect_path)

            logger.info("Model + Vectorizer logged to MLflow.")

    except Exception as e:
        logger.error("Training failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()