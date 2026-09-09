"""
MLOps Assignment 1 - House Price Prediction training script.
Author Student ID: M-Faizan-Khan
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Path configuration (relative to the project root)
DATA_PATH = os.path.join("data", "dataset.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model_M-Faizan-Khan.pkl")

# Hyperparameters (Part 3: add / tweak these during iterative experimentation)
N_ESTIMATORS = 100
RANDOM_STATE = 42
# LEARNING_RATE = 0.1   # <-- Part 3: uncomment/add this hyperparameter line, then diff & reset


def load_data(path):
    """Task 1: load the dataset from the data/ directory."""
    print(f"[Student ID: M-Faizan-Khan] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns.")
    return df


def main():
    df = load_data(DATA_PATH)

    X = df.drop(columns=["price"])
    y = df["price"]

    # ============================================================
  
     X_processed = MinMaxScaler().fit_transform(X) 
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Task 2: train a machine learning model
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, preds):,.0f}")
    print(f"R^2: {r2_score(y_test, preds):.3f}")

    # Task 3: serialize and save the trained model into the model/ directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
