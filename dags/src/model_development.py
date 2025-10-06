"""
Wine Quality Classification Model Development
Using Random Forest Classifier with evaluation metrics
"""
import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Directory paths
WORKING_DIR = "/opt/airflow/working_data"
MODEL_DIR = "/opt/airflow/model"

# Create directories if they don't exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data() -> str:
    """
    Load wine quality CSV and persist raw dataframe to a pickle file.
    Returns path to saved file.
    """
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "wine_quality.csv",
    )

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Save raw data
    out_path = os.path.join(WORKING_DIR, "raw_wine_data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(df, f)
    
    print(f"Raw data saved to: {out_path}")
    return out_path


def data_preprocessing(file_path: str) -> str:
    """
    Load dataframe, convert quality to binary classification,
    split into train/test, scale features, and save.
    Returns path to saved preprocessed data.
    """
    with open(file_path, "rb") as f:
        df = pickle.load(f)
    
    print("Starting data preprocessing...")
    
    # Convert quality to binary: good (>=6) vs bad (<6)
    df['quality_binary'] = (df['quality'] >= 6).astype(int)
    
    # Features and target
    X = df.drop(['quality', 'quality_binary'], axis=1)
    y = df['quality_binary']
    
    print(f"Feature columns: {X.columns.tolist()}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save preprocessed data and scaler
    out_path = os.path.join(WORKING_DIR, "preprocessed_wine.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    
    with open(out_path, "wb") as f:
        pickle.dump((X_train_scaled, X_test_scaled, y_train.values, y_test.values), f)
    
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"Train set size: {X_train_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    print(f"Preprocessed data saved to: {out_path}")
    
    return out_path


def separate_data_outputs(file_path: str) -> str:
    """
    Passthrough function for DAG composition.
    """
    return file_path


def build_model(file_path: str, filename: str) -> str:
    """
    Train Random Forest Classifier and save to MODEL_DIR.
    Returns model path.
    """
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    
    print("Training Random Forest Classifier...")
    
    # Initialize Random Forest with optimized parameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Save model
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model trained and saved to: {model_path}")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Feature importances: {model.feature_importances_[:5]}")  # First 5
    
    return model_path


def evaluate_model(file_path: str, filename: str) -> dict:
    """
    Load model and evaluate performance on test set.
    Save metrics to JSON file and return metrics dict.
    """
    # Load data
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    
    # Load model
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    print("Evaluating model performance...")
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "precision": float(precision_score(y_test, y_pred_test, average='weighted')),
        "recall": float(recall_score(y_test, y_pred_test, average='weighted')),
        "f1_score": float(f1_score(y_test, y_pred_test, average='weighted')),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
    }
    
    # Save metrics
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nModel Performance Metrics:")
    print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    
    return metrics


def load_model(file_path: str, filename: str) -> int:
    """
    Load saved model and make a sample prediction.
    Returns first prediction as int.
    """
    # Load test data
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    
    # Load model
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    print(f"\nModel loaded successfully from: {model_path}")
    print(f"Test set accuracy: {model.score(X_test, y_test):.4f}")
    print(f"Sample prediction: {predictions[0]} (probability: {probabilities[0][predictions[0]]:.4f})")
    
    return int(predictions[0])