#!/usr/bin/env python3
"""
Semiconductor Wafer Pass/Fail Prediction - Training Script
This script trains ML models to classify semiconductor wafers as Pass or Fail
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os

def load_and_preprocess_data(data_path):
    """Load and preprocess the dataset"""
    print("Loading and preprocessing data...")

    # Load dataset
    df = pd.read_csv(data_path)

    # Check for missing values
    print(f"Missing values: {df.isnull().sum().sum()}")

    # Separate features and target
    feature_columns = [col for col in df.columns if col != 'Classification']
    X = df[feature_columns]
    y = df['Classification']

    print(f"Features: {len(feature_columns)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y, feature_columns

def train_models(X_train, y_train, X_test, y_test):
    """Train and evaluate multiple ML models"""

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = {}
    print("\nTraining and evaluating models...")
    print("="*50)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='Pass')
        recall = recall_score(y_test, y_pred, pos_label='Pass')
        f1 = f1_score(y_test, y_pred, pos_label='Pass')

        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        print(f"{name} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")

    return results

def select_best_model(results):
    """Select the best performing model"""
    # Compare models based on F1-score (balanced metric)
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model = results[best_model_name]['model']

    print(f"\nSelected best model: {best_model_name}")
    print(f"Best model F1-score: {results[best_model_name]['f1_score']:.4f}")

    return best_model, best_model_name

def save_model_artifacts(model, scaler, feature_names, model_name):
    """Save trained model and preprocessing artifacts"""
    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Save model and scaler
    joblib.dump(model, 'models/wafer_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')

    print(f"\nModel artifacts saved:")
    print(f"- models/wafer_model.pkl")
    print(f"- models/scaler.pkl")
    print(f"- models/feature_names.pkl")

    # Display feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print(f"\nFeature Importance ({model_name}):")
        print(feature_importance)

def main():
    """Main training pipeline"""
    print("Semiconductor Wafer Pass/Fail Prediction - Model Training")
    print("="*60)

    # Load and preprocess data
    X, y, feature_columns = load_and_preprocess_data('semiconductor_wafer_dataset.csv')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nData split completed:")
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")

    # Train models
    results = train_models(X_train_scaled, y_train, X_test_scaled, y_test)

    # Select best model
    best_model, model_name = select_best_model(results)

    # Save model artifacts
    save_model_artifacts(best_model, scaler, feature_columns, model_name)

    print(f"\nTraining completed successfully!")
    print(f"Run 'streamlit run app.py' to start the web application.")

if __name__ == "__main__":
    main()
