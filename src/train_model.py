import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import load_data, preprocess_and_split, scale_features

def train_and_evaluate():
    print("=" * 60)
    print("Starting Retail Demand Forecasting Training Pipeline")
    print("=" * 60)
    
    # 1. Load data
    data_path = os.path.join("data", "train.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_data.py first.")
    
    raw_df = load_data(data_path)
    print(f"Loaded {len(raw_df)} rows from {data_path}.")
    
    # 2. Preprocess and Split
    split_date = "2025-06-01"
    print(f"Preprocessing data and splitting chronologically at {split_date}...")
    X_train, y_train, X_test, y_test, features = preprocess_and_split(raw_df, split_date)
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    # 3. Scale Features
    print("Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    print("Saved scaler to models/scaler.pkl")
    
    # 4. Define and Train Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Predict on test set
        preds = model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        print(f"{name} Evaluation:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}")
        
        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2
        }
        trained_models[name] = model
    
    # 5. Compare and Save the Best Model
    results_df = pd.DataFrame(results).T
    print("\n" + "=" * 50)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 50)
    print(results_df.to_string())
    print("=" * 50)
    
    # Select best model based on MAE
    best_model_name = results_df['MAE'].idxmin()
    best_model = trained_models[best_model_name]
    print(f"\nBest Model identified by lowest MAE: {best_model_name}")
    
    # Save the best model
    model_save_path = os.path.join("models", "xgboost_model.pkl")
    joblib.dump(best_model, model_save_path)
    print(f"Saved best model ({best_model_name}) to {model_save_path}")
    
    # 6. Feature Importance and Save Plot
    if hasattr(best_model, "feature_importances_"):
        print("\nGenerating feature importance plot...")
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create plot
        os.makedirs("images", exist_ok=True)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=np.array(features)[indices], palette='viridis')
        plt.title(f"Feature Importances - {best_model_name}")
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        
        plot_path = os.path.join("images", "feature_importance.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Feature importance plot saved to {plot_path}")
        
    print("\nTraining pipeline execution completed successfully!")

if __name__ == "__main__":
    train_and_evaluate()
