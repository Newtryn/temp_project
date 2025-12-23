#!/usr/bin/env python3
"""
solar_model_train.py
Tek parça, çalışır hâlde model eğitim + değerlendirme script'i.
Kullanım:
    python solar_model_train.py --data data.csv --target target_column
"""

import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------------
# Utilities
# -----------------------------
def evaluate_model(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name}")
    print(f"  R^2  : {r2:.4f}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  MAE  : {mae:.3f}")
    print("-" * 30)
    return r2, rmse, mae

def load_data(path, target_col):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data. Columns: {df.columns.tolist()}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def get_model_pipelines(random_state=42):
    # Each entry: name -> sklearn estimator (we wrap in pipeline below)
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=random_state),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=10000, random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=random_state),
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, max_depth=20, random_state=random_state, n_jobs=-1),
    }

    pipelines = {}
    for name, estimator in models.items():
        # Use scaler for all pipelines to keep input contract consistent (safe)
        pipelines[name] = Pipeline([
            ("scaler", StandardScaler()),
            ("estimator", estimator)
        ])
    return pipelines

# -----------------------------
# Main training routine
# -----------------------------
def main(args):
    print("Loading data...")
    X, y = load_data(args.data, args.target)
    print(f"Data shape: X={X.shape}, y={y.shape}")

    # quick sanity: drop NA rows (you might prefer imputation)
    na_count = X.isna().sum().sum() + y.isna().sum()
    if na_count > 0:
        print(f"Warning: Found {na_count} missing values — dropping rows with NA. Consider imputation.")
        df = pd.concat([X, y], axis=1).dropna()
        X = df.drop(columns=[args.target])
        y = df[args.target]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    print(f"Train/Test split: {X_train.shape[0]} train, {X_test.shape[0]} test")

    pipelines = get_model_pipelines(random_state=args.random_state)

    results = {}
    # Train & evaluate
    for name, pipe in pipelines.items():
        print(f"Training {name} ...")
        pipe.fit(X_train, y_train)

        # Predictions on test set
        y_pred = pipe.predict(X_test)
        r2, rmse, mae = evaluate_model(y_test, y_pred, name)

        # quick CV on training set (5-fold) to check generalization (use negative MSE internally -> convert)
        try:
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
            cv_mean = np.mean(cv_scores)
        except Exception as e:
            cv_mean = float("nan")
            print(f"  CV failed: {e}")

        print(f"  CV R^2 (5-fold, train): {cv_mean:.4f}")
        print()

        results[name] = {
            "pipeline": pipe,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "cv_r2": cv_mean
        }

    # Select best by test R^2
    best_model_name = max(results.items(), key=lambda x: x[1]["r2"])[0]
    best_entry = results[best_model_name]
    best_pipeline = best_entry["pipeline"]
    best_r2 = best_entry["r2"]

    print(f" Best model: {best_model_name} (R^2 = {best_r2:.4f})")
    if best_r2 >= args.target_r2:
        print(f" Target achieved: R^2 >= {args.target_r2}")
    else:
        print(f" Target not achieved (target R^2 = {args.target_r2})")

    # Save best pipeline (scaler + model)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(args.output_dir, f"solar_model_{best_model_name.replace(' ', '_')}_{timestamp}.pkl")
    joblib.dump(best_pipeline, filename)
    print(f"Model saved to: {filename}")

    # Sanity sample predictions (first 10)
    print("\nSample predictions (first 10 test rows):")
    best_preds = best_pipeline.predict(X_test)
    # align indices for clarity
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = pd.Series(y_test).reset_index(drop=True)
    for i in range(min(10, len(y_test_reset))):
        print(f"Idx {i:03d} | True: {y_test_reset.iloc[i]:.3f} | Pred: {best_preds[i]:.3f}")

    # If best model is tree-based, show top features (if available)
    estimator = best_pipeline.named_steps["estimator"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        feat_names = X.columns
        fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
        print("\nTop 10 feature importances:")
        print(fi.head(10).to_string())
    elif hasattr(estimator, "coef_"):
        coefs = estimator.coef_
        feat_names = X.columns
        co = pd.Series(coefs, index=feat_names).abs().sort_values(ascending=False)
        print("\nTop 10 absolute coefficients:")
        print(co.head(10).to_string())

    print("\nDone.")

# -----------------------------
# Command-line interface
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate regression models for solar power (single-file).")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", default="target", help="Target column name in CSV")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="models", help="Directory to save best model")
    parser.add_argument("--target-r2", type=float, default=0.80, help="Target R^2 threshold for reporting")
    args = parser.parse_args()
    main(args)
