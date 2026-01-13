"""
Tasks 1 & 2: Mutation Status and Oncogenic Ensemble Model
Combines predictions from two models using ensemble methods.

Task 1: Mutation Status (TUMOR-SOMATIC vs CHIP)
Task 2: Oncogenic Classification (Oncogenic vs Benign)

Uses Logistic Regression, Random Forest, and XGBoost to create meta-models.
Configured via YAML file. See configs/task1_mutation_ensemble_config.yaml or task2_oncogenic_ensemble_config.yaml
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
import argparse
import yaml
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt


# ============================================================================
# Configuration and Logging
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict):
    """Setup logging based on config."""
    log_config = config.get('logging', {})

    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    )


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_and_prepare_data(config: dict) -> pd.DataFrame:
    """Load and prepare mutation prediction data."""
    logging.info("=" * 80)
    logging.info("Loading and preparing data")
    logging.info("=" * 80)

    # Load merged predictions file
    data_path = config['input']['data_path']
    logging.info(f"Loading data from {data_path}")

    df = pd.read_csv(data_path)

    # Print basic statistics
    logging.info(f"Total samples: {len(df)}")

    if 'PATIENT_ID' in df.columns:
        logging.info(f"Unique patients: {df['PATIENT_ID'].nunique()}")

    if 'ground_truth' in df.columns:
        logging.info("Ground truth distribution:")
        for label, count in df['ground_truth'].value_counts().items():
            logging.info(f"  {label}: {count}")

    if 'CANCER_TYPE' in df.columns:
        logging.info(f"Cancer types: {df['CANCER_TYPE'].nunique()}")

    # Get model column names from config
    model1_pred = config['model1']['prediction_column']
    model1_prob = config['model1']['probability_column']
    model2_pred = config['model2']['prediction_column']
    model2_prob = config['model2']['probability_column']

    # Select relevant columns
    required_cols = config.get('required_columns', ['SAMPLE_ID', 'PATIENT_ID', 'ground_truth'])
    selected_columns = required_cols + [model1_pred, model1_prob, model2_pred, model2_prob]

    # Only keep columns that exist in the dataframe
    selected_columns = [col for col in selected_columns if col in df.columns]

    df_selected = df[selected_columns].copy()

    # Drop rows with missing values
    df_selected = df_selected.dropna().reset_index(drop=True)

    logging.info(f"Dataset after cleaning: {len(df_selected)} samples")

    if 'CANCER_TYPE' in df_selected.columns:
        logging.info(f"Unique cancer types: {df_selected['CANCER_TYPE'].nunique()}")

    return df_selected


def prepare_features_and_labels(df: pd.DataFrame, config: dict):
    """Prepare features and labels for machine learning models."""
    model1_pred = config['model1']['prediction_column']
    model1_prob = config['model1']['probability_column']
    model2_pred = config['model2']['prediction_column']
    model2_prob = config['model2']['probability_column']

    # Prepare features - using predictions and probabilities
    features = df[[model1_pred, model1_prob, model2_pred, model2_prob]].copy()
    labels = df['ground_truth']

    # One-hot encode only the categorical prediction columns (not probabilities)
    features_encoded = pd.get_dummies(features, columns=[model1_pred, model2_pred])

    # Ensure probability columns are numeric
    features_encoded[model1_prob] = pd.to_numeric(features_encoded[model1_prob], errors='coerce').fillna(0)
    features_encoded[model2_prob] = pd.to_numeric(features_encoded[model2_prob], errors='coerce').fillna(0)

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Encode original predictions for baseline comparison
    model1_preds_encoded = label_encoder.transform(df[model1_pred])
    model2_preds_encoded = label_encoder.transform(df[model2_pred])

    return features_encoded, labels_encoded, model1_preds_encoded, model2_preds_encoded, label_encoder


# ============================================================================
# Cross-Validation
# ============================================================================

def run_k_fold_cv(df: pd.DataFrame, features: pd.DataFrame, labels: np.ndarray,
                  original_model1: np.ndarray, original_model2: np.ndarray,
                  config: dict, output_dir: str, k: int = 5):
    """Run k-fold cross-validation split by PATIENT_ID."""
    logging.info("=" * 80)
    logging.info(f"Running {k}-Fold Cross-Validation (Split by PATIENT_ID)")
    logging.info("=" * 80)

    model1_name = config['model1']['name']
    model2_name = config['model2']['name']

    # Get unique patient IDs
    unique_patients = df['PATIENT_ID'].unique()
    np.random.seed(config.get('random_seed', 42))
    np.random.shuffle(unique_patients)

    # Split patients into folds
    fold_size = len(unique_patients) // k
    patient_folds = [unique_patients[i*fold_size:(i+1)*fold_size] for i in range(k-1)]
    patient_folds.append(unique_patients[(k-1)*fold_size:])

    results = {
        'fold': [],
        f'{model1_name}_accuracy': [], f'{model1_name}_f1': [], f'{model1_name}_precision': [], f'{model1_name}_recall': [],
        f'{model2_name}_accuracy': [], f'{model2_name}_f1': [], f'{model2_name}_precision': [], f'{model2_name}_recall': [],
        'LogReg_accuracy': [], 'LogReg_f1': [], 'LogReg_precision': [], 'LogReg_recall': [],
        'XGBoost_accuracy': [], 'XGBoost_f1': [], 'XGBoost_precision': [], 'XGBoost_recall': [],
        'RandomForest_accuracy': [], 'RandomForest_f1': [], 'RandomForest_precision': [], 'RandomForest_recall': []
    }

    for i, test_patients in enumerate(patient_folds):
        # Create boolean masks based on patient IDs
        test_mask = df['PATIENT_ID'].isin(test_patients)
        train_mask = ~test_mask

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        logging.info(f"Fold {i+1}: {len(test_patients)} patients, {len(test_idx)} samples in test set")

        # Original model metrics
        model1_preds = original_model1[test_idx]
        model2_preds = original_model2[test_idx]

        # Calculate metrics for model1
        acc_model1 = accuracy_score(y_test, model1_preds)
        f1_model1 = f1_score(y_test, model1_preds, average='weighted', zero_division=0)
        prec_model1 = precision_score(y_test, model1_preds, average='weighted', zero_division=0)
        rec_model1 = recall_score(y_test, model1_preds, average='weighted', zero_division=0)

        # Calculate metrics for model2
        acc_model2 = accuracy_score(y_test, model2_preds)
        f1_model2 = f1_score(y_test, model2_preds, average='weighted', zero_division=0)
        prec_model2 = precision_score(y_test, model2_preds, average='weighted', zero_division=0)
        rec_model2 = recall_score(y_test, model2_preds, average='weighted', zero_division=0)

        # Train ensemble models
        log_model = LogisticRegression(max_iter=1000, random_state=42)
        log_model.fit(X_train, y_train)
        log_preds = log_model.predict(X_test)
        acc_log = accuracy_score(y_test, log_preds)
        f1_log = f1_score(y_test, log_preds, average='weighted', zero_division=0)
        prec_log = precision_score(y_test, log_preds, average='weighted', zero_division=0)
        rec_log = recall_score(y_test, log_preds, average='weighted', zero_division=0)

        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        acc_xgb = accuracy_score(y_test, xgb_preds)
        f1_xgb = f1_score(y_test, xgb_preds, average='weighted', zero_division=0)
        prec_xgb = precision_score(y_test, xgb_preds, average='weighted', zero_division=0)
        rec_xgb = recall_score(y_test, xgb_preds, average='weighted', zero_division=0)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        acc_rf = accuracy_score(y_test, rf_preds)
        f1_rf = f1_score(y_test, rf_preds, average='weighted', zero_division=0)
        prec_rf = precision_score(y_test, rf_preds, average='weighted', zero_division=0)
        rec_rf = recall_score(y_test, rf_preds, average='weighted', zero_division=0)

        # Store results
        results['fold'].append(i + 1)
        results[f'{model1_name}_accuracy'].append(acc_model1)
        results[f'{model1_name}_f1'].append(f1_model1)
        results[f'{model1_name}_precision'].append(prec_model1)
        results[f'{model1_name}_recall'].append(rec_model1)

        results[f'{model2_name}_accuracy'].append(acc_model2)
        results[f'{model2_name}_f1'].append(f1_model2)
        results[f'{model2_name}_precision'].append(prec_model2)
        results[f'{model2_name}_recall'].append(rec_model2)

        results['LogReg_accuracy'].append(acc_log)
        results['LogReg_f1'].append(f1_log)
        results['LogReg_precision'].append(prec_log)
        results['LogReg_recall'].append(rec_log)

        results['XGBoost_accuracy'].append(acc_xgb)
        results['XGBoost_f1'].append(f1_xgb)
        results['XGBoost_precision'].append(prec_xgb)
        results['XGBoost_recall'].append(rec_xgb)

        results['RandomForest_accuracy'].append(acc_rf)
        results['RandomForest_f1'].append(f1_rf)
        results['RandomForest_precision'].append(prec_rf)
        results['RandomForest_recall'].append(rec_rf)

        logging.info(f"  {model1_name}: {acc_model1:.4f}, {model2_name}: {acc_model2:.4f}, "
                    f"LogReg: {acc_log:.4f}, XGBoost: {acc_xgb:.4f}, RF: {acc_rf:.4f}")

    # Save detailed results
    results_df = pd.DataFrame(results)
    cv_detailed_file = os.path.join(output_dir, f'{k}_fold_cv_detailed_results.csv')
    results_df.to_csv(cv_detailed_file, index=False)
    logging.info(f"{k}-fold CV detailed results saved to: {cv_detailed_file}")

    # Calculate summary statistics
    methods = [model1_name, model2_name, 'LogReg', 'XGBoost', 'RandomForest']
    method_names = [f'{model1_name}', f'{model2_name}', 'Logistic Regression', 'XGBoost', 'Random Forest']

    summary_data = {
        'Method': method_names,
        'Mean_Accuracy': [results_df[f'{method}_accuracy'].mean() for method in methods],
        'Std_Accuracy': [results_df[f'{method}_accuracy'].std() for method in methods],
        'Mean_F1': [results_df[f'{method}_f1'].mean() for method in methods],
        'Std_F1': [results_df[f'{method}_f1'].std() for method in methods],
        'Mean_Precision': [results_df[f'{method}_precision'].mean() for method in methods],
        'Std_Precision': [results_df[f'{method}_precision'].std() for method in methods],
        'Mean_Recall': [results_df[f'{method}_recall'].mean() for method in methods],
        'Std_Recall': [results_df[f'{method}_recall'].std() for method in methods]
    }

    summary_df = pd.DataFrame(summary_data)
    cv_summary_file = os.path.join(output_dir, f'{k}_fold_cv_summary.csv')
    summary_df.to_csv(cv_summary_file, index=False)
    logging.info(f"{k}-fold CV summary saved to: {cv_summary_file}")

    # Create and save plot
    plt.figure(figsize=(10, 6))
    plt.bar(method_names, summary_df['Mean_Accuracy'], yerr=summary_df['Std_Accuracy'], capsize=5)
    plt.ylabel("Accuracy")
    plt.title(f"{k}-Fold Cross-Validation Accuracy (Mean ± Std Dev)")
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{k}_fold_cv_accuracy_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary
    logging.info("\nCross-Validation Metrics Summary:")
    for _, row in summary_df.iterrows():
        logging.info(f"{row['Method']}:")
        logging.info(f"  Accuracy: {row['Mean_Accuracy']:.4f} ± {row['Std_Accuracy']:.4f}")
        logging.info(f"  F1 Score: {row['Mean_F1']:.4f} ± {row['Std_F1']:.4f}")

    return results_df, summary_df


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Ensemble model for mutation status or oncogenic classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Task 1: Mutation Status
  python task12_mutation_ensemble.py --config ../configs/task1_mutation_ensemble_config.yaml

  # Task 2: Oncogenic Classification
  python task12_mutation_ensemble.py --config ../configs/task2_oncogenic_ensemble_config.yaml
        """
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--fold', type=int, default=None,
                       help='Number of folds for cross-validation (overrides config)')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    setup_logging(config)

    task_name = config.get('task_name', 'Mutation Ensemble Model')

    logging.info("=" * 80)
    logging.info(task_name)
    logging.info("=" * 80)
    logging.info(f"Configuration loaded from {args.config}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = config.get('output_dir', "/data1/morrisq/yuj13/llm_genomics/mutation_meta_model_result")
    output_dir = f"{output_base}/{config['model1']['name']}_{config['model2']['name']}_{timestamp}"
    output_dir = output_dir.replace('{timestamp}', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Results will be saved to: {output_dir}")

    # Load and prepare data
    df = load_and_prepare_data(config)
    features, labels, original_model1, original_model2, label_encoder = prepare_features_and_labels(df, config)

    logging.info(f"\nDataset shape: {df.shape}")
    logging.info(f"Ground truth classes: {list(label_encoder.classes_)}")

    # Run k-fold cross-validation
    k = args.fold if args.fold else config.get('k_fold', 5)
    cv_results, cv_summary = run_k_fold_cv(
        df, features, labels, original_model1, original_model2,
        config, output_dir, k=k
    )

    # Save overall summary
    overall_summary = {
        'Evaluation_Method': [f'{k}-Fold CV (Patient-Level)'],
        'Best_Method': [cv_summary.loc[cv_summary['Mean_Accuracy'].idxmax(), 'Method']],
        'Best_Accuracy': [cv_summary['Mean_Accuracy'].max()]
    }

    overall_df = pd.DataFrame(overall_summary)
    overall_df.to_csv(os.path.join(output_dir, 'overall_summary.csv'), index=False)

    logging.info("=" * 80)
    logging.info("Analysis Complete!")
    logging.info("=" * 80)
    logging.info(f"All results saved to: {output_dir}")
    logging.info("\nGenerated files:")
    for file in sorted(os.listdir(output_dir)):
        logging.info(f"  - {file}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
