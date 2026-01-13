"""
Task 3: Cancer Type Ensemble Model
Combines predictions from task-specific model (GDD) and LLM using ensemble methods.

Uses Logistic Regression, Random Forest, and XGBoost to create meta-models.
Configured via YAML file. See configs/task3_cancer_type_ensemble_config.yaml
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import logging
import argparse
import yaml
import unicodedata
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

def normalize(name: str) -> str:
    """Normalize cancer type names by handling Unicode characters and punctuation."""
    # Normalize unicode (e.g., replace en dash/em dash with hyphen)
    name = unicodedata.normalize('NFKD', name)
    name = name.replace('‑', '-')  # non-breaking hyphen
    name = name.replace('‐', '-')  # hyphen
    name = name.replace('–', '-')  # en dash
    name = name.replace('—', '-')  # em dash
    name = name.replace(',', '')  # remove commas
    return name


def load_predictions(directory: str, sample_filter=None) -> pd.DataFrame:
    """Load all CSV files from a directory and concatenate."""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    if not csv_files:
        logging.warning(f"No CSV files found in {directory}")
        return pd.DataFrame()

    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    if sample_filter is not None:
        df = df[df["SAMPLE_ID"].isin(sample_filter["SAMPLE_ID"])]

    return df


def load_and_prepare_data(config: dict) -> pd.DataFrame:
    """Load and prepare the genomic data for analysis."""
    logging.info("=" * 80)
    logging.info("Loading and preparing data")
    logging.info("=" * 80)

    # Load task-specific model predictions (e.g., GDD)
    model1_name = config['model1']['name']
    model1_path = config['model1']['predictions_dir']

    logging.info(f"Loading {model1_name} predictions from {model1_path}")
    model1_df = load_predictions(model1_path)

    # Load LLM predictions
    model2_name = config['model2']['name']
    model2_path = config['model2']['predictions_dir']

    logging.info(f"Loading {model2_name} predictions from {model2_path}")
    model2_df = load_predictions(model2_path)

    # Rename columns for model1 (task-specific model)
    model1_rename = config['model1'].get('column_mapping', {})
    if model1_rename:
        model1_df = model1_df.rename(columns=model1_rename)

    # Rename columns for model2 (LLM)
    model2_rename = config['model2'].get('column_mapping', {})
    if model2_rename:
        model2_df = model2_df.rename(columns=model2_rename)

    # Normalize LLM predictions if enabled
    if config['model2'].get('normalize_predictions', False):
        for col in ['prediction1', 'prediction2']:
            if col in model2_df.columns:
                model2_df[col] = model2_df[col].fillna('indetermined')
                model2_df[col] = model2_df[col].apply(normalize)

    # Apply any specific fixes
    if 'prediction_fixes' in config['model2']:
        for old, new in config['model2']['prediction_fixes'].items():
            for col in ['prediction1', 'prediction2']:
                if col in model2_df.columns:
                    model2_df.loc[model2_df[col] == old, col] = new

    # Merge predictions
    joint_pred = model2_df.merge(
        model1_df,
        on='SAMPLE_ID',
        how='outer',
        suffixes=('_llm', '_model1')
    )

    # Standardize column names
    joint_pred = joint_pred.rename(columns={
        'prediction1_llm': f'{model2_name}_Prediction1',
        'prediction2_llm': f'{model2_name}_Prediction2',
        'prob1_llm': f'{model2_name}_Conf1',
        'prob2_llm': f'{model2_name}_Conf2',
        'Pred1': f'{model1_name}_Prediction1',
        'Pred2': f'{model1_name}_Prediction2',
        'Conf1': f'{model1_name}_Conf1',
        'Conf2': f'{model1_name}_Conf2'
    })

    # Load clinical data if path provided
    if 'clinical_data_path' in config:
        clinical_df = pd.read_csv(config['clinical_data_path'], sep='\t', comment='#')

        # Merge with clinical data
        clinical_cols = config.get('clinical_columns', ['SAMPLE_ID', 'PATIENT_ID'])
        joint_pred = joint_pred.merge(
            clinical_df[clinical_cols],
            on='SAMPLE_ID',
            how='left'
        )

    # Load center information if assay info path provided
    if 'assay_info_path' in config:
        assay_df = pd.read_table(config['assay_info_path'])
        assay_df_unique = assay_df[['SEQ_ASSAY_ID', 'CENTER']].drop_duplicates()
        seq_assay_to_center = dict(zip(assay_df_unique['SEQ_ASSAY_ID'], assay_df_unique['CENTER']))
        joint_pred['CENTER'] = joint_pred['SEQ_ASSAY_ID'].map(seq_assay_to_center)

    # Define required columns
    required_cols = config.get('required_columns', [
        f'{model2_name}_Prediction1',
        f'{model1_name}_Prediction1',
        'ground_truth'
    ])

    # Drop rows with missing required columns
    joint_pred = joint_pred.dropna(subset=required_cols).reset_index(drop=True)

    # Print data quality summary
    logging.info(f"Final dataset shape: {joint_pred.shape}")
    logging.info(f"Unique ground truth cancer types: {joint_pred['ground_truth'].nunique()}")
    logging.info(f"Unique {model1_name} predictions: {joint_pred[f'{model1_name}_Prediction1'].nunique()}")
    logging.info(f"Unique {model2_name} predictions: {joint_pred[f'{model2_name}_Prediction1'].nunique()}")

    if 'PATIENT_ID' in joint_pred.columns:
        logging.info(f"Unique patients: {joint_pred['PATIENT_ID'].nunique()}")
    if 'CENTER' in joint_pred.columns:
        logging.info(f"Unique centers: {joint_pred['CENTER'].nunique()}")

    return joint_pred


def prepare_features_and_labels(joint_pred: pd.DataFrame, config: dict):
    """Prepare features and labels for machine learning models."""
    model1_name = config['model1']['name']
    model2_name = config['model2']['name']

    # Define feature columns
    feature_cols = [
        f'{model2_name}_Prediction1', f'{model2_name}_Conf1',
        f'{model1_name}_Prediction1', f'{model1_name}_Conf1',
        f'{model2_name}_Prediction2', f'{model2_name}_Conf2',
        f'{model1_name}_Prediction2', f'{model1_name}_Conf2'
    ]

    features = joint_pred[feature_cols].copy()
    labels = joint_pred['ground_truth']

    # Encode categorical predictions
    cat_cols = [col for col in feature_cols if 'Prediction' in col]
    features_encoded = pd.get_dummies(features, columns=cat_cols)

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Encode original predictions for baseline comparison
    model1_preds_encoded = label_encoder.transform(joint_pred[f'{model1_name}_Prediction1'])

    # Handle LLM predictions with potential unseen classes
    model2_preds = joint_pred[f'{model2_name}_Prediction1'].apply(
        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else len(label_encoder.classes_)
    )
    model2_preds_encoded = model2_preds.values

    return features_encoded, labels_encoded, model1_preds_encoded, model2_preds_encoded, label_encoder


# ============================================================================
# Cross-Validation
# ============================================================================

def run_k_fold_cv(joint_pred: pd.DataFrame, features: pd.DataFrame, labels: np.ndarray,
                  original_model1: np.ndarray, original_model2: np.ndarray,
                  config: dict, output_dir: str, k: int = 5):
    """Run k-fold cross-validation split by PATIENT_ID."""
    logging.info("=" * 80)
    logging.info(f"Running {k}-Fold Cross-Validation (Split by PATIENT_ID)")
    logging.info("=" * 80)

    model1_name = config['model1']['name']
    model2_name = config['model2']['name']

    # Get unique patient IDs
    unique_patients = joint_pred['PATIENT_ID'].unique()
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
        test_mask = joint_pred['PATIENT_ID'].isin(test_patients)
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
        description='Ensemble model for cancer type classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python task3_cancer_type_ensemble.py --config ../configs/task3_cancer_type_ensemble_config.yaml
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

    logging.info("=" * 80)
    logging.info("Cancer Type Ensemble Model")
    logging.info("=" * 80)
    logging.info(f"Configuration loaded from {args.config}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.get('output_dir', f"/data1/morrisq/yuj13/llm_genomics/genie_meta_model_result/{timestamp}")
    output_dir = output_dir.replace('{timestamp}', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Results will be saved to: {output_dir}")

    # Load and prepare data
    joint_pred = load_and_prepare_data(config)
    features, labels, original_model1, original_model2, label_encoder = prepare_features_and_labels(joint_pred, config)

    # Run k-fold cross-validation
    k = args.fold if args.fold else config.get('k_fold', 5)
    cv_results, cv_summary = run_k_fold_cv(
        joint_pred, features, labels, original_model1, original_model2,
        config, output_dir, k=k
    )

    # Save overall summary
    overall_summary = {
        'Evaluation_Method': [f'{k}-Fold CV'],
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
