"""
Tasks 1 & 2: Mutation Status and Oncogenic Ensemble Model
Combines predictions from two models using ensemble methods.

Task 1: Mutation Status (TUMOR-SOMATIC vs CHIP)
Task 2: Oncogenic Classification (Oncogenic vs Benign)

Uses Logistic Regression, Random Forest, and XGBoost to create meta-models.
Configured via YAML file. See configs/task1_mutation_ensemble_config.yaml or task2_oncogenic_ensemble_config.yaml
"""

import logging
import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ensemble_utils import load_config, setup_logging, run_k_fold_cv


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_and_prepare_data(config: dict) -> pd.DataFrame:
    """Load and prepare mutation prediction data."""
    logging.info("=" * 80)
    logging.info("Loading and preparing data")
    logging.info("=" * 80)

    data_path = config['input']['data_path']
    logging.info(f"Loading data from {data_path}")

    df = pd.read_csv(data_path)

    logging.info(f"Total samples: {len(df)}")

    if 'PATIENT_ID' in df.columns:
        logging.info(f"Unique patients: {df['PATIENT_ID'].nunique()}")

    if 'ground_truth' in df.columns:
        logging.info("Ground truth distribution:")
        for label, count in df['ground_truth'].value_counts().items():
            logging.info(f"  {label}: {count}")

    if 'CANCER_TYPE' in df.columns:
        logging.info(f"Cancer types: {df['CANCER_TYPE'].nunique()}")

    model1_pred = config['model1']['prediction_column']
    model1_prob = config['model1']['probability_column']
    model2_pred = config['model2']['prediction_column']
    model2_prob = config['model2']['probability_column']

    required_cols = config.get('required_columns', ['SAMPLE_ID', 'PATIENT_ID', 'ground_truth'])
    selected_columns = required_cols + [model1_pred, model1_prob, model2_pred, model2_prob]
    selected_columns = [col for col in selected_columns if col in df.columns]

    df_selected = df[selected_columns].copy().dropna().reset_index(drop=True)

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

    features = df[[model1_pred, model1_prob, model2_pred, model2_prob]].copy()
    labels = df['ground_truth']

    features_encoded = pd.get_dummies(features, columns=[model1_pred, model2_pred])
    features_encoded[model1_prob] = pd.to_numeric(features_encoded[model1_prob], errors='coerce').fillna(0)
    features_encoded[model2_prob] = pd.to_numeric(features_encoded[model2_prob], errors='coerce').fillna(0)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    model1_preds_encoded = label_encoder.transform(df[model1_pred])
    model2_preds_encoded = label_encoder.transform(df[model2_pred])

    return features_encoded, labels_encoded, model1_preds_encoded, model2_preds_encoded, label_encoder


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

    config = load_config(args.config)
    setup_logging(config)

    task_name = config.get('task_name', 'Mutation Ensemble Model')

    logging.info("=" * 80)
    logging.info(task_name)
    logging.info("=" * 80)
    logging.info(f"Configuration loaded from {args.config}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = config.get('output_dir', "/data1/morrisq/yuj13/llm_genomics/mutation_meta_model_result")
    output_dir = f"{output_base}/{config['model1']['name']}_{config['model2']['name']}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Results will be saved to: {output_dir}")

    df = load_and_prepare_data(config)
    features, labels, original_model1, original_model2, label_encoder = prepare_features_and_labels(df, config)

    logging.info(f"\nDataset shape: {df.shape}")
    logging.info(f"Ground truth classes: {list(label_encoder.classes_)}")

    k = args.fold if args.fold else config.get('k_fold', 5)
    cv_results, cv_summary = run_k_fold_cv(
        df, features, labels, original_model1, original_model2,
        config, output_dir, k=k
    )

    overall_summary = {
        'Evaluation_Method': [f'{k}-Fold CV (Patient-Level)'],
        'Best_Method': [cv_summary.loc[cv_summary['Mean_Accuracy'].idxmax(), 'Method']],
        'Best_Accuracy': [cv_summary['Mean_Accuracy'].max()],
    }

    pd.DataFrame(overall_summary).to_csv(os.path.join(output_dir, 'overall_summary.csv'), index=False)

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
