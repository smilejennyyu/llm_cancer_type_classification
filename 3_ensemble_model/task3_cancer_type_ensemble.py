"""
Task 3: Cancer Type Ensemble Model
Combines predictions from task-specific model (GDD) and LLM using ensemble methods.

Uses Logistic Regression, Random Forest, and XGBoost to create meta-models.
Configured via YAML file. See configs/task3_cancer_type_ensemble_config.yaml
"""

import logging
import os
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ensemble_utils import load_config, setup_logging, normalize, run_k_fold_cv


# ============================================================================
# Data Loading and Preparation
# ============================================================================

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

    model1_name = config['model1']['name']
    model1_path = config['model1']['predictions_dir']
    logging.info(f"Loading {model1_name} predictions from {model1_path}")
    model1_df = load_predictions(model1_path)

    model2_name = config['model2']['name']
    model2_path = config['model2']['predictions_dir']
    logging.info(f"Loading {model2_name} predictions from {model2_path}")
    model2_df = load_predictions(model2_path)

    model1_rename = config['model1'].get('column_mapping', {})
    if model1_rename:
        model1_df = model1_df.rename(columns=model1_rename)

    model2_rename = config['model2'].get('column_mapping', {})
    if model2_rename:
        model2_df = model2_df.rename(columns=model2_rename)

    if config['model2'].get('normalize_predictions', False):
        for col in ['prediction1', 'prediction2']:
            if col in model2_df.columns:
                model2_df[col] = model2_df[col].fillna('indetermined').apply(normalize)

    if 'prediction_fixes' in config['model2']:
        for old, new in config['model2']['prediction_fixes'].items():
            for col in ['prediction1', 'prediction2']:
                if col in model2_df.columns:
                    model2_df.loc[model2_df[col] == old, col] = new

    joint_pred = model2_df.merge(
        model1_df,
        on='SAMPLE_ID',
        how='outer',
        suffixes=('_llm', '_model1')
    )

    joint_pred = joint_pred.rename(columns={
        'prediction1_llm': f'{model2_name}_Prediction1',
        'prediction2_llm': f'{model2_name}_Prediction2',
        'prob1_llm': f'{model2_name}_Conf1',
        'prob2_llm': f'{model2_name}_Conf2',
        'Pred1': f'{model1_name}_Prediction1',
        'Pred2': f'{model1_name}_Prediction2',
        'Conf1': f'{model1_name}_Conf1',
        'Conf2': f'{model1_name}_Conf2',
    })

    if 'clinical_data_path' in config:
        clinical_df = pd.read_csv(config['clinical_data_path'], sep='\t', comment='#')
        clinical_cols = config.get('clinical_columns', ['SAMPLE_ID', 'PATIENT_ID'])
        joint_pred = joint_pred.merge(clinical_df[clinical_cols], on='SAMPLE_ID', how='left')

    if 'assay_info_path' in config:
        assay_df = pd.read_table(config['assay_info_path'])
        assay_df_unique = assay_df[['SEQ_ASSAY_ID', 'CENTER']].drop_duplicates()
        seq_assay_to_center = dict(zip(assay_df_unique['SEQ_ASSAY_ID'], assay_df_unique['CENTER']))
        joint_pred['CENTER'] = joint_pred['SEQ_ASSAY_ID'].map(seq_assay_to_center)

    required_cols = config.get('required_columns', [
        f'{model2_name}_Prediction1',
        f'{model1_name}_Prediction1',
        'ground_truth',
    ])

    joint_pred = joint_pred.dropna(subset=required_cols).reset_index(drop=True)

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

    feature_cols = [
        f'{model2_name}_Prediction1', f'{model2_name}_Conf1',
        f'{model1_name}_Prediction1', f'{model1_name}_Conf1',
        f'{model2_name}_Prediction2', f'{model2_name}_Conf2',
        f'{model1_name}_Prediction2', f'{model1_name}_Conf2',
    ]

    features = joint_pred[feature_cols].copy()
    labels = joint_pred['ground_truth']

    cat_cols = [col for col in feature_cols if 'Prediction' in col]
    features_encoded = pd.get_dummies(features, columns=cat_cols)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    model1_preds_encoded = label_encoder.transform(joint_pred[f'{model1_name}_Prediction1'])

    model2_preds_encoded = joint_pred[f'{model2_name}_Prediction1'].apply(
        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else len(label_encoder.classes_)
    ).values

    return features_encoded, labels_encoded, model1_preds_encoded, model2_preds_encoded, label_encoder


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

    config = load_config(args.config)
    setup_logging(config)

    logging.info("=" * 80)
    logging.info("Cancer Type Ensemble Model")
    logging.info("=" * 80)
    logging.info(f"Configuration loaded from {args.config}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.get('output_dir', f"/data1/morrisq/yuj13/llm_genomics/genie_meta_model_result/{timestamp}")
    output_dir = output_dir.replace('{timestamp}', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Results will be saved to: {output_dir}")

    joint_pred = load_and_prepare_data(config)
    features, labels, original_model1, original_model2, label_encoder = prepare_features_and_labels(joint_pred, config)

    k = args.fold if args.fold else config.get('k_fold', 5)
    cv_results, cv_summary = run_k_fold_cv(
        joint_pred, features, labels, original_model1, original_model2,
        config, output_dir, k=k
    )

    overall_summary = {
        'Evaluation_Method': [f'{k}-Fold CV'],
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
