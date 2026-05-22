"""Shared utilities for ensemble models (Tasks 1, 2, and 3)."""

import logging
import os
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    log = config.get('logging', {})
    logging.basicConfig(
        level=getattr(logging, log.get('level', 'INFO')),
        format=log.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    )


def normalize(name: str) -> str:
    """Normalize cancer type names by handling Unicode characters and punctuation."""
    name = unicodedata.normalize('NFKD', name)
    name = name.replace('‑', '-')  # non-breaking hyphen
    name = name.replace('‐', '-')  # hyphen
    name = name.replace('–', '-')  # en dash
    name = name.replace('—', '-')  # em dash
    name = name.replace(',', '')
    return name


def run_k_fold_cv(
    df: pd.DataFrame,
    features: pd.DataFrame,
    labels: np.ndarray,
    original_model1: np.ndarray,
    original_model2: np.ndarray,
    config: dict,
    output_dir: str,
    k: int = 5,
):
    """Run k-fold cross-validation split by PATIENT_ID and save results."""
    logging.info("=" * 80)
    logging.info(f"Running {k}-Fold Cross-Validation (Split by PATIENT_ID)")
    logging.info("=" * 80)

    model1_name = config['model1']['name']
    model2_name = config['model2']['name']

    unique_patients = df['PATIENT_ID'].unique()
    np.random.seed(config.get('random_seed', 42))
    np.random.shuffle(unique_patients)

    fold_size = len(unique_patients) // k
    patient_folds = [unique_patients[i * fold_size:(i + 1) * fold_size] for i in range(k - 1)]
    patient_folds.append(unique_patients[(k - 1) * fold_size:])

    results = {
        'fold': [],
        f'{model1_name}_accuracy': [], f'{model1_name}_f1': [],
        f'{model1_name}_precision': [], f'{model1_name}_recall': [],
        f'{model2_name}_accuracy': [], f'{model2_name}_f1': [],
        f'{model2_name}_precision': [], f'{model2_name}_recall': [],
        'LogReg_accuracy': [], 'LogReg_f1': [], 'LogReg_precision': [], 'LogReg_recall': [],
        'XGBoost_accuracy': [], 'XGBoost_f1': [], 'XGBoost_precision': [], 'XGBoost_recall': [],
        'RandomForest_accuracy': [], 'RandomForest_f1': [],
        'RandomForest_precision': [], 'RandomForest_recall': [],
    }

    for i, test_patients in enumerate(patient_folds):
        test_mask = df['PATIENT_ID'].isin(test_patients)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]

        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        logging.info(f"Fold {i+1}: {len(test_patients)} patients, {len(test_idx)} samples in test set")

        model1_preds = original_model1[test_idx]
        model2_preds = original_model2[test_idx]

        acc_m1 = accuracy_score(y_test, model1_preds)
        f1_m1 = f1_score(y_test, model1_preds, average='weighted', zero_division=0)
        prec_m1 = precision_score(y_test, model1_preds, average='weighted', zero_division=0)
        rec_m1 = recall_score(y_test, model1_preds, average='weighted', zero_division=0)

        acc_m2 = accuracy_score(y_test, model2_preds)
        f1_m2 = f1_score(y_test, model2_preds, average='weighted', zero_division=0)
        prec_m2 = precision_score(y_test, model2_preds, average='weighted', zero_division=0)
        rec_m2 = recall_score(y_test, model2_preds, average='weighted', zero_division=0)

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

        results['fold'].append(i + 1)

        results[f'{model1_name}_accuracy'].append(acc_m1)
        results[f'{model1_name}_f1'].append(f1_m1)
        results[f'{model1_name}_precision'].append(prec_m1)
        results[f'{model1_name}_recall'].append(rec_m1)

        results[f'{model2_name}_accuracy'].append(acc_m2)
        results[f'{model2_name}_f1'].append(f1_m2)
        results[f'{model2_name}_precision'].append(prec_m2)
        results[f'{model2_name}_recall'].append(rec_m2)

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

        logging.info(
            f"  {model1_name}: {acc_m1:.4f}, {model2_name}: {acc_m2:.4f}, "
            f"LogReg: {acc_log:.4f}, XGBoost: {acc_xgb:.4f}, RF: {acc_rf:.4f}"
        )

    results_df = pd.DataFrame(results)
    cv_detailed_file = os.path.join(output_dir, f'{k}_fold_cv_detailed_results.csv')
    results_df.to_csv(cv_detailed_file, index=False)
    logging.info(f"{k}-fold CV detailed results saved to: {cv_detailed_file}")

    methods = [model1_name, model2_name, 'LogReg', 'XGBoost', 'RandomForest']
    method_names = [model1_name, model2_name, 'Logistic Regression', 'XGBoost', 'Random Forest']

    summary_data = {
        'Method': method_names,
        'Mean_Accuracy': [results_df[f'{m}_accuracy'].mean() for m in methods],
        'Std_Accuracy': [results_df[f'{m}_accuracy'].std() for m in methods],
        'Mean_F1': [results_df[f'{m}_f1'].mean() for m in methods],
        'Std_F1': [results_df[f'{m}_f1'].std() for m in methods],
        'Mean_Precision': [results_df[f'{m}_precision'].mean() for m in methods],
        'Std_Precision': [results_df[f'{m}_precision'].std() for m in methods],
        'Mean_Recall': [results_df[f'{m}_recall'].mean() for m in methods],
        'Std_Recall': [results_df[f'{m}_recall'].std() for m in methods],
    }

    summary_df = pd.DataFrame(summary_data)
    cv_summary_file = os.path.join(output_dir, f'{k}_fold_cv_summary.csv')
    summary_df.to_csv(cv_summary_file, index=False)
    logging.info(f"{k}-fold CV summary saved to: {cv_summary_file}")

    plt.figure(figsize=(10, 6))
    plt.bar(method_names, summary_df['Mean_Accuracy'], yerr=summary_df['Std_Accuracy'], capsize=5)
    plt.ylabel("Accuracy")
    plt.title(f"{k}-Fold Cross-Validation Accuracy (Mean ± Std Dev)")
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{k}_fold_cv_accuracy_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logging.info("\nCross-Validation Metrics Summary:")
    for _, row in summary_df.iterrows():
        logging.info(f"{row['Method']}:")
        logging.info(f"  Accuracy: {row['Mean_Accuracy']:.4f} ± {row['Std_Accuracy']:.4f}")
        logging.info(f"  F1 Score: {row['Mean_F1']:.4f} ± {row['Std_F1']:.4f}")

    return results_df, summary_df
