import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import argparse

def load_and_prepare_data(data_path, model1_pred, model1_prob, model2_pred, model2_prob,
                          required_columns=['SAMPLE_ID', 'PATIENT_ID', 'ground_truth', 'CANCER_TYPE']):
    """Load and prepare mutation status prediction data."""
    # Load the merged predictions
    # data_path passed as parameter
    df = pd.read_csv(data_path)

    # Print basic statistics
    print(f"Total samples: {len(df)}")
    print(f"Unique patients: {df['PATIENT_ID'].nunique()}")
    print(f"Ground truth distribution:")
    print(df['ground_truth'].value_counts())
    print(f"\nCancer types: {df['CANCER_TYPE'].nunique()}")

    # Select relevant columns for meta-model
    selected_columns = required_columns + [model1_pred, model1_prob, model2_pred, model2_prob]

    # Only keep columns that exist in the dataframe
    selected_columns = [col for col in selected_columns if col in df.columns]

    df_selected = df[selected_columns].copy()

    # Drop rows with missing values
    df_selected = df_selected.dropna().reset_index(drop=True)

    print(f"\nDataset after cleaning: {len(df_selected)} samples")
    print(f"Unique cancer types: {df_selected['CANCER_TYPE'].nunique()}")

    return df_selected

def prepare_features_and_labels(df, model1_pred, model1_prob, model2_pred, model2_prob):
    """Prepare features and labels for machine learning models."""
    # Prepare features - using predictions and probabilities
    features = df[[model1_pred, model1_prob, model2_pred, model2_prob]]
    labels = df['ground_truth']

    # Encode categorical predictions
    features_encoded = pd.get_dummies(features, columns=[model1_pred, model2_pred, model1_prob, model2_prob])

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Encode original predictions for comparison
    model1_preds_encoded = label_encoder.transform(df[model1_pred])
    model2_preds_encoded = label_encoder.transform(df[model2_pred])

    return features_encoded, labels_encoded, model1_preds_encoded, model2_preds_encoded, label_encoder

def run_5_fold_cv(df, features, labels, original_model1, original_model2, model1_name, model2_name, output_dir, fold=5):
    """Run 5-fold cross-validation split by PATIENT_ID and save results."""
    print(f"\n===== Running {fold}-Fold Cross-Validation (Split by PATIENT_ID) =====")

    # Get unique patient IDs
    unique_patients = df['PATIENT_ID'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_patients)

    # Split patients into folds
    fold_size = len(unique_patients) // fold
    patient_folds = [unique_patients[i*fold_size:(i+1)*fold_size] for i in range(fold-1)]
    patient_folds.append(unique_patients[(fold-1)*fold_size:])  # Last fold gets remaining patients

    results = {
        'fold': [],
        f'{model1_name}_accuracy': [], f'{model1_name}_f1': [], f'{model1_name}_precision': [], f'{model1_name}_recall': [],
        f'{model2_name}_accuracy': [], f'{model2_name}_f1': [], f'{model2_name}_precision': [], f'{model2_name}_recall': [],
        'LogReg_accuracy': [], 'LogReg_f1': [], 'LogReg_precision': [], 'LogReg_recall': [],
        'XGBoost_accuracy': [], 'XGBoost_f1': [], 'XGBoost_precision': [], 'XGBoost_recall': [],
        'RandomForest_accuracy': [], 'RandomForest_f1': [], 'RandomForest_precision': [], 'RandomForest_recall': []
    }

    print("\n===== Per-Fold Metrics =====")
    for i, test_patients in enumerate(patient_folds):
        # Create boolean masks based on patient IDs
        test_mask = df['PATIENT_ID'].isin(test_patients)
        train_mask = ~test_mask

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        print(f"\nFold {i+1}: {len(test_patients)} patients, {len(test_idx)} samples in test set")

        # Original model metrics
        model1_preds = original_model1[test_idx]
        model2_preds = original_model2[test_idx]

        acc_model1 = accuracy_score(y_test, model1_preds)
        f1_model1 = f1_score(y_test, model1_preds, average='weighted', zero_division=0)
        prec_model1 = precision_score(y_test, model1_preds, average='weighted', zero_division=0)
        rec_model1 = recall_score(y_test, model1_preds, average='weighted', zero_division=0)

        acc_model2 = accuracy_score(y_test, model2_preds)
        f1_model2 = f1_score(y_test, model2_preds, average='weighted', zero_division=0)
        prec_model2 = precision_score(y_test, model2_preds, average='weighted', zero_division=0)
        rec_model2 = recall_score(y_test, model2_preds, average='weighted', zero_division=0)

        # Train and evaluate ensemble models
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

        print(f"Fold {i+1}:")
        print(f"  {model1_name}: Acc={acc_model1:.4f}, F1={f1_model1:.4f}")
        print(f"  {model2_name}: Acc={acc_model2:.4f}, F1={f1_model2:.4f}")
        print(f"  LogReg: Acc={acc_log:.4f}, F1={f1_log:.4f}")
        print(f"  XGBoost: Acc={acc_xgb:.4f}, F1={f1_xgb:.4f}")
        print(f"  RF: Acc={acc_rf:.4f}, F1={f1_rf:.4f}")

    # Save detailed results to CSV
    results_df = pd.DataFrame(results)
    cv_detailed_file = os.path.join(output_dir, f'{fold}_fold_cv_detailed_results.csv')
    results_df.to_csv(cv_detailed_file, index=False)
    print(f"\n{fold}-fold CV detailed results saved to: {cv_detailed_file}")

    # Calculate and save summary statistics
    methods = [model1_name, model2_name, 'LogReg', 'XGBoost', 'RandomForest']
    method_names = [model1_name, model2_name, 'LogReg', 'XGBoost', 'Random Forest']

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
    cv_summary_file = os.path.join(output_dir, f'{fold}_fold_cv_summary.csv')
    summary_df.to_csv(cv_summary_file, index=False)
    print(f"{fold}-fold CV summary saved to: {cv_summary_file}")

    # Create and save plot
    plt.figure(figsize=(10, 6))
    plt.bar(method_names, summary_df['Mean_Accuracy'], yerr=summary_df['Std_Accuracy'], capsize=5)
    plt.ylabel("Accuracy")
    plt.title(f"{fold}-Fold Cross-Validation Accuracy (Mean ± Std Dev)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fold}_fold_cv_accuracy_plot.png'), dpi=300, bbox_inches='tight')

    # Print summary
    print("\n===== Cross-Validation Metrics Summary =====")
    for _, row in summary_df.iterrows():
        print(f"{row['Method']}:")
        print(f"  Accuracy: {row['Mean_Accuracy']:.4f} ± {row['Std_Accuracy']:.4f}")
        print(f"  F1 Score: {row['Mean_F1']:.4f} ± {row['Std_F1']:.4f}")
        print(f"  Precision: {row['Mean_Precision']:.4f} ± {row['Std_Precision']:.4f}")
        print(f"  Recall: {row['Mean_Recall']:.4f} ± {row['Std_Recall']:.4f}")

    return results_df, summary_df

def main():
    """Main function to run the complete analysis."""
    parser = argparse.ArgumentParser(description='Run meta-model analysis on mutation predictions')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the merged predictions CSV file')
    parser.add_argument('--model1_pred', type=str, required=True, help='Column name for model 1 predictions')
    parser.add_argument('--model1_prob', type=str, required=True, help='Column name for model 1 probabilities')
    parser.add_argument('--model2_pred', type=str, required=True, help='Column name for model 2 predictions')
    parser.add_argument('--model2_prob', type=str, required=True, help='Column name for model 2 probabilities')
    parser.add_argument('--model1_name', type=str, required=True, help='Display name for model 1 (e.g., gpt-5)')
    parser.add_argument('--model2_name', type=str, required=True, help='Display name for model 2 (e.g., AlphaMissense)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: auto-generated)')
    parser.add_argument('--fold', type=int, default=5, help='Number of folds for cross-validation (default: 5)')

    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"/data1/morrisq/yuj13/llm_genomics/mutation_meta_model_result/{args.model1_name}_{args.model2_name}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Load and prepare data
    print("Loading and preparing data...")
    df = load_and_prepare_data(args.data_path, args.model1_pred, args.model1_prob,
                                args.model2_pred, args.model2_prob)

    features, labels, original_model1, original_model2, label_encoder = prepare_features_and_labels(
        df, args.model1_pred, args.model1_prob, args.model2_pred, args.model2_prob
    )

    print(f"\nDataset shape: {df.shape}")
    if 'CANCER_TYPE' in df.columns:
        print(f"Number of unique cancer types: {df['CANCER_TYPE'].nunique()}")
    print(f"Ground truth classes: {label_encoder.classes_}")

    # Run n-fold cross-validation (split by PATIENT_ID)
    cv_results, cv_summary = run_5_fold_cv(df, features, labels, original_model1, original_model2,
                                           args.model1_name, args.model2_name, output_dir, fold=args.fold)

    # Save overall summary
    overall_summary = {
        'Evaluation_Method': [f'{args.fold}-Fold CV (Patient-Level)'],
        'Best_Method': [
            cv_summary.loc[cv_summary['Mean_Accuracy'].idxmax(), 'Method']
        ],
        'Best_Accuracy': [
            cv_summary['Mean_Accuracy'].max()
        ]
    }

    overall_df = pd.DataFrame(overall_summary)
    overall_df.to_csv(os.path.join(output_dir, 'overall_summary.csv'), index=False)

    print(f"\n===== Analysis Complete =====")
    print(f"All results saved to: {output_dir}")
    print("\nFiles generated:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")

if __name__ == "__main__":
    main()
