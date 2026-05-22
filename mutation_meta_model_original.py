import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import os
from datetime import datetime
import argparse

def load_and_prepare_data(data_path, model1_pred, model1_prob, model2_pred, model2_prob,
                          required_columns=['SAMPLE_ID', 'ground_truth', 'CANCER_TYPE']):
    """Load and prepare mutation status prediction data."""
    # Load the merged predictions
    data_path = "/data1/morrisq/yuj13/llm_genomics/llm_cancer_type_classification/analysis/1_mutation_status_prediction/merged_predictions_all_models.csv"
    df = pd.read_csv(data_path)

    # Print basic statistics
    print(f"Total samples: {len(df)}")
    print(f"Unique patients: {df['PATIENT_ID'].nunique()}")
    print(f"Ground truth distribution:")
    print(df['ground_truth'].value_counts())
    print(f"\nCancer types: {df['CANCER_TYPE'].nunique()}")

    # Select relevant columns for meta-model
    # Using gpt-4o and MetaCH as specified
    selected_columns = [
        'SAMPLE_ID', 'ground_truth', 'CANCER_TYPE', 'CANCER_TYPE_DETAILED',
        'gpt-4o_pred', 'gpt-4o_prob', 'MetaCH_pred', 'MetaCH_prob'
    ]

    df_selected = df[selected_columns].copy()

    # Drop rows with missing values
    df_selected = df_selected.dropna().reset_index(drop=True)

    print(f"\nDataset after cleaning: {len(df_selected)} samples")
    print(f"Unique cancer types: {df_selected['CANCER_TYPE'].nunique()}")

    return df_selected

def prepare_features_and_labels(df):
    """Prepare features and labels for machine learning models."""
    # Prepare features - using predictions and probabilities
    features = df[[
        'gpt-4o_pred', 'gpt-4o_prob',
        'MetaCH_pred', 'MetaCH_prob'
    ]]
    labels = df['ground_truth']

    # Encode categorical predictions
    features_encoded = pd.get_dummies(features, columns=[
        'gpt-4o_pred', 'MetaCH_pred', 'gpt-4o_prob', 'MetaCH_prob'
    ])

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Encode original predictions for comparison
    gpt4o_preds_encoded = label_encoder.transform(df['gpt-4o_pred'])
    metach_preds_encoded = label_encoder.transform(df['MetaCH_pred'])

    return features_encoded, labels_encoded, gpt4o_preds_encoded, metach_preds_encoded, label_encoder

def run_5_fold_cv(features, labels, original_gpt4o, original_metach, output_dir):
    """Run 5-fold cross-validation and save results."""
    print("\n===== Running 5-Fold Cross-Validation =====")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {
        'fold': [],
        'gpt-4o_accuracy': [], 'gpt-4o_f1': [], 'gpt-4o_precision': [], 'gpt-4o_recall': [],
        'MetaCH_accuracy': [], 'MetaCH_f1': [], 'MetaCH_precision': [], 'MetaCH_recall': [],
        'LogReg_accuracy': [], 'LogReg_f1': [], 'LogReg_precision': [], 'LogReg_recall': [],
        'XGBoost_accuracy': [], 'XGBoost_f1': [], 'XGBoost_precision': [], 'XGBoost_recall': [],
        'RandomForest_accuracy': [], 'RandomForest_f1': [], 'RandomForest_precision': [], 'RandomForest_recall': []
    }

    print("\n===== Per-Fold Metrics =====")
    for i, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Original model metrics
        gpt4o_preds = original_gpt4o[test_idx]
        metach_preds = original_metach[test_idx]

        acc_gpt4o = accuracy_score(y_test, gpt4o_preds)
        f1_gpt4o = f1_score(y_test, gpt4o_preds, average='weighted', zero_division=0)
        prec_gpt4o = precision_score(y_test, gpt4o_preds, average='weighted', zero_division=0)
        rec_gpt4o = recall_score(y_test, gpt4o_preds, average='weighted', zero_division=0)

        acc_metach = accuracy_score(y_test, metach_preds)
        f1_metach = f1_score(y_test, metach_preds, average='weighted', zero_division=0)
        prec_metach = precision_score(y_test, metach_preds, average='weighted', zero_division=0)
        rec_metach = recall_score(y_test, metach_preds, average='weighted', zero_division=0)

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

        results['gpt-4o_accuracy'].append(acc_gpt4o)
        results['gpt-4o_f1'].append(f1_gpt4o)
        results['gpt-4o_precision'].append(prec_gpt4o)
        results['gpt-4o_recall'].append(rec_gpt4o)

        results['MetaCH_accuracy'].append(acc_metach)
        results['MetaCH_f1'].append(f1_metach)
        results['MetaCH_precision'].append(prec_metach)
        results['MetaCH_recall'].append(rec_metach)

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
        print(f"  gpt-4o: Acc={acc_gpt4o:.4f}, F1={f1_gpt4o:.4f}")
        print(f"  MetaCH: Acc={acc_metach:.4f}, F1={f1_metach:.4f}")
        print(f"  LogReg: Acc={acc_log:.4f}, F1={f1_log:.4f}")
        print(f"  XGBoost: Acc={acc_xgb:.4f}, F1={f1_xgb:.4f}")
        print(f"  RF: Acc={acc_rf:.4f}, F1={f1_rf:.4f}")

    # Save detailed results to CSV
    results_df = pd.DataFrame(results)
    cv_detailed_file = os.path.join(output_dir, '5_fold_cv_detailed_results.csv')
    results_df.to_csv(cv_detailed_file, index=False)
    print(f"\n5-fold CV detailed results saved to: {cv_detailed_file}")

    # Calculate and save summary statistics
    methods = ['gpt-4o', 'MetaCH', 'LogReg', 'XGBoost', 'RandomForest']
    method_names = ['gpt-4o', 'MetaCH', 'LogReg', 'XGBoost', 'Random Forest']

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
    cv_summary_file = os.path.join(output_dir, '5_fold_cv_summary.csv')
    summary_df.to_csv(cv_summary_file, index=False)
    print(f"5-fold CV summary saved to: {cv_summary_file}")

    # Create and save plot
    plt.figure(figsize=(10, 6))
    plt.bar(method_names, summary_df['Mean_Accuracy'], yerr=summary_df['Std_Accuracy'], capsize=5)
    plt.ylabel("Accuracy")
    plt.title("5-Fold Cross-Validation Accuracy (Mean ± Std Dev)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_fold_cv_accuracy_plot.png'), dpi=300, bbox_inches='tight')

    # Print summary
    print("\n===== Cross-Validation Metrics Summary =====")
    for _, row in summary_df.iterrows():
        print(f"{row['Method']}:")
        print(f"  Accuracy: {row['Mean_Accuracy']:.4f} ± {row['Std_Accuracy']:.4f}")
        print(f"  F1 Score: {row['Mean_F1']:.4f} ± {row['Std_F1']:.4f}")
        print(f"  Precision: {row['Mean_Precision']:.4f} ± {row['Std_Precision']:.4f}")
        print(f"  Recall: {row['Mean_Recall']:.4f} ± {row['Std_Recall']:.4f}")

    return results_df, summary_df

def run_leave_one_cancer_type_out(df_original, features, labels, original_gpt4o, original_metach, label_encoder, output_dir):
    """Run leave-one-cancer-type-out cross-validation and save results."""
    print("\n===== Running Leave-One-Cancer-Type-Out Cross-Validation =====")

    cancer_types = df_original['CANCER_TYPE']
    unique_cancer_types = cancer_types.unique()

    results = {
        'CANCER_TYPE': [],
        'num_test_samples': [],
        'gpt-4o_accuracy': [], 'gpt-4o_f1': [], 'gpt-4o_precision': [], 'gpt-4o_recall': [],
        'MetaCH_accuracy': [], 'MetaCH_f1': [], 'MetaCH_precision': [], 'MetaCH_recall': [],
        'LogReg_accuracy': [], 'LogReg_f1': [], 'LogReg_precision': [], 'LogReg_recall': [],
        'XGBoost_accuracy': [], 'XGBoost_f1': [], 'XGBoost_precision': [], 'XGBoost_recall': [],
        'RandomForest_accuracy': [], 'RandomForest_f1': [], 'RandomForest_precision': [], 'RandomForest_recall': []
    }

    # Get all unique classes to ensure consistency across folds
    all_classes = np.unique(labels)
    n_classes = len(all_classes)

    for cancer_type in unique_cancer_types:
        print(f"\n===== Leave-One-Cancer-Type-Out: {cancer_type} =====")
        train_mask = cancer_types != cancer_type
        test_mask = cancer_types == cancer_type

        X_train = features[train_mask]
        X_test = features[test_mask]
        y_train = labels[train_mask]
        y_test = labels[test_mask]
        original_preds_gpt4o_test = original_gpt4o[test_mask]
        original_preds_metach_test = original_metach[test_mask]

        if len(y_test) == 0:
            print("No test samples for this cancer type. Skipping.")
            continue

        # Check if training set has all classes
        train_classes = np.unique(y_train)
        test_classes = np.unique(y_test)

        print(f"Training classes: {len(train_classes)}, Test classes: {len(test_classes)}")
        print(f"Test samples: {len(y_test)}")

        # Calculate metrics for original models
        acc_gpt4o = accuracy_score(y_test, original_preds_gpt4o_test)
        f1_gpt4o = f1_score(y_test, original_preds_gpt4o_test, average='weighted', zero_division=0)
        prec_gpt4o = precision_score(y_test, original_preds_gpt4o_test, average='weighted', zero_division=0)
        rec_gpt4o = recall_score(y_test, original_preds_gpt4o_test, average='weighted', zero_division=0)

        acc_metach = accuracy_score(y_test, original_preds_metach_test)
        f1_metach = f1_score(y_test, original_preds_metach_test, average='weighted', zero_division=0)
        prec_metach = precision_score(y_test, original_preds_metach_test, average='weighted', zero_division=0)
        rec_metach = recall_score(y_test, original_preds_metach_test, average='weighted', zero_division=0)

        # Initialize metric variables
        acc_log = f1_log = prec_log = rec_log = 0.0
        acc_xgb = f1_xgb = prec_xgb = rec_xgb = 0.0
        acc_rf = f1_rf = prec_rf = rec_rf = 0.0

        try:
            # Logistic Regression
            log_reg = LogisticRegression(max_iter=1000, random_state=42)
            log_reg.fit(X_train, y_train)
            log_preds = log_reg.predict(X_test)
            acc_log = accuracy_score(y_test, log_preds)
            f1_log = f1_score(y_test, log_preds, average='weighted', zero_division=0)
            prec_log = precision_score(y_test, log_preds, average='weighted', zero_division=0)
            rec_log = recall_score(y_test, log_preds, average='weighted', zero_division=0)
        except Exception as e:
            print(f"LogReg failed for cancer type {cancer_type}: {e}")
            acc_log = f1_log = prec_log = rec_log = 0.0

        try:
            # XGBoost with relabeled classes to ensure consecutiveness
            train_classes_sorted = np.sort(train_classes)
            class_mapping = {old_label: new_label for new_label, old_label in enumerate(train_classes_sorted)}
            reverse_mapping = {new_label: old_label for old_label, new_label in class_mapping.items()}

            # Relabel training data
            y_train_relabeled = np.array([class_mapping[label] for label in y_train])

            # Relabel test data (only for classes that exist in training)
            y_test_relabeled = []
            test_mask_valid = []
            for i, label in enumerate(y_test):
                if label in class_mapping:
                    y_test_relabeled.append(class_mapping[label])
                    test_mask_valid.append(i)

            if len(y_test_relabeled) > 0:
                y_test_relabeled = np.array(y_test_relabeled)
                X_test_valid = X_test.iloc[test_mask_valid] if hasattr(X_test, 'iloc') else X_test[test_mask_valid]

                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    num_class=len(train_classes_sorted),
                    objective='multi:softprob' if len(train_classes_sorted) > 2 else 'binary:logistic'
                )
                xgb_model.fit(X_train, y_train_relabeled)

                # Predict and map back to original labels
                preds_relabeled = xgb_model.predict(X_test_valid)
                preds_original = np.array([reverse_mapping[pred] for pred in preds_relabeled])

                # Calculate metrics only for valid test samples
                y_test_valid = y_test[test_mask_valid]
                acc_xgb = accuracy_score(y_test_valid, preds_original)
                f1_xgb = f1_score(y_test_valid, preds_original, average='weighted', zero_division=0)
                prec_xgb = precision_score(y_test_valid, preds_original, average='weighted', zero_division=0)
                rec_xgb = recall_score(y_test_valid, preds_original, average='weighted', zero_division=0)
            else:
                print(f"No valid test samples for XGBoost in cancer type {cancer_type}")
                acc_xgb = f1_xgb = prec_xgb = rec_xgb = 0.0

        except Exception as e:
            print(f"XGBoost failed for cancer type {cancer_type}: {e}")
            acc_xgb = f1_xgb = prec_xgb = rec_xgb = 0.0

        try:
            # Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_preds = rf_model.predict(X_test)
            acc_rf = accuracy_score(y_test, rf_preds)
            f1_rf = f1_score(y_test, rf_preds, average='weighted', zero_division=0)
            prec_rf = precision_score(y_test, rf_preds, average='weighted', zero_division=0)
            rec_rf = recall_score(y_test, rf_preds, average='weighted', zero_division=0)
        except Exception as e:
            print(f"RandomForest failed for cancer type {cancer_type}: {e}")
            acc_rf = f1_rf = prec_rf = rec_rf = 0.0

        # Store results
        results['CANCER_TYPE'].append(cancer_type)
        results['num_test_samples'].append(len(y_test))

        results['gpt-4o_accuracy'].append(acc_gpt4o)
        results['gpt-4o_f1'].append(f1_gpt4o)
        results['gpt-4o_precision'].append(prec_gpt4o)
        results['gpt-4o_recall'].append(rec_gpt4o)

        results['MetaCH_accuracy'].append(acc_metach)
        results['MetaCH_f1'].append(f1_metach)
        results['MetaCH_precision'].append(prec_metach)
        results['MetaCH_recall'].append(rec_metach)

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

        print(f"gpt-4o: Acc={acc_gpt4o:.4f}, F1={f1_gpt4o:.4f}")
        print(f"MetaCH: Acc={acc_metach:.4f}, F1={f1_metach:.4f}")
        print(f"LogReg: Acc={acc_log:.4f}, F1={f1_log:.4f}")
        print(f"XGBoost: Acc={acc_xgb:.4f}, F1={f1_xgb:.4f}")
        print(f"RF: Acc={acc_rf:.4f}, F1={f1_rf:.4f}")

    # Save detailed results to CSV
    results_df = pd.DataFrame(results)
    loco_detailed_file = os.path.join(output_dir, 'leave_one_cancer_type_out_detailed_results.csv')
    results_df.to_csv(loco_detailed_file, index=False)
    print(f"\nLeave-one-cancer-type-out detailed results saved to: {loco_detailed_file}")

    # Calculate and save summary statistics (excluding failed runs with 0.0 accuracy)
    def safe_stats(series):
        """Calculate stats excluding 0.0 values (failed runs)"""
        non_zero = series[series > 0.0]
        if len(non_zero) == 0:
            return 0.0, 0.0, 0.0, 0.0
        return non_zero.mean(), non_zero.std(), non_zero.min(), non_zero.max()

    summary_data = {
        'Method': ['gpt-4o', 'MetaCH', 'LogReg', 'XGBoost', 'Random Forest'],
        'Mean_Accuracy': [], 'Std_Accuracy': [],
        'Mean_F1': [], 'Std_F1': [],
        'Mean_Precision': [], 'Std_Precision': [],
        'Mean_Recall': [], 'Std_Recall': [],
        'Successful_Runs': []
    }

    for method_prefix in ['gpt-4o', 'MetaCH', 'LogReg', 'XGBoost', 'RandomForest']:
        # Accuracy
        mean_acc, std_acc, min_acc, max_acc = safe_stats(results_df[f'{method_prefix}_accuracy'])
        summary_data['Mean_Accuracy'].append(mean_acc)
        summary_data['Std_Accuracy'].append(std_acc)

        # F1
        mean_f1, std_f1, _, _ = safe_stats(results_df[f'{method_prefix}_f1'])
        summary_data['Mean_F1'].append(mean_f1)
        summary_data['Std_F1'].append(std_f1)

        # Precision
        mean_prec, std_prec, _, _ = safe_stats(results_df[f'{method_prefix}_precision'])
        summary_data['Mean_Precision'].append(mean_prec)
        summary_data['Std_Precision'].append(std_prec)

        # Recall
        mean_rec, std_rec, _, _ = safe_stats(results_df[f'{method_prefix}_recall'])
        summary_data['Mean_Recall'].append(mean_rec)
        summary_data['Std_Recall'].append(std_rec)

        # Successful runs
        successful_runs = len(results_df[results_df[f'{method_prefix}_accuracy'] > 0.0]) if method_prefix not in ['gpt-4o', 'MetaCH'] else len(results_df)
        summary_data['Successful_Runs'].append(successful_runs)

    summary_df = pd.DataFrame(summary_data)
    loco_summary_file = os.path.join(output_dir, 'leave_one_cancer_type_out_summary.csv')
    summary_df.to_csv(loco_summary_file, index=False)
    print(f"Leave-one-cancer-type-out summary saved to: {loco_summary_file}")

    # Create and save plot
    plt.figure(figsize=(16, 8))

    # Filter results for plotting
    plot_results = results_df.copy()
    cancer_types_sorted = sorted(plot_results['CANCER_TYPE'])
    x = np.arange(len(cancer_types_sorted))
    width = 0.15

    methods = ['gpt-4o_accuracy', 'MetaCH_accuracy', 'LogReg_accuracy', 'XGBoost_accuracy', 'RandomForest_accuracy']
    method_names = ['gpt-4o', 'MetaCH', 'Logistic Regression', 'XGBoost', 'Random Forest']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        accuracies = [plot_results[plot_results['CANCER_TYPE'] == c][method].iloc[0] for c in cancer_types_sorted]
        plt.bar(x + i*width - 2*width, accuracies, width=width, label=name, color=color)

    plt.ylabel("Accuracy")
    plt.xlabel("Cancer Type")
    plt.title("Leave-One-Cancer-Type-Out Accuracy per Model")
    plt.xticks(x, cancer_types_sorted, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'leave_one_cancer_type_out_accuracy_plot.png'), dpi=300, bbox_inches='tight')

    # Print summary
    print("\n===== Leave-One-Cancer-Type-Out Metrics Summary =====")
    for _, row in summary_df.iterrows():
        if row['Successful_Runs'] > 0:
            print(f"{row['Method']} ({row['Successful_Runs']} successful runs):")
            print(f"  Accuracy: {row['Mean_Accuracy']:.4f} ± {row['Std_Accuracy']:.4f}")
            print(f"  F1 Score: {row['Mean_F1']:.4f} ± {row['Std_F1']:.4f}")
            print(f"  Precision: {row['Mean_Precision']:.4f} ± {row['Std_Precision']:.4f}")
            print(f"  Recall: {row['Mean_Recall']:.4f} ± {row['Std_Recall']:.4f}")
        else:
            print(f"{row['Method']}: No successful runs")

    return results_df, summary_df

def main():
    """Main function to run the complete analysis."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/data1/morrisq/yuj13/llm_genomics/mutation_meta_model_result/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Results will be saved to: {output_dir}")

    # Load and prepare data
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    features, labels, original_gpt4o, original_metach, label_encoder = prepare_features_and_labels(df)

    print(f"\nDataset shape: {df.shape}")
    print(f"Number of unique cancer types: {df['CANCER_TYPE'].nunique()}")
    print(f"Ground truth classes: {label_encoder.classes_}")

    # Run 5-fold cross-validation
    cv_results, cv_summary = run_5_fold_cv(features, labels, original_gpt4o, original_metach, output_dir)

    # Run leave-one-cancer-type-out cross-validation
    loco_results, loco_summary = run_leave_one_cancer_type_out(
        df, features, labels, original_gpt4o, original_metach, label_encoder, output_dir
    )

    # Save overall summary
    overall_summary = {
        'Evaluation_Method': ['5-Fold CV', 'Leave-One-Cancer-Type-Out'],
        'Best_Method': [
            cv_summary.loc[cv_summary['Mean_Accuracy'].idxmax(), 'Method'],
            loco_summary.loc[loco_summary['Mean_Accuracy'].idxmax(), 'Method']
        ],
        'Best_Accuracy': [
            cv_summary['Mean_Accuracy'].max(),
            loco_summary['Mean_Accuracy'].max()
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
