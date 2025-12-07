import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import unicodedata
from datetime import datetime

def normalize(name):
    """Normalize cancer type names by handling Unicode characters and punctuation."""
    # Normalize unicode (e.g., replace en dash/em dash with hyphen)
    name = unicodedata.normalize('NFKD', name)
    name = name.replace('‑', '-')  # non-breaking hyphen
    name = name.replace('‐', '-')  # hyphen
    name = name.replace('–', '-')  # en dash
    name = name.replace('—', '-')  # em dash
    name = name.replace(',', '')  # remove commas
    # name = name.strip().lower()  # lowercase for uniformity
    return name

def load_and_prepare_data():
    """Load and prepare the genomic data for analysis."""
    # Define paths
    gdd_path = "/data1/morrisq/yuj13/llm_genomics/genie_output/GDD/filtered_batch_msk"
    gpt_5_path = "/data1/morrisq/yuj13/llm_genomics/genie_output/gpt-5/filtered_batch_msk"
    medgemma_path = "/data1/morrisq/yuj13/llm_genomics/genie_output/medgemma/filtered_batch_msk"
    
    # Load validation set
    val_set = pd.read_csv("/data1/morrisq/yuj13/llm_genomics/genie_output/o3-mini/output_nosite_extended_o3-mini.csv")
    val_set["SAMPLE_ID"] = "GENIE-MSK-" + val_set["SAMPLE_ID"].astype(str)
    
    # Load clinical data and mapping
    data_clinical_sample = pd.read_csv('/data1/morrisq/yuj13/llm_genomics/genie_data_v17.0/data_clinical_sample.txt',sep='\t',comment='#')
    mapping_df = pd.read_table("/data1/morrisq/yuj13/llm_genomics/GDD_ENS/data/tumor_type_final.txt")
    mapping_df_unique = mapping_df[['CANCER_TYPE', 'Cancer_Type']].drop_duplicates()
    mapping = dict(zip(mapping_df_unique['Cancer_Type'], mapping_df_unique['CANCER_TYPE']))
    
    # Load MSK data
    val_cancertypes_gdd_msk = pd.concat((pd.read_csv(f) for f in glob.glob(os.path.join(gdd_path, "*.csv"))), ignore_index=True)
    val_cancertypes_gdd_msk = val_cancertypes_gdd_msk[val_cancertypes_gdd_msk["SAMPLE_ID"].isin(val_set["SAMPLE_ID"])]
    
    val_cancertypes_msk = pd.concat((pd.read_csv(f) for f in glob.glob(os.path.join(gpt_5_path, "*.csv"))), ignore_index=True)
    val_cancertypes_msk = val_cancertypes_msk[val_cancertypes_msk["SAMPLE_ID"].isin(val_set["SAMPLE_ID"])]
    
    val_cancertypes_gemma_msk = pd.concat((pd.read_csv(f) for f in glob.glob(os.path.join(medgemma_path, "*.csv"))), ignore_index=True)
    val_cancertypes_gemma_msk = val_cancertypes_gemma_msk[val_cancertypes_gemma_msk["SAMPLE_ID"].isin(val_set["SAMPLE_ID"])]
    
    # Load non-MSK data
    gdd_path_non_msk = "/data1/morrisq/yuj13/llm_genomics/genie_output/GDD/filtered_batch_non_msk"
    gpt_5_path_non_msk = "/data1/morrisq/yuj13/llm_genomics/genie_output/gpt-5/filtered_batch_non_msk"
    medgemma_path_non_msk = "/data1/morrisq/yuj13/llm_genomics/genie_output/medgemma/filtered_batch_non_msk"
    
    val_cancertypes_gdd = pd.concat((pd.read_csv(f) for f in glob.glob(os.path.join(gdd_path_non_msk, "*.csv"))), ignore_index=True)
    val_cancertypes = pd.concat((pd.read_csv(f) for f in glob.glob(os.path.join(gpt_5_path_non_msk, "*.csv"))), ignore_index=True)
    val_cancertypes_gemma = pd.concat((pd.read_csv(f) for f in glob.glob(os.path.join(medgemma_path_non_msk, "*.csv"))), ignore_index=True)
    
    # Combine MSK and non-MSK data
    val_cancertypes_gdd = pd.concat([val_cancertypes_gdd, val_cancertypes_gdd_msk], ignore_index=True).drop_duplicates(subset="SAMPLE_ID")
    val_cancertypes = pd.concat([val_cancertypes, val_cancertypes_msk], ignore_index=True).drop_duplicates(subset="SAMPLE_ID")
    val_cancertypes_gemma = pd.concat([val_cancertypes_gemma, val_cancertypes_gemma_msk], ignore_index=True).drop_duplicates(subset="SAMPLE_ID")
    
    # Process GDD data with cancer type mapping
    val_cancertypes_gdd = val_cancertypes_gdd.rename(columns={'Cancer_Type': 'ground_truth'})
    ground_truth_cancers = val_cancertypes_gdd.ground_truth.unique()
    gdd_pred_cancers = val_cancertypes_gdd.Pred1.unique()
    
    # Merge with clinical data
    val_cancertypes_gdd = val_cancertypes_gdd.merge(
        data_clinical_sample[['SAMPLE_ID', 'PATIENT_ID', 'CANCER_TYPE_DETAILED', 'SEQ_ASSAY_ID']],
        on='SAMPLE_ID',
        how='left'
    )
    
    # Map predictions using mapping_df for Pred1
    val_cancertypes_gdd = val_cancertypes_gdd.merge(
        mapping_df,
        left_on=['CANCER_TYPE_DETAILED', 'Pred1'],
        right_on=['CANCER_TYPE_DETAILED', 'Cancer_Type'],
        how='left'
    )
    val_cancertypes_gdd = val_cancertypes_gdd.rename(columns={'CANCER_TYPE': 'GDD_Prediction1'})
    val_cancertypes_gdd['GDD_Prediction1'] = (
        val_cancertypes_gdd['GDD_Prediction1']
        .fillna(val_cancertypes_gdd['Pred1'].map(mapping))
        .fillna('Unknown')
    )
    
    # Map predictions using mapping_df for Pred2
    val_cancertypes_gdd = val_cancertypes_gdd.merge(
        mapping_df,
        left_on=['CANCER_TYPE_DETAILED', 'Pred2'],
        right_on=['CANCER_TYPE_DETAILED', 'Cancer_Type'],
        how='left'
    )
    val_cancertypes_gdd = val_cancertypes_gdd.rename(columns={'CANCER_TYPE': 'GDD_Prediction2'})
    val_cancertypes_gdd['GDD_Prediction2'] = (
        val_cancertypes_gdd['GDD_Prediction2']
        .fillna(val_cancertypes_gdd['Pred2'].map(mapping))
        .fillna('Unknown')
    )
    
    unknown_rows = val_cancertypes_gdd[val_cancertypes_gdd['GDD_Prediction1'] == "Unknown"]
    print(f"Number of Unknown rows in GDD predictions: {len(unknown_rows)}")
    
    # Process gpt-5 data with normalization
    val_cancertypes['prediction1'] = val_cancertypes['prediction1'].fillna('indetermined')
    val_cancertypes['prediction2'] = val_cancertypes['prediction2'].fillna('indetermined')
    val_cancertypes['prediction1'] = val_cancertypes['prediction1'].apply(normalize)
    val_cancertypes['prediction2'] = val_cancertypes['prediction2'].apply(normalize)
    
    # Fix specific cancer type naming inconsistencies
    val_cancertypes.loc[val_cancertypes['prediction1'] == 'Skin Cancer Non-Melanoma', 'prediction1'] = 'Skin Cancer, Non-Melanoma'
    val_cancertypes.loc[val_cancertypes['prediction1'] == 'Pancreatic cancer', 'prediction1'] = 'Pancreatic Cancer'
    val_cancertypes.loc[val_cancertypes['prediction2'] == 'Skin Cancer Non-Melanoma', 'prediction2'] = 'Skin Cancer, Non-Melanoma'
    val_cancertypes.loc[val_cancertypes['prediction2'] == 'Pancreatic cancer', 'prediction2'] = 'Pancreatic Cancer'
    
    # Merge all predictions
    joint_pred = val_cancertypes.merge(val_cancertypes_gdd, on='SAMPLE_ID', how='outer', suffixes=('', '_gdd'))
    joint_pred = joint_pred.rename(columns={
        'prediction1': 'gpt_5_Prediction1', 'prediction2': 'gpt_5_Prediction2',
        'prob1': 'gpt_5_Conf1', 'prob2': 'gpt_5_Conf2',
        'Conf1': 'GDD_Conf1', 'Conf2': 'GDD_Conf2'
    })
    
    joint_pred = joint_pred.merge(val_cancertypes_gemma, on='SAMPLE_ID', how='outer', suffixes=('', '_gemma'))
    joint_pred = joint_pred.rename(columns={
        'prediction1': 'medgemma_Prediction1', 'prediction2': 'medgemma_Prediction2',
        'prob1': 'medgemma_Conf1', 'prob2': 'medgemma_Conf2'
    })
    
    # Load assay information and add CENTER
    assay_df = pd.read_table("/data1/morrisq/yuj13/llm_genomics/genie_data_v17.0/assay_information.txt")
    assay_df_unique = assay_df[['SEQ_ASSAY_ID', 'CENTER']].drop_duplicates()
    seq_assay_to_center = dict(zip(assay_df_unique['SEQ_ASSAY_ID'], assay_df_unique['CENTER']))
    joint_pred['CENTER'] = joint_pred['SEQ_ASSAY_ID'].map(seq_assay_to_center)
    
    # Define columns to check for NaN
    cols_to_check = ['gpt_5_Prediction1', 'GDD_Prediction1', 'ground_truth', 'CENTER']
    joint_pred = joint_pred.dropna(subset=cols_to_check).reset_index(drop=True)
    
    # Print data quality summary
    print(f"Final dataset shape: {joint_pred.shape}")
    print(f"Unique ground truth cancer types: {joint_pred['ground_truth'].nunique()}")
    print(f"Unique GDD predictions: {joint_pred['GDD_Prediction1'].nunique()}")
    print(f"Unique gpt-5 predictions: {joint_pred['gpt_5_Prediction1'].nunique()}")
    
    return joint_pred

def prepare_features_and_labels(joint_pred):
    """Prepare features and labels for machine learning models."""
    # Prepare features
    features_all = joint_pred[[
        'gpt_5_Prediction1', 'gpt_5_Conf1',
        'GDD_Prediction1', 'GDD_Conf1',
        'gpt_5_Prediction2', 'gpt_5_Conf2',
        'GDD_Prediction2', 'GDD_Conf2'
    ]]
    labels = joint_pred['ground_truth']
    
    # Encode categorical predictions
    features_encoded_all = pd.get_dummies(features_all, columns=[
        'gpt_5_Prediction1', 'GDD_Prediction1', 'gpt_5_Prediction2', 'GDD_Prediction2',
        'gpt_5_Conf1', 'GDD_Conf1', 'gpt_5_Conf2', 'GDD_Conf2'
    ])
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    original_preds_encoded = label_encoder.transform(joint_pred['GDD_Prediction1'])
    
    # Handle gpt-5 predictions with potential unseen classes
    gpt_5_preds = joint_pred['gpt_5_Prediction1'].apply(
        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else len(label_encoder.classes_)
    )
    original_preds_encoded_llm = gpt_5_preds.values
    
    return features_encoded_all, labels_encoded, original_preds_encoded, original_preds_encoded_llm, label_encoder

def run_5_fold_cv(joint_pred, features, labels, original_gdd, original_gpt5, output_dir, fold=5):
    """Run 5-fold cross-validation split by PATIENT_ID and save results."""
    print(f"\n===== Running {fold}-Fold Cross-Validation (Split by PATIENT_ID) =====")

    # Get unique patient IDs
    unique_patients = joint_pred['PATIENT_ID'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_patients)

    # Split patients into folds
    fold_size = len(unique_patients) // fold
    patient_folds = [unique_patients[i*fold_size:(i+1)*fold_size] for i in range(fold-1)]
    patient_folds.append(unique_patients[(fold-1)*fold_size:])  # Last fold gets remaining patients

    results = {
        'fold': [],
        'GDD_accuracy': [], 'GDD_f1': [], 'GDD_precision': [], 'GDD_recall': [],
        'gpt_5_accuracy': [], 'gpt_5_f1': [], 'gpt_5_precision': [], 'gpt_5_recall': [],
        'LogReg_accuracy': [], 'LogReg_f1': [], 'LogReg_precision': [], 'LogReg_recall': [],
        'XGBoost_accuracy': [], 'XGBoost_f1': [], 'XGBoost_precision': [], 'XGBoost_recall': [],
        'RandomForest_accuracy': [], 'RandomForest_f1': [], 'RandomForest_precision': [], 'RandomForest_recall': []
    }

    print("\n===== Per-Fold Metrics =====")
    for i, test_patients in enumerate(patient_folds):
        # Create boolean masks based on patient IDs
        test_mask = joint_pred['PATIENT_ID'].isin(test_patients)
        train_mask = ~test_mask

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        print(f"\nFold {i+1}: {len(test_patients)} patients, {len(test_idx)} samples in test set")

        # Original model accuracies and metrics
        gdd_preds = original_gdd[test_idx]
        gpt5_preds = original_gpt5[test_idx]

        acc_gdd = accuracy_score(y_test, gdd_preds)
        f1_gdd = f1_score(y_test, gdd_preds, average='weighted', zero_division=0)
        prec_gdd = precision_score(y_test, gdd_preds, average='weighted', zero_division=0)
        rec_gdd = recall_score(y_test, gdd_preds, average='weighted', zero_division=0)

        acc_openai = accuracy_score(y_test, gpt5_preds)
        f1_openai = f1_score(y_test, gpt5_preds, average='weighted', zero_division=0)
        prec_openai = precision_score(y_test, gpt5_preds, average='weighted', zero_division=0)
        rec_openai = recall_score(y_test, gpt5_preds, average='weighted', zero_division=0)

        # Train and evaluate ensemble models
        log_model = LogisticRegression(max_iter=1000)
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
        results['GDD_accuracy'].append(acc_gdd)
        results['GDD_f1'].append(f1_gdd)
        results['GDD_precision'].append(prec_gdd)
        results['GDD_recall'].append(rec_gdd)

        results['gpt_5_accuracy'].append(acc_openai)
        results['gpt_5_f1'].append(f1_openai)
        results['gpt_5_precision'].append(prec_openai)
        results['gpt_5_recall'].append(rec_openai)

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
        print(f"  GDD: {acc_gdd:.4f}")
        print(f"  gpt-5: {acc_openai:.4f}")
        print(f"  LogReg: {acc_log:.4f}")
        print(f"  XGBoost: {acc_xgb:.4f}")
        print(f"  RF: {acc_rf:.4f}")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(results)
    cv_detailed_file = os.path.join(output_dir, f'{fold}_fold_cv_detailed_results.csv')
    results_df.to_csv(cv_detailed_file, index=False)
    print(f"{fold}-fold CV detailed results saved to: {cv_detailed_file}")

    # Calculate and save summary statistics
    methods = ['GDD', 'gpt_5', 'LogReg', 'XGBoost', 'RandomForest']
    method_names = ['GDD Original', 'gpt-5 Original', 'LogReg', 'XGBoost', 'Random Forest']

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
    # plt.show()
    
    # Print summary
    print("\n===== Cross-Validation Metrics Summary =====")
    for _, row in summary_df.iterrows():
        print(f"{row['Method']}:")
        print(f"  Accuracy: {row['Mean_Accuracy']:.4f} ± {row['Std_Accuracy']:.4f}")
        print(f"  F1 Score: {row['Mean_F1']:.4f} ± {row['Std_F1']:.4f}")
        print(f"  Precision: {row['Mean_Precision']:.4f} ± {row['Std_Precision']:.4f}")
        print(f"  Recall: {row['Mean_Recall']:.4f} ± {row['Std_Recall']:.4f}")
    
    return results_df, summary_df

def run_leave_one_center_out(joint_pred, features, labels, original_gdd, original_gpt5, label_encoder, output_dir):
    """Run leave-one-center-out cross-validation and save results."""
    print("\n===== Running Leave-One-CENTER-Out Cross-Validation =====")
    
    centers = joint_pred['CENTER']
    unique_centers = centers.unique()
    
    results = {
        'CENTER': [],
        'num_test_samples': [],
        'GDD_accuracy': [], 'GDD_f1': [], 'GDD_precision': [], 'GDD_recall': [],
        'gpt_5_accuracy': [], 'gpt_5_f1': [], 'gpt_5_precision': [], 'gpt_5_recall': [],
        'LogReg_accuracy': [], 'LogReg_f1': [], 'LogReg_precision': [], 'LogReg_recall': [],
        'XGBoost_accuracy': [], 'XGBoost_f1': [], 'XGBoost_precision': [], 'XGBoost_recall': [],
        'RandomForest_accuracy': [], 'RandomForest_f1': [], 'RandomForest_precision': [], 'RandomForest_recall': []
    }
    
    # Get all unique classes to ensure consistency across folds
    all_classes = np.unique(labels)
    n_classes = len(all_classes)
    
    for center in unique_centers:
        print(f"\n===== Leave-One-CENTER-Out: {center} =====")
        train_mask = centers != center
        test_mask = centers == center
        
        X_train = features[train_mask]
        X_test = features[test_mask]
        y_train = labels[train_mask]
        y_test = labels[test_mask]
        original_preds_gdd_test = original_gdd[test_mask]
        original_preds_openai_test = original_gpt5[test_mask]
        
        if len(y_test) == 0:
            print("No test samples for this center. Skipping.")
            continue
        
        # Check if training set has all classes, if not, skip problematic models
        train_classes = np.unique(y_train)
        test_classes = np.unique(y_test)
        
        print(f"Training classes: {len(train_classes)}, Test classes: {len(test_classes)}")
        
        # Calculate metrics for original models
        acc_gdd = accuracy_score(y_test, original_preds_gdd_test)
        f1_gdd = f1_score(y_test, original_preds_gdd_test, average='weighted', zero_division=0)
        prec_gdd = precision_score(y_test, original_preds_gdd_test, average='weighted', zero_division=0)
        rec_gdd = recall_score(y_test, original_preds_gdd_test, average='weighted', zero_division=0)

        acc_openai = accuracy_score(y_test, original_preds_openai_test)
        f1_openai = f1_score(y_test, original_preds_openai_test, average='weighted', zero_division=0)
        prec_openai = precision_score(y_test, original_preds_openai_test, average='weighted', zero_division=0)
        rec_openai = recall_score(y_test, original_preds_openai_test, average='weighted', zero_division=0)

        # Initialize metric variables
        acc_log = f1_log = prec_log = rec_log = 0.0
        acc_xgb = f1_xgb = prec_xgb = rec_xgb = 0.0
        acc_rf = f1_rf = prec_rf = rec_rf = 0.0
        
        try:
            # Logistic Regression - more robust to missing classes
            log_reg = LogisticRegression(max_iter=1000, random_state=42)
            log_reg.fit(X_train, y_train)
            log_preds = log_reg.predict(X_test)
            acc_log = accuracy_score(y_test, log_preds)
            f1_log = f1_score(y_test, log_preds, average='weighted', zero_division=0)
            prec_log = precision_score(y_test, log_preds, average='weighted', zero_division=0)
            rec_log = recall_score(y_test, log_preds, average='weighted', zero_division=0)
        except Exception as e:
            print(f"LogReg failed for center {center}: {e}")
            acc_log = f1_log = prec_log = rec_log = 0.0
        
        try:
            # XGBoost with relabeled classes to ensure consecutiveness
            # Create mapping for consecutive labels
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
                print(f"No valid test samples for XGBoost in center {center}")
                acc_xgb = f1_xgb = prec_xgb = rec_xgb = 0.0

        except Exception as e:
            print(f"XGBoost failed for center {center}: {e}")
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
            print(f"RandomForest failed for center {center}: {e}")
            acc_rf = f1_rf = prec_rf = rec_rf = 0.0
        
        # Store results
        results['CENTER'].append(center)
        results['num_test_samples'].append(len(y_test))

        results['GDD_accuracy'].append(acc_gdd)
        results['GDD_f1'].append(f1_gdd)
        results['GDD_precision'].append(prec_gdd)
        results['GDD_recall'].append(rec_gdd)

        results['gpt_5_accuracy'].append(acc_openai)
        results['gpt_5_f1'].append(f1_openai)
        results['gpt_5_precision'].append(prec_openai)
        results['gpt_5_recall'].append(rec_openai)

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
        
        print(f"Test samples: {len(y_test)}")
        print(f"GDD accuracy: {acc_gdd:.4f}")
        print(f"gpt-5 accuracy: {acc_openai:.4f}")
        print(f"LogReg accuracy: {acc_log:.4f}")
        print(f"XGBoost accuracy: {acc_xgb:.4f}")
        print(f"RF accuracy: {acc_rf:.4f}")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(results)
    loco_detailed_file = os.path.join(output_dir, 'leave_one_center_out_detailed_results.csv')
    results_df.to_csv(loco_detailed_file, index=False)
    print(f"Leave-one-center-out detailed results saved to: {loco_detailed_file}")
    
    # Calculate and save summary statistics (excluding failed runs with 0.0 accuracy)
    def safe_stats(series):
        """Calculate stats excluding 0.0 values (failed runs)"""
        non_zero = series[series > 0.0]
        if len(non_zero) == 0:
            return 0.0, 0.0, 0.0, 0.0
        return non_zero.mean(), non_zero.std(), non_zero.min(), non_zero.max()
    
    summary_data = {
        'Method': ['GDD Original', 'gpt-5 Original', 'LogReg', 'XGBoost', 'Random Forest'],
        'Mean_Accuracy': [], 'Std_Accuracy': [],
        'Mean_F1': [], 'Std_F1': [],
        'Mean_Precision': [], 'Std_Precision': [],
        'Mean_Recall': [], 'Std_Recall': [],
        'Successful_Runs': []
    }

    for method_prefix in ['GDD', 'gpt_5', 'LogReg', 'XGBoost', 'RandomForest']:
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
        successful_runs = len(results_df[results_df[f'{method_prefix}_accuracy'] > 0.0]) if method_prefix not in ['GDD', 'gpt_5'] else len(results_df)
        summary_data['Successful_Runs'].append(successful_runs)
    
    summary_df = pd.DataFrame(summary_data)
    loco_summary_file = os.path.join(output_dir, 'leave_one_center_out_summary.csv')
    summary_df.to_csv(loco_summary_file, index=False)
    print(f"Leave-one-center-out summary saved to: {loco_summary_file}")
    
    # Create and save plot (only for centers where all methods succeeded)
    plt.figure(figsize=(16, 8))
    
    # Filter results to only include successful runs for plotting
    plot_results = results_df.copy()
    centers_sorted = sorted(plot_results['CENTER'])
    x = np.arange(len(centers_sorted))
    width = 0.15
    
    methods = ['GDD_accuracy', 'gpt_5_accuracy', 'LogReg_accuracy', 'XGBoost_accuracy', 'RandomForest_accuracy']
    method_names = ['GDD Original', 'gpt-5 Original', 'Logistic Regression', 'XGBoost', 'Random Forest']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        accuracies = [plot_results[plot_results['CENTER'] == c][method].iloc[0] for c in centers_sorted]
        plt.bar(x + i*width - 2*width, accuracies, width=width, label=name, color=color)
    
    plt.ylabel("Accuracy")
    plt.xlabel("CENTER")
    plt.title("Leave-One-CENTER-Out Accuracy per Model")
    plt.xticks(x, centers_sorted, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'leave_one_center_out_accuracy_plot.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    
    # Print summary
    print("\n===== Leave-One-CENTER-Out Metrics Summary =====")
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

def main(fold=5):
    """Main function to run the complete analysis."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/data1/morrisq/yuj13/llm_genomics/genie_meta_model_result/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Results will be saved to: {output_dir}")

    # Load and prepare data
    print("Loading and preparing data...")
    joint_pred = load_and_prepare_data()
    features, labels, original_gdd, original_gpt5, label_encoder = prepare_features_and_labels(joint_pred)

    print(f"Dataset shape: {joint_pred.shape}")
    print(f"Number of unique centers: {joint_pred['CENTER'].nunique()}")
    print(f"Number of unique patients: {joint_pred['PATIENT_ID'].nunique()}")
    print(f"Number of unique cancer types: {joint_pred['ground_truth'].nunique()}")

    # Run n-fold cross-validation (split by PATIENT_ID)
    cv_results, cv_summary = run_5_fold_cv(joint_pred, features, labels, original_gdd, original_gpt5, output_dir, fold=fold)
    
    # Run leave-one-center-out cross-validation
    loco_results, loco_summary = run_leave_one_center_out(
        joint_pred, features, labels, original_gdd, original_gpt5, label_encoder, output_dir
    )
    
    # Save overall summary
    overall_summary = {
        'Evaluation_Method': [f'{fold}-Fold CV', 'Leave-One-Center-Out'],
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
    import argparse
    parser = argparse.ArgumentParser(description='Run meta-model analysis on cancer type predictions')
    parser.add_argument('--fold', type=int, default=5, help='Number of folds for cross-validation (default: 5)')
    args = parser.parse_args()
    main(fold=args.fold)