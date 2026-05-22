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
import random

SEED = 42
TOP_N_FEATURES = 5  # Number of top features to save per sample

def set_seed(seed=SEED):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_sample_feature_importance(model, X_sample, feature_names, model_type, predicted_class=None):
    """
    Compute per-sample feature importance/contributions.

    For LogReg: contribution = feature_value * coefficient[predicted_class]
    For XGBoost/RF: contribution = feature_value * global_feature_importance

    Returns dict with top N features and their contributions.
    """
    X_values = X_sample.values if hasattr(X_sample, 'values') else X_sample

    if model_type == 'LogReg':
        # For logistic regression, use coefficients for the predicted class
        if len(model.classes_) == 2:
            coefs = model.coef_[0]
        else:
            # Convert class label to index in model.classes_
            class_idx = np.where(model.classes_ == predicted_class)[0][0]
            coefs = model.coef_[class_idx]
        contributions = X_values * coefs
    else:
        # For tree-based models, use global feature importance weighted by feature value
        importance = model.feature_importances_
        contributions = X_values * importance

    # Get indices of top N features by absolute contribution
    top_indices = np.argsort(np.abs(contributions))[::-1][:TOP_N_FEATURES]

    result = {}
    for rank, idx in enumerate(top_indices, 1):
        result[f'top{rank}_feature'] = feature_names[idx]
        result[f'top{rank}_contribution'] = contributions[idx]

    return result

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


def create_cancer_type_mappings(mapping_df):
    """Create mappings between dot notation and human-readable cancer type names."""
    # Use keep='first' to get the most common/correct mapping for each cancer type
    mapping_df_first = mapping_df[['CANCER_TYPE', 'Cancer_Type']].drop_duplicates(subset='Cancer_Type', keep='first')
    # dot_to_readable: "Colorectal.Cancer" -> "Colorectal Cancer"
    dot_to_readable = dict(zip(mapping_df_first['Cancer_Type'], mapping_df_first['CANCER_TYPE']))
    # readable_to_dot: "Colorectal Cancer" -> "Colorectal.Cancer"
    # For this direction, also use keep='first' on CANCER_TYPE
    mapping_df_readable_first = mapping_df[['CANCER_TYPE', 'Cancer_Type']].drop_duplicates(subset='CANCER_TYPE', keep='first')
    readable_to_dot = dict(zip(mapping_df_readable_first['CANCER_TYPE'], mapping_df_readable_first['Cancer_Type']))
    return dot_to_readable, readable_to_dot

def load_model_data(folder_path):
    """Load all CSV files from folder and subfolders."""
    files = glob.glob(os.path.join(folder_path, "**/*.csv"), recursive=True)
    if not files:
        raise ValueError(f"No CSV files found in {folder_path}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return df.drop_duplicates(subset="SAMPLE_ID")

def standardize_columns(df, model_name, mapping_df=None):
    """Auto-detect and standardize column names based on format."""
    df = df.copy()

    # GDD original format: Pred1, Pred2, Conf1, Conf2, Cancer_Type
    if 'Pred1' in df.columns:
        df = df.rename(columns={
            'Pred1': f'{model_name}_Prediction1',
            'Pred2': f'{model_name}_Prediction2',
            'Conf1': f'{model_name}_Conf1',
            'Conf2': f'{model_name}_Conf2',
            'Cancer_Type': 'ground_truth'
        })

    # GDD-ENS-retrained format: target_name + probability columns for each cancer type
    elif 'target_name' in df.columns and 'inference_name' in df.columns:
        # Get the list of cancer type probability columns (exclude known non-cancer columns)
        non_prob_cols = {'SAMPLE_ID', 'Classification_Category', 'target_name', 'inference_name', 'Unnamed: 0'}
        prob_cols = [c for c in df.columns if c not in non_prob_cols]

        if prob_cols:
            # Load mapping to convert dot notation to human-readable
            # Use drop_duplicates with keep='first' to get the most common/correct mapping
            # (e.g., Colorectal.Cancer -> Colorectal Cancer, not Gastrointestinal Neuroendocrine Tumor)
            mapping_file = "/data1/morrisq/yuj13/llm_genomics/GDD_ENS/data/tumor_type_final.txt"
            mapping_df_local = pd.read_table(mapping_file)
            mapping_df_first = mapping_df_local[['CANCER_TYPE', 'Cancer_Type']].drop_duplicates(subset='Cancer_Type', keep='first')
            dot_to_readable = dict(zip(mapping_df_first['Cancer_Type'], mapping_df_first['CANCER_TYPE']))

            # Extract top 2 predictions from probability columns
            # Fill NaN with 0 to handle cases where COLU and DFCI have different columns after concat
            prob_df = df[prob_cols].fillna(0)

            # Get top 2 predictions for each row
            top2_indices = prob_df.values.argsort(axis=1)[:, -2:][:, ::-1]  # Sort descending, get top 2

            # Extract prediction names (in dot notation) and confidence values
            pred1_dot = [prob_cols[i] for i in top2_indices[:, 0]]
            pred2_dot = [prob_cols[i] for i in top2_indices[:, 1]]
            conf1 = [prob_df.iloc[row, top2_indices[row, 0]] for row in range(len(df))]
            conf2 = [prob_df.iloc[row, top2_indices[row, 1]] for row in range(len(df))]

            # Convert predictions from dot notation to human-readable format
            pred1 = [dot_to_readable.get(p, p) for p in pred1_dot]
            pred2 = [dot_to_readable.get(p, p) for p in pred2_dot]

            df[f'{model_name}_Prediction1'] = pred1
            df[f'{model_name}_Prediction2'] = pred2
            df[f'{model_name}_Conf1'] = conf1
            df[f'{model_name}_Conf2'] = conf2

            # Convert target_name from dot notation to human-readable for ground_truth
            df['ground_truth'] = df['target_name'].map(lambda x: dot_to_readable.get(x, x))
        else:
            print(f"Warning: No probability columns found for {model_name}")

    # GPT-5 format: prediction1, prediction2, prob1, prob2
    elif 'prediction1' in df.columns:
        # Apply normalization for GPT-5 style predictions
        df['prediction1'] = df['prediction1'].fillna('indetermined').apply(normalize)
        df['prediction2'] = df['prediction2'].fillna('indetermined').apply(normalize)
        # Fix specific cancer type naming inconsistencies
        df.loc[df['prediction1'] == 'Skin Cancer Non-Melanoma', 'prediction1'] = 'Skin Cancer, Non-Melanoma'
        df.loc[df['prediction1'] == 'Pancreatic cancer', 'prediction1'] = 'Pancreatic Cancer'
        df.loc[df['prediction2'] == 'Skin Cancer Non-Melanoma', 'prediction2'] = 'Skin Cancer, Non-Melanoma'
        df.loc[df['prediction2'] == 'Pancreatic cancer', 'prediction2'] = 'Pancreatic Cancer'
        df = df.rename(columns={
            'prediction1': f'{model_name}_Prediction1',
            'prediction2': f'{model_name}_Prediction2',
            'prob1': f'{model_name}_Conf1',
            'prob2': f'{model_name}_Conf2'
        })

    return df

def load_and_prepare_data(
    model1_path="/data1/morrisq/yuj13/llm_genomics/genie_output/GDD",
    model2_path="/data1/morrisq/yuj13/llm_genomics/genie_output/gpt-5",
    model1_name="GDD",
    model2_name="gpt_5"
):
    """Load and prepare the genomic data for analysis."""
    print(f"Loading model1 ({model1_name}) from: {model1_path}")
    print(f"Loading model2 ({model2_name}) from: {model2_path}")

    # Load model data recursively from folders
    model1_data = load_model_data(model1_path)
    model2_data = load_model_data(model2_path)
    print(f"Loaded {len(model1_data)} samples from {model1_name}, {len(model2_data)} samples from {model2_name}")

    # Load clinical data and mapping
    # Use drop_duplicates with keep='first' to get the most common/correct mapping
    data_clinical_sample = pd.read_csv('/data1/morrisq/yuj13/llm_genomics/genie_data_v17.0/data_clinical_sample.txt', sep='\t', comment='#')
    mapping_df = pd.read_table("/data1/morrisq/yuj13/llm_genomics/GDD_ENS/data/tumor_type_final.txt")
    mapping_df_first = mapping_df[['CANCER_TYPE', 'Cancer_Type']].drop_duplicates(subset='Cancer_Type', keep='first')
    mapping = dict(zip(mapping_df_first['Cancer_Type'], mapping_df_first['CANCER_TYPE']))

    # Create bidirectional mappings
    dot_to_readable, readable_to_dot = create_cancer_type_mappings(mapping_df)

    # Detect format BEFORE standardization
    # Original GDD format has 'Pred1' column (needs fine-to-coarse mapping)
    # GDD-ENS-retrained has 'target_name' and 'inference_name' (already coarse-grained)
    # GPT-5 format has 'prediction1' column
    model1_is_original_gdd = 'Pred1' in model1_data.columns
    model2_is_original_gdd = 'Pred1' in model2_data.columns

    # Check if original GDD predictions are already coarse-grained BEFORE any conversion
    # If predictions are in the coarse Cancer_Type set, they don't need fine-to-coarse mapping
    coarse_types = set(mapping_df['Cancer_Type'].unique())
    model1_is_coarse = False
    model2_is_coarse = False
    if model1_is_original_gdd and 'Pred1' in model1_data.columns:
        sample_preds = model1_data['Pred1'].dropna().unique()[:50]
        model1_is_coarse = all(p in coarse_types for p in sample_preds)
    if model2_is_original_gdd and 'Pred1' in model2_data.columns:
        sample_preds = model2_data['Pred1'].dropna().unique()[:50]
        model2_is_coarse = all(p in coarse_types for p in sample_preds)

    # Standardize columns for both models
    model1_data = standardize_columns(model1_data, model1_name, mapping_df)
    model2_data = standardize_columns(model2_data, model2_name, mapping_df)

    # Convert dot notation to human-readable format for GDD-ENS-retrained style data
    def convert_to_readable(df, model_name, dot_to_readable):
        """Convert dot notation predictions to human-readable format."""
        pred1_col = f'{model_name}_Prediction1'
        pred2_col = f'{model_name}_Prediction2'

        if pred1_col in df.columns:
            # Check if values are in dot notation (contain dots but not spaces)
            sample_val = df[pred1_col].dropna().iloc[0] if len(df[pred1_col].dropna()) > 0 else ""
            if '.' in str(sample_val) and ' ' not in str(sample_val):
                df[pred1_col] = df[pred1_col].map(lambda x: dot_to_readable.get(x, x))
                df[pred2_col] = df[pred2_col].map(lambda x: dot_to_readable.get(x, x))

        if 'ground_truth' in df.columns:
            sample_gt = df['ground_truth'].dropna().iloc[0] if len(df['ground_truth'].dropna()) > 0 else ""
            if '.' in str(sample_gt) and ' ' not in str(sample_gt):
                df['ground_truth'] = df['ground_truth'].map(lambda x: dot_to_readable.get(x, x))

        return df

    model1_data = convert_to_readable(model1_data, model1_name, dot_to_readable)
    model2_data = convert_to_readable(model2_data, model2_name, dot_to_readable)

    # For GDD-format models, apply cancer type mapping to convert fine-grained to coarse-grained
    def apply_gdd_mapping(df, model_name, mapping_df, mapping):
        pred1_col = f'{model_name}_Prediction1'
        pred2_col = f'{model_name}_Prediction2'
        if pred1_col not in df.columns:
            return df
        # Merge with clinical data first to get CANCER_TYPE_DETAILED
        df = df.merge(
            data_clinical_sample[['SAMPLE_ID', 'PATIENT_ID', 'CANCER_TYPE_DETAILED', 'SEQ_ASSAY_ID']],
            on='SAMPLE_ID',
            how='left'
        )
        # Store original predictions
        orig_pred1 = df[pred1_col].copy()
        orig_pred2 = df[pred2_col].copy()
        # Map Pred1
        df = df.merge(
            mapping_df,
            left_on=['CANCER_TYPE_DETAILED', pred1_col],
            right_on=['CANCER_TYPE_DETAILED', 'Cancer_Type'],
            how='left'
        )
        df[pred1_col] = df['CANCER_TYPE'].fillna(orig_pred1.map(mapping)).fillna('Unknown')
        df = df.drop(columns=['CANCER_TYPE', 'Cancer_Type'], errors='ignore')
        # Map Pred2
        df = df.merge(
            mapping_df,
            left_on=['CANCER_TYPE_DETAILED', pred2_col],
            right_on=['CANCER_TYPE_DETAILED', 'Cancer_Type'],
            how='left'
        )
        df[pred2_col] = df['CANCER_TYPE'].fillna(orig_pred2.map(mapping)).fillna('Unknown')
        df = df.drop(columns=['CANCER_TYPE', 'Cancer_Type'], errors='ignore')
        return df

    # Apply GDD mapping only for original fine-grained GDD format (not for coarse-grained or GPT-5)
    # Skip if predictions are already coarse-grained (detected earlier before conversion)
    if model1_is_original_gdd and not model1_is_coarse:
        model1_data = apply_gdd_mapping(model1_data, model1_name, mapping_df, mapping)
    if model2_is_original_gdd and not model2_is_coarse:
        model2_data = apply_gdd_mapping(model2_data, model2_name, mapping_df, mapping)

    # Merge model predictions
    joint_pred = model1_data.merge(model2_data, on='SAMPLE_ID', how='outer', suffixes=('', '_m2'))

    # Handle duplicate ground_truth columns from merge
    if 'ground_truth_m2' in joint_pred.columns:
        # Use model1's ground_truth, fill missing with model2's
        joint_pred['ground_truth'] = joint_pred['ground_truth'].fillna(joint_pred['ground_truth_m2'])
        joint_pred = joint_pred.drop(columns=['ground_truth_m2'])

    # Always merge with clinical data to get PATIENT_ID and SEQ_ASSAY_ID
    # (some models don't have these from apply_gdd_mapping)
    if 'PATIENT_ID' not in joint_pred.columns or 'SEQ_ASSAY_ID' not in joint_pred.columns:
        joint_pred = joint_pred.merge(
            data_clinical_sample[['SAMPLE_ID', 'PATIENT_ID', 'CANCER_TYPE_DETAILED', 'SEQ_ASSAY_ID']],
            on='SAMPLE_ID',
            how='left'
        )

    # Load assay information and add CENTER
    assay_df = pd.read_table("/data1/morrisq/yuj13/llm_genomics/genie_data_v17.0/assay_information.txt")
    assay_df_unique = assay_df[['SEQ_ASSAY_ID', 'CENTER']].drop_duplicates()
    seq_assay_to_center = dict(zip(assay_df_unique['SEQ_ASSAY_ID'], assay_df_unique['CENTER']))
    joint_pred['CENTER'] = joint_pred['SEQ_ASSAY_ID'].map(seq_assay_to_center)

    # Define columns to check for NaN
    pred1_col_m1 = f'{model1_name}_Prediction1'
    pred1_col_m2 = f'{model2_name}_Prediction1'
    cols_to_check = [pred1_col_m1, pred1_col_m2, 'ground_truth', 'CENTER']
    joint_pred = joint_pred.dropna(subset=cols_to_check).reset_index(drop=True)

    # Print data quality summary
    print(f"Final dataset shape: {joint_pred.shape}")
    print(f"Unique ground truth cancer types: {joint_pred['ground_truth'].nunique()}")
    print(f"Unique {model1_name} predictions: {joint_pred[pred1_col_m1].nunique()}")
    print(f"Unique {model2_name} predictions: {joint_pred[pred1_col_m2].nunique()}")

    return joint_pred, model1_name, model2_name

def prepare_features_and_labels(joint_pred, model1_name="GDD", model2_name="gpt_5"):
    """Prepare features and labels for machine learning models."""
    # Build column names dynamically
    feature_cols = [
        f'{model1_name}_Prediction1', f'{model1_name}_Conf1',
        f'{model2_name}_Prediction1', f'{model2_name}_Conf1',
        f'{model1_name}_Prediction2', f'{model1_name}_Conf2',
        f'{model2_name}_Prediction2', f'{model2_name}_Conf2'
    ]
    # Filter to only columns that exist
    feature_cols = [c for c in feature_cols if c in joint_pred.columns]

    features_all = joint_pred[feature_cols]
    labels = joint_pred['ground_truth']

    # Encode categorical predictions
    features_encoded_all = pd.get_dummies(features_all, columns=feature_cols)

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Encode model1 predictions
    model1_pred_col = f'{model1_name}_Prediction1'
    original_preds_model1 = joint_pred[model1_pred_col].apply(
        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else len(label_encoder.classes_)
    ).values

    # Encode model2 predictions
    model2_pred_col = f'{model2_name}_Prediction1'
    original_preds_model2 = joint_pred[model2_pred_col].apply(
        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else len(label_encoder.classes_)
    ).values

    return features_encoded_all, labels_encoded, original_preds_model1, original_preds_model2, label_encoder

def run_5_fold_cv(joint_pred, features, labels, original_model1, original_model2, label_encoder, output_dir, fold=5, model1_name="GDD", model2_name="gpt_5", save_test_data=False):
    """Run 5-fold cross-validation split by PATIENT_ID and save results."""
    print(f"\n===== Running {fold}-Fold Cross-Validation (Split by PATIENT_ID) =====")

    # Get unique patient IDs
    unique_patients = joint_pred['PATIENT_ID'].unique()
    np.random.seed(SEED)
    np.random.shuffle(unique_patients)

    # Split patients into folds
    fold_size = len(unique_patients) // fold
    patient_folds = [unique_patients[i*fold_size:(i+1)*fold_size] for i in range(fold-1)]
    patient_folds.append(unique_patients[(fold-1)*fold_size:])  # Last fold gets remaining patients

    # Initialize test data collection if save_test_data is True
    test_data_records = {
        'LogReg': [],
        'XGBoost': [],
        'RandomForest': []
    }

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
        test_mask = joint_pred['PATIENT_ID'].isin(test_patients)
        train_mask = ~test_mask

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        print(f"\nFold {i+1}: {len(test_patients)} patients, {len(test_idx)} samples in test set")

        # Original model accuracies and metrics
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

        # Train and evaluate ensemble models
        log_model = LogisticRegression(max_iter=1000)
        log_model.fit(X_train, y_train)
        log_preds = log_model.predict(X_test)
        log_proba = log_model.predict_proba(X_test)
        acc_log = accuracy_score(y_test, log_preds)
        f1_log = f1_score(y_test, log_preds, average='weighted', zero_division=0)
        prec_log = precision_score(y_test, log_preds, average='weighted', zero_division=0)
        rec_log = recall_score(y_test, log_preds, average='weighted', zero_division=0)

        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=SEED)
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        xgb_proba = xgb_model.predict_proba(X_test)
        acc_xgb = accuracy_score(y_test, xgb_preds)
        f1_xgb = f1_score(y_test, xgb_preds, average='weighted', zero_division=0)
        prec_xgb = precision_score(y_test, xgb_preds, average='weighted', zero_division=0)
        rec_xgb = recall_score(y_test, xgb_preds, average='weighted', zero_division=0)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=SEED)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        rf_proba = rf_model.predict_proba(X_test)
        acc_rf = accuracy_score(y_test, rf_preds)
        f1_rf = f1_score(y_test, rf_preds, average='weighted', zero_division=0)
        prec_rf = precision_score(y_test, rf_preds, average='weighted', zero_division=0)
        rec_rf = recall_score(y_test, rf_preds, average='weighted', zero_division=0)

        # Collect test data if save_test_data is True
        if save_test_data:
            test_samples = joint_pred.iloc[test_idx]
            feature_names = X_test.columns.tolist()
            for j, idx in enumerate(test_idx):
                base_record = {
                    'SAMPLE_ID': test_samples.iloc[j]['SAMPLE_ID'],
                    'PATIENT_ID': test_samples.iloc[j]['PATIENT_ID'],
                    'ground_truth': test_samples.iloc[j]['ground_truth'],
                    'fold': i + 1
                }
                # LogReg
                log_importance = compute_sample_feature_importance(
                    log_model, X_test.iloc[j], feature_names, 'LogReg', predicted_class=log_preds[j]
                )
                test_data_records['LogReg'].append({
                    **base_record,
                    'ensemble_prediction': label_encoder.inverse_transform([log_preds[j]])[0],
                    'prob_ensemble_prediction': log_proba[j].max(),
                    **log_importance
                })
                # XGBoost
                xgb_importance = compute_sample_feature_importance(
                    xgb_model, X_test.iloc[j], feature_names, 'XGBoost'
                )
                test_data_records['XGBoost'].append({
                    **base_record,
                    'ensemble_prediction': label_encoder.inverse_transform([xgb_preds[j]])[0],
                    'prob_ensemble_prediction': xgb_proba[j].max(),
                    **xgb_importance
                })
                # RandomForest
                rf_importance = compute_sample_feature_importance(
                    rf_model, X_test.iloc[j], feature_names, 'RandomForest'
                )
                test_data_records['RandomForest'].append({
                    **base_record,
                    'ensemble_prediction': label_encoder.inverse_transform([rf_preds[j]])[0],
                    'prob_ensemble_prediction': rf_proba[j].max(),
                    **rf_importance
                })

        # Store results
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

        print(f"Fold {i+1}:")
        print(f"  {model1_name}: {acc_m1:.4f}")
        print(f"  {model2_name}: {acc_m2:.4f}")
        print(f"  LogReg: {acc_log:.4f}")
        print(f"  XGBoost: {acc_xgb:.4f}")
        print(f"  RF: {acc_rf:.4f}")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(results)
    cv_detailed_file = os.path.join(output_dir, f'{fold}_fold_cv_detailed_results.csv')
    results_df.to_csv(cv_detailed_file, index=False)
    print(f"{fold}-fold CV detailed results saved to: {cv_detailed_file}")

    # Calculate and save summary statistics
    methods = [model1_name, model2_name, 'LogReg', 'XGBoost', 'RandomForest']
    method_names = [f'{model1_name} Original', f'{model2_name} Original', 'LogReg', 'XGBoost', 'Random Forest']

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

    # Save test data if requested
    if save_test_data:
        test_data_dir = os.path.join(output_dir, 'test_data')
        os.makedirs(test_data_dir, exist_ok=True)
        for model_name, records in test_data_records.items():
            if records:
                test_data_df = pd.DataFrame(records)
                test_data_file = os.path.join(test_data_dir, f'{fold}_fold_cv_{model_name}_test_data.csv')
                test_data_df.to_csv(test_data_file, index=False)
                print(f"Test data saved to: {test_data_file}")

    return results_df, summary_df

def run_leave_one_center_out(joint_pred, features, labels, original_model1, original_model2, label_encoder, output_dir, model1_name="GDD", model2_name="gpt_5", save_test_data=False):
    """Run leave-one-center-out cross-validation and save results."""
    print("\n===== Running Leave-One-CENTER-Out Cross-Validation =====")

    centers = joint_pred['CENTER']
    unique_centers = centers.unique()

    # Check if we have enough centers
    if len(unique_centers) < 2:
        print(f"Cannot perform leave-one-center-out CV: only {len(unique_centers)} center(s) found. Need at least 2.")
        return None, None

    # Initialize test data collection if save_test_data is True
    test_data_records = {
        'LogReg': [],
        'XGBoost': [],
        'RandomForest': []
    }

    results = {
        'CENTER': [],
        'num_test_samples': [],
        f'{model1_name}_accuracy': [], f'{model1_name}_f1': [], f'{model1_name}_precision': [], f'{model1_name}_recall': [],
        f'{model2_name}_accuracy': [], f'{model2_name}_f1': [], f'{model2_name}_precision': [], f'{model2_name}_recall': [],
        'LogReg_accuracy': [], 'LogReg_f1': [], 'LogReg_precision': [], 'LogReg_recall': [],
        'XGBoost_accuracy': [], 'XGBoost_f1': [], 'XGBoost_precision': [], 'XGBoost_recall': [],
        'RandomForest_accuracy': [], 'RandomForest_f1': [], 'RandomForest_precision': [], 'RandomForest_recall': []
    }

    for center in unique_centers:
        print(f"\n===== Leave-One-CENTER-Out: {center} =====")
        train_mask = centers != center
        test_mask = centers == center

        X_train = features[train_mask]
        X_test = features[test_mask]
        y_train = labels[train_mask]
        y_test = labels[test_mask]
        original_preds_m1_test = original_model1[test_mask]
        original_preds_m2_test = original_model2[test_mask]

        if len(y_test) == 0:
            print("No test samples for this center. Skipping.")
            continue

        # Check if training set has all classes, if not, skip problematic models
        train_classes = np.unique(y_train)
        test_classes = np.unique(y_test)

        print(f"Training classes: {len(train_classes)}, Test classes: {len(test_classes)}")

        # Calculate metrics for original models
        acc_m1 = accuracy_score(y_test, original_preds_m1_test)
        f1_m1 = f1_score(y_test, original_preds_m1_test, average='weighted', zero_division=0)
        prec_m1 = precision_score(y_test, original_preds_m1_test, average='weighted', zero_division=0)
        rec_m1 = recall_score(y_test, original_preds_m1_test, average='weighted', zero_division=0)

        acc_m2 = accuracy_score(y_test, original_preds_m2_test)
        f1_m2 = f1_score(y_test, original_preds_m2_test, average='weighted', zero_division=0)
        prec_m2 = precision_score(y_test, original_preds_m2_test, average='weighted', zero_division=0)
        rec_m2 = recall_score(y_test, original_preds_m2_test, average='weighted', zero_division=0)

        # Initialize metric variables
        acc_log = f1_log = prec_log = rec_log = 0.0
        acc_xgb = f1_xgb = prec_xgb = rec_xgb = 0.0
        acc_rf = f1_rf = prec_rf = rec_rf = 0.0
        
        log_preds = None
        log_proba = None
        try:
            # Logistic Regression - more robust to missing classes
            log_reg = LogisticRegression(max_iter=1000, random_state=SEED)
            log_reg.fit(X_train, y_train)
            log_preds = log_reg.predict(X_test)
            log_proba = log_reg.predict_proba(X_test)
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
                    random_state=SEED,
                    num_class=len(train_classes_sorted),
                    objective='multi:softprob' if len(train_classes_sorted) > 2 else 'binary:logistic'
                )
                xgb_model.fit(X_train, y_train_relabeled)

                # Predict and map back to original labels
                preds_relabeled = xgb_model.predict(X_test_valid)
                preds_proba = xgb_model.predict_proba(X_test_valid)
                preds_original = np.array([reverse_mapping[pred] for pred in preds_relabeled])

                # Calculate metrics only for valid test samples
                y_test_valid = y_test[test_mask_valid]
                acc_xgb = accuracy_score(y_test_valid, preds_original)
                f1_xgb = f1_score(y_test_valid, preds_original, average='weighted', zero_division=0)
                prec_xgb = precision_score(y_test_valid, preds_original, average='weighted', zero_division=0)
                rec_xgb = recall_score(y_test_valid, preds_original, average='weighted', zero_division=0)

                # Collect XGBoost test data if save_test_data is True
                if save_test_data:
                    test_samples = joint_pred[test_mask].iloc[test_mask_valid]
                    feature_names = X_test_valid.columns.tolist() if hasattr(X_test_valid, 'columns') else [f'feature_{k}' for k in range(X_test_valid.shape[1])]
                    for j in range(len(test_mask_valid)):
                        xgb_importance = compute_sample_feature_importance(
                            xgb_model, X_test_valid.iloc[j] if hasattr(X_test_valid, 'iloc') else X_test_valid[j],
                            feature_names, 'XGBoost'
                        )
                        test_data_records['XGBoost'].append({
                            'SAMPLE_ID': test_samples.iloc[j]['SAMPLE_ID'],
                            'PATIENT_ID': test_samples.iloc[j]['PATIENT_ID'],
                            'ground_truth': test_samples.iloc[j]['ground_truth'],
                            'ensemble_prediction': label_encoder.inverse_transform([preds_original[j]])[0],
                            'prob_ensemble_prediction': preds_proba[j].max(),
                            'CENTER': center,
                            **xgb_importance
                        })
            else:
                print(f"No valid test samples for XGBoost in center {center}")
                acc_xgb = f1_xgb = prec_xgb = rec_xgb = 0.0

        except Exception as e:
            print(f"XGBoost failed for center {center}: {e}")
            acc_xgb = f1_xgb = prec_xgb = rec_xgb = 0.0
        
        rf_preds = None
        rf_proba = None
        try:
            # Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=SEED)
            rf_model.fit(X_train, y_train)
            rf_preds = rf_model.predict(X_test)
            rf_proba = rf_model.predict_proba(X_test)
            acc_rf = accuracy_score(y_test, rf_preds)
            f1_rf = f1_score(y_test, rf_preds, average='weighted', zero_division=0)
            prec_rf = precision_score(y_test, rf_preds, average='weighted', zero_division=0)
            rec_rf = recall_score(y_test, rf_preds, average='weighted', zero_division=0)
        except Exception as e:
            print(f"RandomForest failed for center {center}: {e}")
            acc_rf = f1_rf = prec_rf = rec_rf = 0.0

        # Collect test data for LogReg and RandomForest if save_test_data is True
        if save_test_data:
            test_samples = joint_pred[test_mask]
            feature_names = X_test.columns.tolist() if hasattr(X_test, 'columns') else [f'feature_{k}' for k in range(X_test.shape[1])]
            for j in range(len(y_test)):
                base_record = {
                    'SAMPLE_ID': test_samples.iloc[j]['SAMPLE_ID'],
                    'PATIENT_ID': test_samples.iloc[j]['PATIENT_ID'],
                    'ground_truth': test_samples.iloc[j]['ground_truth'],
                    'CENTER': center
                }
                # LogReg (if successful)
                if log_preds is not None and log_proba is not None:
                    log_importance = compute_sample_feature_importance(
                        log_reg, X_test.iloc[j] if hasattr(X_test, 'iloc') else X_test[j],
                        feature_names, 'LogReg', predicted_class=log_preds[j]
                    )
                    test_data_records['LogReg'].append({
                        **base_record,
                        'ensemble_prediction': label_encoder.inverse_transform([log_preds[j]])[0],
                        'prob_ensemble_prediction': log_proba[j].max(),
                        **log_importance
                    })
                # RandomForest (if successful)
                if rf_preds is not None and rf_proba is not None:
                    rf_importance = compute_sample_feature_importance(
                        rf_model, X_test.iloc[j] if hasattr(X_test, 'iloc') else X_test[j],
                        feature_names, 'RandomForest'
                    )
                    test_data_records['RandomForest'].append({
                        **base_record,
                        'ensemble_prediction': label_encoder.inverse_transform([rf_preds[j]])[0],
                        'prob_ensemble_prediction': rf_proba[j].max(),
                        **rf_importance
                    })

        # Store results
        results['CENTER'].append(center)
        results['num_test_samples'].append(len(y_test))

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

        print(f"Test samples: {len(y_test)}")
        print(f"{model1_name} accuracy: {acc_m1:.4f}")
        print(f"{model2_name} accuracy: {acc_m2:.4f}")
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
        'Method': [f'{model1_name} Original', f'{model2_name} Original', 'LogReg', 'XGBoost', 'Random Forest'],
        'Mean_Accuracy': [], 'Std_Accuracy': [],
        'Mean_F1': [], 'Std_F1': [],
        'Mean_Precision': [], 'Std_Precision': [],
        'Mean_Recall': [], 'Std_Recall': [],
        'Successful_Runs': []
    }

    for method_prefix in [model1_name, model2_name, 'LogReg', 'XGBoost', 'RandomForest']:
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
        successful_runs = len(results_df[results_df[f'{method_prefix}_accuracy'] > 0.0]) if method_prefix not in [model1_name, model2_name] else len(results_df)
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
    
    methods = [f'{model1_name}_accuracy', f'{model2_name}_accuracy', 'LogReg_accuracy', 'XGBoost_accuracy', 'RandomForest_accuracy']
    method_names = [f'{model1_name} Original', f'{model2_name} Original', 'Logistic Regression', 'XGBoost', 'Random Forest']
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

    # Save test data if requested
    if save_test_data:
        test_data_dir = os.path.join(output_dir, 'test_data')
        os.makedirs(test_data_dir, exist_ok=True)
        for model_name, records in test_data_records.items():
            if records:
                test_data_df = pd.DataFrame(records)
                test_data_file = os.path.join(test_data_dir, f'leave_one_center_out_{model_name}_test_data.csv')
                test_data_df.to_csv(test_data_file, index=False)
                print(f"Test data saved to: {test_data_file}")

    return results_df, summary_df

def main(fold=5, model1_path=None, model2_path=None, model1_name="GDD", model2_name="gpt_5", output_path=None, save_test_data=False):
    """Main function to run the complete analysis."""
    # Set seed for reproducibility
    set_seed(SEED)

    # Set default paths if not provided
    if model1_path is None:
        model1_path = "/data1/morrisq/yuj13/llm_genomics/genie_output/GDD"
    if model2_path is None:
        model2_path = "/data1/morrisq/yuj13/llm_genomics/genie_output/gpt-5"

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_dir = f"/data1/morrisq/yuj13/llm_genomics/genie_meta_model_result/{timestamp}"
    else:
        output_dir = f"{output_path}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Results will be saved to: {output_dir}")

    # Load and prepare data
    print("Loading and preparing data...")
    joint_pred, model1_name, model2_name = load_and_prepare_data(
        model1_path=model1_path,
        model2_path=model2_path,
        model1_name=model1_name,
        model2_name=model2_name
    )
    features, labels, original_model1, original_model2, label_encoder = prepare_features_and_labels(
        joint_pred, model1_name=model1_name, model2_name=model2_name
    )

    print(f"Dataset shape: {joint_pred.shape}")
    print(f"Number of unique centers: {joint_pred['CENTER'].nunique()}")
    print(f"Number of unique patients: {joint_pred['PATIENT_ID'].nunique()}")
    print(f"Number of unique cancer types: {joint_pred['ground_truth'].nunique()}")

    # Run n-fold cross-validation (split by PATIENT_ID)
    cv_results, cv_summary = run_5_fold_cv(
        joint_pred, features, labels, original_model1, original_model2, label_encoder, output_dir,
        fold=fold, model1_name=model1_name, model2_name=model2_name, save_test_data=save_test_data
    )

    # Run leave-one-center-out cross-validation
    loco_results, loco_summary = run_leave_one_center_out(
        joint_pred, features, labels, original_model1, original_model2, label_encoder, output_dir,
        model1_name=model1_name, model2_name=model2_name, save_test_data=save_test_data
    )

    # Save overall summary
    if loco_summary is not None:
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
    else:
        overall_summary = {
            'Evaluation_Method': [f'{fold}-Fold CV'],
            'Best_Method': [cv_summary.loc[cv_summary['Mean_Accuracy'].idxmax(), 'Method']],
            'Best_Accuracy': [cv_summary['Mean_Accuracy'].max()]
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
    parser.add_argument('--model1-path', type=str, default=None, help='Path to model1 data folder (default: GDD)')
    parser.add_argument('--model2-path', type=str, default=None, help='Path to model2 data folder (default: gpt-5)')
    parser.add_argument('--model1-name', type=str, default='GDD', help='Name for model1 (default: GDD)')
    parser.add_argument('--model2-name', type=str, default='gpt_5', help='Name for model2 (default: gpt_5)')
    parser.add_argument('--output-path', type=str, default=None, help='Custom output path (will append timestamp: {output_path}_{timestamp})')
    parser.add_argument('--save-test-data', action='store_true', help='Save test set predictions with sample_id, patient_id, ground_truth, ensemble_prediction, prob_ensemble_prediction')
    args = parser.parse_args()
    main(
        fold=args.fold,
        model1_path=args.model1_path,
        model2_path=args.model2_path,
        model1_name=args.model1_name,
        model2_name=args.model2_name,
        output_path=args.output_path,
        save_test_data=args.save_test_data
    )