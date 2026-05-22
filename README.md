# LLM-Based Genomic Classification

This repository benchmarks large language models (LLMs) on three genomic classification tasks:

| # | Task | Labels |
|---|------|--------|
| 1 | **Mutation Status Prediction** | TUMOR-SOMATIC vs CHIP (Clonal Hematopoiesis of Indeterminate Potential) |
| 2 | **Oncogenic Driver Identification** | Oncogenic vs Benign |
| 3 | **Cancer Type Classification** | 34 cancer types from comprehensive genomic profiles |

Each task follows the same three-step pipeline:

## Step 0: **Generate features**

   A sample Genie feature table is included at `input_features/genie_ft_9_samples.csv`. To create your own for the entire Genie data:

   - Clone the [GDD-ENS repository](https://github.com/smilejennyyu/GDD_ENS.git).  
   - Download the `h19.fa` reference and GENIE data.  
   - Run:
      ```bash
      python generate_feature_table.py \
      --reference h19.fa \
      --genie-data /path/to/GENIE \
      --output input_features/genie_ft_samples.csv
   - See the full workflow in the [GDD-ENS documentation](https://github.com/smilejennyyu/GDD_ENS/blob/main/doc/workflow.md).

## Step 1: **Data Preprocessing and Report Generation**

   This step includes three tasks for preprocessing genomic data and generating mutation reports. All scripts are located in `1_generate_report/`.

   ### Task 1: Mutation Report Generation (`task1_mutation_report.py`)

   Processes mutation data from MSK and GENIE datasets to generate mutation reports with various output formats.

   **Input:**
   - `/data1/morrisq/yuj13/llm_genomics/msk_ch_2020`, `/data1/morrisq/yuj13/llm_genomics/msk_ch_2023`, `/data1/morrisq/yuj13/llm_genomics/genie_data_v17.0`

   **Output:** 4 variants (VAF/noVAF × aggregate/non-aggregate)
   - `/data1/morrisq/yuj13/llm_genomics/mutation_report/mutation_status_final*`

   ```bash
   # Run with config file (recommended)
   python 1_generate_report/task1_mutation_report.py --config configs/task1_mutation_config.yaml
   ```

   Configuration file: [configs/task1_mutation_config.yaml](configs/task1_mutation_config.yaml)

   ### Task 2: Oncogenic Mutation Report Generation (`task2_oncogenic_report.py`)

   Processes oncogenic mutation data from MSK IMPACT solid/heme dataset.

   **Input:**
   - `/data1/morrisq/yuj13/llm_genomics/msk_solid_heme`

   **Output:**
   - Balanced oncogenic:benign (2:1): `/data1/morrisq/yuj13/llm_genomics/mutation_report/impact_oncogenic_sample_new`
   - NSCLC subset: `/data1/morrisq/yuj13/llm_genomics/mutation_report/impact_oncogenic_nsclc_final`

   ```bash
   # Run with config file (recommended)
   python 1_generate_report/task2_oncogenic_report.py --config configs/task2_oncogenic_config.yaml
   ```

   Configuration file: [configs/task2_oncogenic_config.yaml](configs/task2_oncogenic_config.yaml)

   ### Task 3: Comprehensive Cancer Type Report Generation (`task3_cancer_type_report.py`)

   Main comprehensive report generator for cancer type classification. Generates per-sample genomic reports (mutations, CNAs, fusions, chromosomal arm changes, mutational signatures) from raw data using a YAML configuration.

   ```bash
   python 1_generate_report/task3_cancer_type_report.py --config configs/task3_cancer_type_config.yaml
   ```

   Configuration file: [configs/task3_cancer_type_config.yaml](configs/task3_cancer_type_config.yaml)

   **Output:** CSV file with tumor report and ground truth cancer type (e.g., `report/genie_reports.csv`)

   | Column                 | Description                                                    |
   | ---------------------- | -------------------------------------------------------------- |
   | `SAMPLE_ID`            | Tumor sample ID                                                |
   | `REPORT`               | Text summary of mutations, CNAs, fusions, and arm-level events |
   | `CANCER_TYPE`          | Ground-truth tumor type                                        |
   | `CANCER_TYPE_DETAILED` | Cancer subtype annotation                                      |
   | `SAMPLE_TYPE_DETAILED` | Sample type information (E.g., primary)                        |

   **Note:** For backwards compatibility, you can also run from the parent directory:
   ```bash
   python generate_report.py --config configs/task3_cancer_type_config.yaml
   ```


## Step 2: **LLM Classification**

   This step uses large language models to classify genomic data. The unified `llm_classifier.py` script supports three classification tasks. All scripts are located in `2_llm_classification/`.

   ### Task 1: Mutation Status Classification

   Classifies mutations as TUMOR-SOMATIC (tumor-specific) vs CHIP (clonal hematopoiesis).

   **Input:** Reports from Step 1 Task 1
   **Output:** Predictions with confidence scores

   ```bash
   python 2_llm_classification/llm_classifier.py --config configs/task1_mutation_status_llm_config.yaml
   ```

   Configuration file: [configs/task1_mutation_status_llm_config.yaml](configs/task1_mutation_status_llm_config.yaml)

   ### Task 2: Oncogenic Classification

   Classifies mutations as Oncogenic (cancer-driving) vs Benign.

   **Input:** Reports from Step 1 Task 2
   **Output:** Oncogenic predictions with probabilities

   ```bash
   python 2_llm_classification/llm_classifier.py --config configs/task2_oncogenic_llm_config.yaml
   ```

   Configuration file: [configs/task2_oncogenic_llm_config.yaml](configs/task2_oncogenic_llm_config.yaml)

   ### Task 3: Cancer Type Classification

   Predicts cancer type from comprehensive genomic reports.

   **Input:** Reports from Step 1 Task 3
   **Output:** Top 2 cancer type predictions with detailed genomic features

   ```bash
   python 2_llm_classification/llm_classifier.py --config configs/task3_cancer_type_llm_config.yaml
   ```

   Configuration file: [configs/task3_cancer_type_llm_config.yaml](configs/task3_cancer_type_llm_config.yaml)

   **Output Columns (Task 3):**

   | Column            | Description                                                                                      |
   | ----------------- | ------------------------------------------------------------------------------------------------ |
   | `SAMPLE_ID`       | Tumor sample ID                                                                                  |
   | `prediction1`     | Top predicted tumor type                                                                         |
   | `prob1`           | Probability of `prediction1`                                                                     |
   | `prediction2`     | Second-ranked predicted tumor type                                                               |
   | `prob2`           | Probability of `prediction2`                                                                     |
   | `sex`             | Patient's sex                                                                                    |
   | `sex_used`        | Whether sex influenced the prediction                                                            |
   | `key_genes`       | Top 3 mutated genes driving the prediction                                                       |
   | `key_arm_changes` | Top 3 chromosomal arm–level events driving the prediction                                        |
   | `key_scnas`       | Top 3 focal somatic copy-number abnormalities driving the prediction                             |
   | `key_fusions`     | Top 3 somatic fusions features driving the prediction                                            |
   | `important_data_part` | Most influential data type (mutations, CNAs, fusions, or arm changes)                        |
   | `explanation`     | Detailed rationale explaining the prediction (if enabled)                                        |
   | `ground_truth`    | Cancer type ground truth (for benchmarking)                                                      |

   See [2_llm_classification/README.md](2_llm_classification/README.md) for detailed usage and configuration options.


## Step 3: **Ensemble Models**

   This step combines predictions from task-specific models with LLM predictions using machine learning ensemble methods (Logistic Regression, Random Forest, XGBoost). All scripts are located in `3_ensemble_model/`.

   ### Task 1: Mutation Status Ensemble

   Combines two models' predictions for TUMOR-SOMATIC vs CHIP classification.

   **Example**: MetaCH + LLM (e.g., gpt-4o)

   #### Getting MetaCH predictions

   [MetaCH](https://github.com/ncoudray/MetaCH) is a random-forest classifier trained to distinguish clonal hematopoiesis (CHIP) mutations from tumor-somatic mutations. Run it on your mutation data to produce a predictions file, then rename the relevant columns to match the ensemble config:

   | Column | Value |
   |--------|-------|
   | `SAMPLE_ID` | sample identifier |
   | `PATIENT_ID` | patient identifier |
   | `ground_truth` | `TUMOR-SOMATIC` or `CHIP` |
   | `MetaCH_prediction` | `TUMOR-SOMATIC` or `CHIP` |
   | `MetaCH_prob` | confidence score (0–1) |
   | `CANCER_TYPE` | cancer type label |

   #### Merging with LLM predictions

   The ensemble expects a **single merged CSV** containing both MetaCH and LLM columns. Merge on `SAMPLE_ID` before running:

   ```python
   import pandas as pd
   metach = pd.read_csv("metach_predictions.csv")   # MetaCH_prediction, MetaCH_prob, ...
   llm    = pd.read_csv("llm_predictions.csv")       # prediction, prob, ...
   merged = metach.merge(llm[["SAMPLE_ID", "prediction", "prob"]], on="SAMPLE_ID")
   merged.to_csv("merged_predictions.csv", index=False)
   ```

   Then set `input.data_path` in `configs/task1_mutation_ensemble_config.yaml` to the merged file.

   ```bash
   python 3_ensemble_model/task12_mutation_ensemble.py --config configs/task1_mutation_ensemble_config.yaml
   ```

   Configuration file: [configs/task1_mutation_ensemble_config.yaml](configs/task1_mutation_ensemble_config.yaml)

   ### Task 2: Oncogenic Classification Ensemble

   Combines two models' predictions for Oncogenic vs Benign classification.

   **Example**: AlphaMissense + LLM (e.g., gpt-4o)

   #### Getting AlphaMissense predictions

   [AlphaMissense](https://github.com/google-deepmind/alphamissense) is Google DeepMind's precomputed database of pathogenicity scores for human missense variants. Download the lookup table and join it to your mutation data on `(CHROM, POS, REF, ALT)` or `(Hugo_Symbol, HGVSp_Short)`.

   Convert the raw `am_pathogenicity` score to a binary label using Google's recommended thresholds:

   ```python
   import pandas as pd

   am = pd.read_csv("AlphaMissense_hg38.tsv", sep="\t", comment="#")
   # Join to your mutations on chromosome position or protein change, then:
   am["AlphaMissense_prediction"] = am["am_class"].map({
       "likely_pathogenic": "Oncogenic",
       "ambiguous":         "Benign",
       "likely_benign":     "Benign",
   })
   am["AlphaMissense_prob"] = am["am_pathogenicity"]
   ```

   The merged file must contain:

   | Column | Value |
   |--------|-------|
   | `SAMPLE_ID` | sample identifier |
   | `PATIENT_ID` | patient identifier |
   | `ground_truth` | `Oncogenic` or `Benign` |
   | `AlphaMissense_prediction` | `Oncogenic` or `Benign` |
   | `AlphaMissense_prob` | `am_pathogenicity` score (0–1) |
   | `CANCER_TYPE` | cancer type label |

   #### Merging with LLM predictions

   ```python
   import pandas as pd
   am  = pd.read_csv("alphamissense_predictions.csv")  # AlphaMissense_prediction, AlphaMissense_prob, ...
   llm = pd.read_csv("llm_predictions.csv")             # prediction, prob, ...
   merged = am.merge(llm[["SAMPLE_ID", "prediction", "prob"]], on="SAMPLE_ID")
   merged.to_csv("merged_oncogenic_predictions.csv", index=False)
   ```

   Then set `input.data_path` in `configs/task2_oncogenic_ensemble_config.yaml` to the merged file.

   ```bash
   python 3_ensemble_model/task12_mutation_ensemble.py --config configs/task2_oncogenic_ensemble_config.yaml
   ```

   Configuration file: [configs/task2_oncogenic_ensemble_config.yaml](configs/task2_oncogenic_ensemble_config.yaml)

   ### Task 3: Cancer Type Ensemble

   Combines task-specific model (e.g., GDD) with LLM predictions for cancer type classification.

   **Example**: GDD + gpt-5

   **Input**: Two result folders
   - Task-specific model: `/data1/morrisq/yuj13/llm_genomics/genie_output/GDD/filtered_batch_non_msk`
   - LLM model: `/data1/morrisq/yuj13/llm_genomics/genie_output/gpt-5/filtered_batch_non_msk`

   **Output**: `/data1/morrisq/yuj13/llm_genomics/genie_meta_model_result/{timestamp}`

   ```bash
   python 3_ensemble_model/task3_cancer_type_ensemble.py --config configs/task3_cancer_type_ensemble_config.yaml
   ```

   Configuration file: [configs/task3_cancer_type_ensemble_config.yaml](configs/task3_cancer_type_ensemble_config.yaml)

   **Output Files:**

   | File | Description |
   |------|-------------|
   | `5_fold_cv_detailed_results.csv` | Per-fold metrics for all methods |
   | `5_fold_cv_summary.csv` | Mean ± std metrics for each method |
   | `5_fold_cv_accuracy_plot.png` | Bar plot comparing methods |
   | `overall_summary.csv` | Best performing method |

   **Methods Compared:**
   - Individual model baselines (e.g., GDD, gpt-5)
   - Logistic Regression ensemble
   - XGBoost ensemble
   - Random Forest ensemble

   See [3_ensemble_model/README.md](3_ensemble_model/README.md) for detailed usage and configuration options.


---


## Prerequisites

- **Python 3.8+**
- **pip** for installing dependencies: `pip install -r requirements.txt`
- Access to data files (CSV/txt) as specified in the configs

**For API-based models:**
- (Azure OpenAI) Azure OpenAI endpoint (`export AZURE_OPENAI_ENDPOINT='xxx'`) & API key (`export AZURE_OPENAI_API_KEY='xxx'`)
- (OpenAI) OpenAI API key (`export OPENAI_API_KEY='xxx'`)
- (Claude) LiteLLM proxy or Claude API access

**For local models:**
- GPU with CUDA support (recommended) or CPU
- `pip install transformers torch`
- For quantization (recommended): `pip install bitsandbytes accelerate`
- Supported models: MedGemma, Qwen, DeepSeek, or any HuggingFace chat model

**For ensemble models:**
- `pip install scikit-learn xgboost matplotlib`
- Predictions from at least two models (from Step 2 or task-specific models)

---



