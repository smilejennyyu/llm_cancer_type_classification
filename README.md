# LLM-Driven Genomic Reporting & Classification

This repository contains two main steps for working with genomic data and large-language-model (LLM)–based cancer type classification:

Step 0: **Generate features**

   A sample Genie feature table is included at `input_features/genie_ft_9_samples.csv`. To create your own for the entire Genie data:

   - Clone the [GDD-ENS repository](https://github.com/smilejennyyu/GDD_ENS.git).  
   - Download the `h19.fa` reference and GENIE data.  
   - Run:
      ```bash
      python generate_feature_table.py \
      --reference h19.fa \
      --genie-data /path/to/GENIE \
      --output input_features/genie_ft_<n>_samples.csv
   - See the full workflow in the [GDD-ENS documentation](https://github.com/smilejennyyu/GDD_ENS/blob/main/doc/workflow.md).

Step 1: **Run `generate_report.py`**  
   Generates per-sample genomic reports (mutations, CNAs, fusions, chromosomal arm changes) from raw data using a YAML configuration.  
   ```
   python generate_report.py --config report_config.yaml
   ```
   Output: csv file with tumor report and ground truth cancer type. (e.g. `report/genie_reports.csv`). The output file column names are shown below:

   | Column                 | Description                                                    |
   | ---------------------- | -------------------------------------------------------------- |
   | `SAMPLE_ID`            | Tumor sample identifier                                        |
   | `REPORT`               | Text summary of mutations, CNAs, fusions, and arm-level events |
   | `CANCER_TYPE`          | Ground-truth tumor category                                    |
   | `CANCER_TYPE_DETAILED` | Fine-grained subtype annotation                                |
   | `SAMPLE_TYPE_DETAILED` | Detailed description of the sample source or assay             |


Step 2: **Run `classify_cancer_llm.py`**  
   Runs an LLM (Azure/OpenAI or local MedGemma) to predict tumor types and probabilities from those reports.
   ```
   python classify_cancer_llm.py --config llm_config.yaml
   ```
   Output: csv file with the predictions and corresponding explanations from LLM (e.g. `output/output.csv`). The output file column names are shown below:
   | Column            | Description                                                                                      |
   | ----------------- | ------------------------------------------------------------------------------------------------ |
   | `SAMPLE_ID`       | Tumor sample identifier                                                                          |
   | `prediction1`     | Top predicted tumor type                                                                         |
   | `prob1`           | Probability of `prediction1`                                                                     |
   | `prediction2`     | Second-ranked predicted tumor type                                                               |
   | `prob2`           | Probability of `prediction2`                                                                     |
   | `sex`             | Patient’s sex                                                                                    |
   | `sex_used`        | `true` if sex influenced the prediction; otherwise `false`                                       |
   | `key_genes`       | Top 3 mutated genes driving the prediction (or `None`)                                           |
   | `key_arm_changes` | Top 3 chromosomal arm–level events (amplifications/deletions) driving the prediction (or `None`) |
   | `key_scnas`       | Top 3 focal somatic copy-number abnormalities driving the prediction (or `None`)                 |
   | `key_fusions`     | Top 3 somatic fusions or rearrangements driving the prediction (or `None`)                       |
   | `explanation`     | 1–2 sentence rationale linking features to the top prediction                                    |
   | `ground_truth`    | Known cancer type (for benchmarking)                                                             |


---


## Prerequisites

- **Python 3.8+**  
- **pip** for installing dependencies: `pip install -r requirements.txt`
- Access to data files (CSV/txt) as specified in the configs  
- (If using Azure) an Azure OpenAI endpoint (`export AZURE_OPENAI_ENDPOINT='xxx'`) & API key (`export AZURE_OPENAI_API_KEY='xxx'`), or  
- (If using MedGemma) GPU or CPU setup with huggingface Transformers. 

---



