"""
Unified LLM Classifier for Genomic Data
Supports three classification tasks:
  - Task 1: Mutation status (TUMOR-SOMATIC vs CHIP)
  - Task 2: Oncogenic classification (Benign vs Oncogenic)
  - Task 3: Cancer type classification

Configured via YAML file. See configs/ directory for task-specific configurations.
"""

import pandas as pd
import os
import csv
import json
import sys
import logging
import time
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from openai import AzureOpenAI, OpenAI

# Optional imports for local models
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None


# ============================================================================
# Pydantic Models for Response Validation
# ============================================================================

class MutationStatusResponse(BaseModel):
    """Response model for Task 1: Mutation status classification"""
    prediction: str
    prob: float
    explanation: Optional[str] = None


class OncogenicResponse(BaseModel):
    """Response model for Task 2: Oncogenic classification"""
    prediction: str
    prob: float
    explanation: Optional[str] = None


class CancerTypeResponse(BaseModel):
    """Response model for Task 3: Cancer type classification"""
    prediction1: str
    prob1: float
    prediction2: str
    prob2: float
    sex: Optional[str] = None
    sex_used: Optional[str] = None
    key_genes: Optional[str] = None
    key_arm_changes: Optional[str] = None
    key_scnas: Optional[str] = None
    key_fusions: Optional[str] = None
    important_data_part: Optional[str] = None
    explanation: Optional[str] = None


# ============================================================================
# Configuration and Logging
# ============================================================================

def load_config(config_path: str) -> Dict:
    """Load and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict):
    """Setup logging based on config."""
    log_config = config.get('logging', {})
    log_file = log_config.get('file', 'llm_classification.log')

    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    )


def infer_task_type(config: Dict) -> str:
    """Infer task type from configuration."""
    # Check for cancer type specific fields
    if 'valid_cancer_types' in config:
        return 'cancer_type'
    # Check for oncogenic classification
    elif 'valid_predictions' in config and 'Oncogenic' in config['valid_predictions']:
        return 'oncogenic'
    # Default to mutation status
    elif 'valid_predictions' in config and 'TUMOR-SOMATIC' in config['valid_predictions']:
        return 'mutation_status'
    else:
        raise ValueError("Cannot infer task type from config. Please check configuration file.")


# ============================================================================
# LLM Client Initialization
# ============================================================================

def load_local_model(config: Dict):
    """Load a local HuggingFace model."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers and torch are required for local models. "
            "Install with: pip install transformers torch"
        )

    local_config = config['local']
    model_path = local_config.get('model_path') or local_config.get('model_name')
    device = local_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    use_quantization = local_config.get('quantization', True)

    logging.info(f"Loading local model: {model_path}")
    logging.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set pad_token to eos_token")

    # Determine dtype
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    # Load model with quantization if enabled and on CUDA
    if use_quantization and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )

        if device == "cpu":
            model = model.to(device)

    logging.info("Model loaded successfully")

    return {
        'model': model,
        'tokenizer': tokenizer,
        'device': device,
        'max_new_tokens': local_config.get('max_new_tokens', 256),
        'temperature': local_config.get('temperature', 0.7),
        'top_p': local_config.get('top_p', 0.9)
    }


def init_llm_client(config: Dict):
    """Initialize LLM client based on provider configuration."""
    provider = config['llm_provider']

    if provider == 'azure':
        azure_config = config['azure']
        return AzureOpenAI(
            api_key=os.getenv(azure_config['api_key_env']),
            api_version=azure_config['api_version'],
            azure_endpoint=os.getenv(azure_config['endpoint_env'])
        ), azure_config['model_name']

    elif provider == 'openai':
        openai_config = config['openai']
        return OpenAI(
            api_key=os.getenv(openai_config['api_key_env'])
        ), openai_config['model_name']

    elif provider == 'claude':
        claude_config = config['claude']
        return OpenAI(
            base_url=claude_config['base_url'],
            api_key=claude_config['api_key']
        ), claude_config['model_name']

    elif provider == 'local':
        # For local models, return the model dict
        return load_local_model(config), None

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ============================================================================
# API Call with Retry Logic
# ============================================================================

def make_local_model_inference(model_dict: Dict, messages: List[Dict]) -> str:
    """
    Make an inference call to a local HuggingFace model.

    Args:
        model_dict: Dictionary containing model, tokenizer, and config
        messages: Messages to send to the model

    Returns:
        The model's response as a string
    """
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    device = model_dict['device']
    max_new_tokens = model_dict['max_new_tokens']
    temperature = model_dict['temperature']
    top_p = model_dict['top_p']

    try:
        # Format messages for chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize with attention to potential issues
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        # Safety check: Verify token IDs are within vocab size
        vocab_size = len(tokenizer)
        max_token_id = model_inputs.input_ids.max().item()

        if max_token_id >= vocab_size:
            logging.error(f"Token ID {max_token_id} exceeds vocab size {vocab_size}")
            raise ValueError(f"Invalid token ID detected: {max_token_id} >= {vocab_size}")

        # Safety check: Verify input length
        input_length = model_inputs.input_ids.shape[1]
        if hasattr(tokenizer, 'model_max_length') and input_length > tokenizer.model_max_length:
            logging.warning(f"Input length {input_length} exceeds max length {tokenizer.model_max_length}, truncating")
            model_inputs = tokenizer(
                [text],
                return_tensors="pt",
                max_length=tokenizer.model_max_length,
                truncation=True
            ).to(device)

        # Generate with error handling
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )

        # Decode only the generated part
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    except RuntimeError as e:
        if "CUDA" in str(e):
            logging.error(f"CUDA error during inference: {e}")
            # Clear CUDA cache
            if torch is not None:
                torch.cuda.empty_cache()
        raise


def make_api_call_with_retry(client, model_name: str, messages: List[Dict],
                             sample_id: str, config: Dict):
    """
    Make an API call with exponential backoff retry logic.
    Handles both API-based and local models.

    Args:
        client: LLM client instance (API client or local model dict)
        model_name: Model name to use (None for local models)
        messages: Messages to send to the API/model
        sample_id: Sample ID for logging
        config: Configuration dict with retry settings

    Returns:
        API response or model output string
    """
    # Check if this is a local model
    if isinstance(client, dict) and 'model' in client:
        # Local model - no retry needed, just inference
        try:
            response_text = make_local_model_inference(client, messages)
            # Create a mock response object similar to OpenAI API
            class MockResponse:
                def __init__(self, content):
                    self.choices = [type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': content
                        })()
                    })()]

            return MockResponse(response_text)

        except Exception as e:
            logging.error(f"Local model inference error for {sample_id}: {e}")
            raise

    # API-based model - use retry logic
    retry_config = config['retry']
    max_retries = retry_config['max_retries']
    wait_time = retry_config['initial_wait_time']
    max_wait_time = retry_config['max_wait_time']

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            return response

        except Exception as e:
            error_message = str(e)

            # Check if it's a rate limit error
            if '429' in error_message or 'rate limit' in error_message.lower():
                if attempt < max_retries - 1:
                    logging.warning(f"Rate limit hit for {sample_id}. Attempt {attempt + 1}/{max_retries}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, max_wait_time)
                else:
                    logging.error(f"Max retries reached for {sample_id} due to rate limiting")
                    raise
            else:
                # For non-rate-limit errors, raise immediately
                raise

    return None


# ============================================================================
# Response Processing
# ============================================================================

def create_reprompt_message(original_prompt: str, report: str,
                           error_message: str, attempt: int,
                           task_type: str) -> List[Dict]:
    """Create a reprompt message when initial attempt fails."""

    if task_type == 'mutation_status':
        format_example = '{"prediction": "TUMOR-SOMATIC or CHIP or UNKNOWN", "prob": 0.0}'
    elif task_type == 'oncogenic':
        format_example = '{"prediction": "Oncogenic or Benign", "prob": 0.0}'
    else:  # cancer_type
        format_example = '{"prediction1": "Cancer Type", "prob1": 0.0, "prediction2": "Cancer Type", "prob2": 0.0}'

    if attempt == 1:
        reprompt = f"""{original_prompt}

IMPORTANT: Your response must be valid JSON only. Do not include any markdown formatting, code blocks, or additional text.

Respond with ONLY a JSON object in this exact format:
{format_example}"""
    elif attempt == 2:
        reprompt = f"""Return ONLY a valid JSON object with no other text, markdown, or formatting:

Example format:
{format_example}

Your JSON response:"""
    else:
        reprompt = f"""Return only valid JSON: {format_example}

No markdown, no code blocks, just JSON."""

    return [
        {"role": "system", "content": reprompt},
        {"role": "user", "content": report}
    ]


def clean_json_response(model_reply: str) -> str:
    """Clean up JSON formatting from model response."""
    # Remove markdown code blocks
    if '```json' in model_reply:
        model_reply = model_reply.split('```json')[1].split('```')[0]
    elif '```' in model_reply:
        model_reply = model_reply.replace('```', '')

    # Remove leading 'json' text
    if model_reply.strip().startswith('json'):
        model_reply = model_reply.strip()[4:].strip()

    return model_reply.strip()


def normalize_prediction(prediction: str, config: Dict) -> str:
    """Normalize prediction value based on config rules."""
    # Handle spelling variations
    normalize_map = config.get('normalize_predictions', {})
    if prediction in normalize_map:
        original = prediction
        prediction = normalize_map[prediction]
        logging.info(f"Normalized {original} to {prediction}")

    return prediction


def validate_and_parse_response(model_reply: str, sample_id: str,
                                task_type: str, config: Dict) -> BaseModel:
    """Validate and parse model response based on task type."""
    # Clean JSON
    model_reply = clean_json_response(model_reply)

    # Parse JSON
    parsed_json = json.loads(model_reply)

    # Parse into appropriate Pydantic model
    if task_type == 'mutation_status':
        parsed = MutationStatusResponse.model_validate(parsed_json)

        # Normalize prediction
        parsed.prediction = normalize_prediction(parsed.prediction, config)

        # Validate prediction value
        valid_predictions = config.get('valid_predictions', ['TUMOR-SOMATIC', 'CHIP', 'UNKNOWN'])
        if parsed.prediction not in valid_predictions:
            raise ValueError(f"Invalid prediction: {parsed.prediction}")

    elif task_type == 'oncogenic':
        parsed = OncogenicResponse.model_validate(parsed_json)

        # Validate prediction value
        valid_predictions = config.get('valid_predictions', ['Oncogenic', 'Benign'])
        if parsed.prediction not in valid_predictions:
            raise ValueError(f"Invalid prediction: {parsed.prediction}")

    else:  # cancer_type
        parsed = CancerTypeResponse.model_validate(parsed_json)

        # Validate cancer types
        valid_cancer_types = config.get('valid_cancer_types', [])
        if valid_cancer_types:
            if parsed.prediction1 not in valid_cancer_types:
                logging.warning(f"prediction1 '{parsed.prediction1}' not in valid cancer types for {sample_id}")
            if parsed.prediction2 not in valid_cancer_types:
                logging.warning(f"prediction2 '{parsed.prediction2}' not in valid cancer types for {sample_id}")

    # Validate probability
    if hasattr(parsed, 'prob') and not 0 <= parsed.prob <= 1:
        raise ValueError(f"Probability must be between 0 and 1, got: {parsed.prob}")

    if hasattr(parsed, 'prob1') and not 0 <= parsed.prob1 <= 1:
        raise ValueError(f"prob1 must be between 0 and 1, got: {parsed.prob1}")

    if hasattr(parsed, 'prob2') and not 0 <= parsed.prob2 <= 1:
        raise ValueError(f"prob2 must be between 0 and 1, got: {parsed.prob2}")

    return parsed


# ============================================================================
# Sample Processing
# ============================================================================

def process_sample_with_reprompt(client, model_name: str, sample_id: str,
                                report: str, ground_truth: Any,
                                prompt_template: str, task_type: str,
                                config: Dict, row_data: Optional[Dict] = None) -> Dict:
    """
    Process a single sample with reprompting logic on errors.

    Args:
        client: LLM client instance
        model_name: Model name to use
        sample_id: Sample identifier
        report: Report text to analyze
        ground_truth: Ground truth label
        prompt_template: Prompt template
        task_type: Type of task (mutation_status, oncogenic, cancer_type)
        config: Configuration dict
        row_data: Dictionary containing all columns from input row

    Returns:
        Dictionary with result data to write to CSV
    """
    max_reprompt_attempts = config['retry']['max_reprompt_attempts']
    last_error = None

    for reprompt_attempt in range(max_reprompt_attempts + 1):
        try:
            # First attempt uses original prompt, subsequent use reprompts
            if reprompt_attempt == 0:
                messages = [
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": report}
                ]
                logging.info(f"Processing {sample_id} - Initial attempt")
            else:
                messages = create_reprompt_message(
                    prompt_template, report, str(last_error),
                    reprompt_attempt, task_type
                )
                logging.info(f"Processing {sample_id} - Reprompt attempt {reprompt_attempt}")

            # Make API call
            response = make_api_call_with_retry(
                client, model_name, messages, sample_id, config
            )

            # Extract response
            model_reply = response.choices[0].message.content.strip()

            # Validate and parse
            parsed = validate_and_parse_response(model_reply, sample_id, task_type, config)

            # Build result dict
            result = build_result_dict(parsed, ground_truth, row_data, task_type, config)

            if reprompt_attempt > 0:
                logging.info(f"Successfully processed {sample_id} after {reprompt_attempt} reprompt(s)")

            return result

        except json.JSONDecodeError as e:
            last_error = e
            logging.warning(f"JSON parsing error for {sample_id} (attempt {reprompt_attempt + 1}): {e}")
            if reprompt_attempt < max_reprompt_attempts:
                time.sleep(1)
            else:
                logging.error(f"All reprompt attempts exhausted for {sample_id}")

        except Exception as e:
            last_error = e
            logging.warning(f"Error processing {sample_id} (attempt {reprompt_attempt + 1}): {e}")
            if reprompt_attempt < max_reprompt_attempts:
                time.sleep(1)
            else:
                logging.error(f"All reprompt attempts exhausted for {sample_id}")

    # All attempts failed - return error result
    return build_error_result(ground_truth, row_data, task_type, config, str(last_error))


def build_result_dict(parsed: BaseModel, ground_truth: Any,
                     row_data: Optional[Dict], task_type: str,
                     config: Dict) -> Dict:
    """Build result dictionary from parsed response."""
    result = {}

    # Add input columns (excluding report_str and ground truth column)
    if row_data is not None:
        gt_col = get_ground_truth_column(task_type)
        result = {k: v for k, v in row_data.items() if k not in ['report_str', gt_col]}

    # Add ground truth
    result['ground_truth'] = ground_truth

    # Add predictions based on task type
    if task_type == 'mutation_status':
        result['prediction'] = parsed.prediction
        result['prob'] = parsed.prob
        if config.get('include_explanation') and parsed.explanation:
            result['explanation'] = parsed.explanation

    elif task_type == 'oncogenic':
        result['prediction'] = parsed.prediction
        result['prob'] = parsed.prob
        if config.get('include_explanation') and parsed.explanation:
            result['explanation'] = parsed.explanation

    else:  # cancer_type
        result['prediction1'] = parsed.prediction1
        result['prob1'] = parsed.prob1
        result['prediction2'] = parsed.prediction2
        result['prob2'] = parsed.prob2

        # Add optional fields
        if parsed.sex:
            result['sex'] = parsed.sex
        if parsed.sex_used:
            result['sex_used'] = parsed.sex_used
        if parsed.key_genes:
            result['key_genes'] = parsed.key_genes
        if parsed.key_arm_changes:
            result['key_arm_changes'] = parsed.key_arm_changes
        if parsed.key_scnas:
            result['key_scnas'] = parsed.key_scnas
        if parsed.key_fusions:
            result['key_fusions'] = parsed.key_fusions
        if parsed.important_data_part:
            result['important_data_part'] = parsed.important_data_part
        if config.get('include_explanation') and parsed.explanation:
            result['explanation'] = parsed.explanation

    return result


def build_error_result(ground_truth: Any, row_data: Optional[Dict],
                      task_type: str, config: Dict, error_msg: str) -> Dict:
    """Build error result dictionary."""
    result = {}

    # Add input columns
    if row_data is not None:
        gt_col = get_ground_truth_column(task_type)
        result = {k: v for k, v in row_data.items() if k not in ['report_str', gt_col]}

    # Add ground truth
    result['ground_truth'] = ground_truth

    # Add error values
    if task_type == 'cancer_type':
        result['prediction1'] = 'Error'
        result['prob1'] = 0.0
        result['prediction2'] = 'Error'
        result['prob2'] = 0.0
    else:
        result['prediction'] = 'Error'
        result['prob'] = 0.0

    if config.get('include_explanation'):
        result['explanation'] = f"Error: {error_msg}"

    return result


def get_ground_truth_column(task_type: str) -> str:
    """Get the ground truth column name for each task type."""
    if task_type == 'mutation_status':
        return 'Mutation_Status'
    elif task_type == 'oncogenic':
        return 'ONCOGENIC'
    else:  # cancer_type
        return 'CANCER_TYPE'


def get_output_columns(input_columns: List[str], task_type: str, config: Dict) -> List[str]:
    """Determine output columns based on task type."""
    # Start with input columns, excluding report_str
    output_cols = [col for col in input_columns if col != 'report_str']

    # Replace ground truth column name
    gt_col = get_ground_truth_column(task_type)
    output_cols = ['ground_truth' if col == gt_col else col for col in output_cols]

    # Add prediction columns
    if task_type == 'cancer_type':
        pred_cols = ['prediction1', 'prob1', 'prediction2', 'prob2',
                    'sex', 'sex_used', 'key_genes', 'key_arm_changes',
                    'key_scnas', 'key_fusions', 'important_data_part']
    else:
        pred_cols = ['prediction', 'prob']

    for col in pred_cols:
        if col not in output_cols:
            output_cols.append(col)

    # Add explanation if needed
    if config.get('include_explanation') and 'explanation' not in output_cols:
        output_cols.append('explanation')

    return output_cols


# ============================================================================
# Main Processing Loop
# ============================================================================

def process_csv_file(input_file: str, output_file: str, client,
                     model_name: str, prompt_template: str,
                     task_type: str, config: Dict):
    """Process a single CSV file."""
    logging.info(f"Processing file: {input_file}")

    # Load data
    try:
        data = pd.read_csv(input_file)
        logging.info(f"Loaded {len(data)} samples from {input_file}")
    except Exception as e:
        logging.error(f"Failed to load {input_file}: {e}")
        return

    # Verify required columns
    required_cols = config['required_columns']
    if not all(col in data.columns for col in required_cols):
        logging.error(f"Missing required columns in {input_file}. Required: {required_cols}")
        return

    # Resume logic
    completed_samples = set()
    if config.get('resume', True) and os.path.exists(output_file):
        try:
            with open(output_file, newline='') as f:
                reader = csv.DictReader(f)
                completed_samples = {row['SAMPLE_ID'] for row in reader if 'SAMPLE_ID' in row}
            logging.info(f"Found {len(completed_samples)} completed samples, resuming...")
        except Exception as e:
            logging.warning(f"Could not read existing output file: {e}")

    # Filter already completed samples
    data_to_process = data[~data['SAMPLE_ID'].isin(completed_samples)]
    logging.info(f"Processing {len(data_to_process)} remaining samples")

    if len(data_to_process) == 0:
        logging.info("No samples to process, skipping file")
        return

    # Determine output columns
    output_columns = get_output_columns(list(data.columns), task_type, config)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Process samples
    write_header = not os.path.exists(output_file)
    gt_col = get_ground_truth_column(task_type)

    with open(output_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_columns, extrasaction='ignore')

        if write_header:
            writer.writeheader()

        for index, row in data_to_process.iterrows():
            sample_id = row['SAMPLE_ID']
            report = row['report_str']
            ground_truth = row[gt_col]
            row_data = row.to_dict()

            # Apply rate limiting
            if index > 0:
                time.sleep(config['rate_limit']['delay_between_calls'])

            # Process sample
            result = process_sample_with_reprompt(
                client, model_name, sample_id, report, ground_truth,
                prompt_template, task_type, config, row_data
            )

            # Write to CSV
            writer.writerow(result)
            f.flush()

            # Log progress
            if (index + 1) % 10 == 0:
                logging.info(f"Completed {index + 1} samples from {input_file}")
                print(f"Progress: {index + 1}/{len(data_to_process)} samples processed")

    logging.info(f"Completed processing {input_file}")


def process_all_csv_files(config: Dict, client, model_name: str,
                          prompt_template: str, task_type: str):
    """Process all CSV files in input directory."""
    input_dir = config['input_dir']
    output_dir = config['output_dir']

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all CSV files
    csv_files = sorted(Path(input_dir).glob('*.csv'))

    if not csv_files:
        logging.warning(f"No CSV files found in {input_dir}")
        return

    logging.info(f"Found {len(csv_files)} CSV files to process")

    # Process each file
    for csv_file in csv_files:
        input_file = str(csv_file)
        output_file = os.path.join(output_dir, csv_file.name)

        process_csv_file(
            input_file, output_file, client, model_name,
            prompt_template, task_type, config
        )

    logging.info("All files processed!")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified LLM classifier for genomic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process mutation status classification
  python llm_classifier.py --config configs/task1_mutation_status_llm_config.yaml

  # Process oncogenic classification
  python llm_classifier.py --config configs/task2_oncogenic_llm_config.yaml

  # Process cancer type classification
  python llm_classifier.py --config configs/task3_cancer_type_llm_config.yaml
        """
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    setup_logging(config)

    logging.info("=" * 80)
    logging.info("Starting LLM Classification")
    logging.info("=" * 80)
    logging.info(f"Configuration loaded from {args.config}")

    # Infer task type
    task_type = infer_task_type(config)
    logging.info(f"Task type: {task_type}")

    # Initialize LLM client
    client, model_name = init_llm_client(config)
    logging.info(f"LLM provider: {config['llm_provider']}")

    if config['llm_provider'] == 'local':
        local_config = config['local']
        model_id = local_config.get('model_path') or local_config.get('model_name')
        logging.info(f"Local model: {model_id}")
    else:
        logging.info(f"Model: {model_name}")

    # Load prompt template
    prompt_template_path = config['prompt_template']
    try:
        with open(prompt_template_path, 'r') as f:
            prompt_template = f.read()
        logging.info(f"Loaded prompt template from {prompt_template_path}")
    except FileNotFoundError:
        logging.error(f"Prompt template file not found: {prompt_template_path}")
        sys.exit(1)

    # Process all files
    process_all_csv_files(config, client, model_name, prompt_template, task_type)

    logging.info("=" * 80)
    logging.info("LLM Classification Complete!")
    logging.info("=" * 80)
    print(f"\nProcessing complete! Results saved to {config['output_dir']}")


if __name__ == "__main__":
    main()
