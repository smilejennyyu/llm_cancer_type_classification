#!/usr/bin/env python3
import os
import sys
import csv
import json
import logging
import argparse
import re
from typing import List, Optional, Dict, Any

import pandas as pd
import yaml
from pydantic import BaseModel, ValidationError

# AzureOpenAI import (only needed if using Azure/OpenAI backend)
try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

# ------------------ Shared schema ------------------
class TumorResponse(BaseModel):
    prediction1: str
    prob1: float
    prediction2: str
    prob2: float
    sex: str
    sex_used: bool
    key_genes: Optional[List[str]]
    key_arm_changes: Optional[List[str]]
    key_scnas: Optional[List[str]]
    key_fusions: Optional[List[str]]
    explanation: str

# ------------------ Helpers ------------------
def clean_model_output(raw: str) -> str:
    text = raw.strip()
    text = text.replace("```", "")
    text = re.sub(r'^\s*json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bjson\s*$', '', text, flags=re.IGNORECASE)
    match = re.search(r'(\{.*\})', text, flags=re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    return text

def setup_logging(log_file: str):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_and_validate(json_str: str) -> TumorResponse:
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e}\nRaw output snippet: {json_str[:500]}")
    try:
        return TumorResponse.model_validate(obj)
    except ValidationError as ve:
        raise ValueError(f"Schema validation failed: {ve}\nParsed object: {obj}")

# ------------------ Backend init ------------------
def init_azure_client(azure_conf: Dict[str, Any]) -> Any:
    if AzureOpenAI is None:
        raise ImportError("AzureOpenAI SDK not installed; required for Azure/OpenAI backend.")
    api_key = azure_conf.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = azure_conf.get("endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
    if not api_key or not endpoint:
        logging.warning("Azure OpenAI api_key or endpoint missing; fallback to env vars or operation may fail.")
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2025-01-01-preview",
        azure_endpoint=endpoint
    )
    return client

def init_medgemma_model(medgemma_conf: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    # … existing cache_dir logic …

    model_id = medgemma_conf["model_id"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # decide default torch dtype
    if device.type == "cuda":
        default_dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
    else:
        default_dtype = torch.float32

    # compute_dtype for BnB comes from config
    dtype_str = medgemma_conf.get("bnb_4bit_compute_dtype", "float16")
    compute_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    # master switch
    quant = medgemma_conf.get("quantization", False)

    # only build a BitsAndBytesConfig if quantization==True
    bnb_config = None
    if quant:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    load_kwargs = {"torch_dtype": default_dtype, "device_map": "auto"}
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return {"model": model, "tokenizer": tokenizer, "device": device}

# ------------------ Inference ------------------
def infer_azure(client: Any, background: str, user_prompt: str, model_name: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": background},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content

def infer_medgemma(local_objs: Dict[str, Any], background: str, user_prompt: str) -> str:
    import torch
    model = local_objs["model"]
    tokenizer = local_objs["tokenizer"]

    messages = [
        {"role": "system", "content": background},
        {"role": "user", "content": user_prompt}
    ]
    if not hasattr(tokenizer, "apply_chat_template"):
        raise AttributeError("Expected tokenizer with apply_chat_template for MedGemma-style formatting.")
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(local_objs["device"])
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        new_tokens = generation[0][input_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return decoded

# ------------------ Main ------------------
def main(config_path: str):
    cfg = load_yaml(config_path)
    model_choice = (cfg.get("model_choice") or "").lower()
    input_prompts_path = cfg["input_prompts"]
    output_path = cfg["output_file"]
    background_path = cfg["background_prompt"]
    log_file = cfg.get("log_file", "app.log")

    setup_logging(log_file)
    logging.info(f"Starting run with model_choice={model_choice}")

    if not os.path.isfile(background_path):
        logging.error(f"Background prompt file missing: {background_path}")
        sys.exit(1)
    with open(background_path, "r") as f:
        background = f.read()

    if not os.path.exists(input_prompts_path):
        logging.error(f"Input prompts CSV not found: {input_prompts_path}")
        sys.exit(1)
    prompts = pd.read_csv(input_prompts_path)

    # Resume support
    completed = set()
    if os.path.exists(output_path):
        with open(output_path, newline='') as f:
            reader = csv.DictReader(f)
            completed = {row["SAMPLE_ID"] for row in reader if "SAMPLE_ID" in row}
    prompts = prompts[~prompts["SAMPLE_ID"].isin(completed)]

    # Output setup
    write_header = not os.path.exists(output_path)
    fieldnames = [
        "SAMPLE_ID", "prediction1", "prob1", "prediction2", "prob2",
        "sex", "sex_used", "key_genes", "key_arm_changes",
        "key_scnas", "key_fusions", "explanation", "ground_truth"
    ]

    # Initialize backend
    azure_client = None
    medgemma_objs = None
    if model_choice == "azure":
        azure_conf = cfg.get("azure", {})
        if azure_conf is None:
            logging.error("Azure config missing in YAML while model_choice is 'azure'.")
            sys.exit(1)
        azure_client = init_azure_client(azure_conf)
        model_name = azure_conf.get("model", "o3-mini")
    elif model_choice == "medgemma":
        medgemma_conf = cfg.get("medgemma", {})
        if medgemma_conf is None or "model_id" not in medgemma_conf:
            logging.error("MedGemma config missing or 'model_id' not provided.")
            sys.exit(1)
        medgemma_objs = init_medgemma_model(medgemma_conf)
    else:
        logging.error(f"Unsupported model_choice '{model_choice}'; must be 'azure' or 'medgemma'.")
        sys.exit(1)

    with open(output_path, mode="a", newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for idx, row in prompts.iterrows():
            sample_id = row.get("SAMPLE_ID")
            user_prompt = row.get("REPORT") or row.get("report") or ""
            ground_truth = row.get("CANCER_TYPE") or row.get("cancer_type") or ""

            if pd.isna(sample_id) or pd.isna(user_prompt):
                logging.warning(f"Skipping malformed row index={idx}")
                continue

            try:
                if model_choice == "azure":
                    raw = infer_azure(azure_client, background, user_prompt, model_name)
                else:
                    raw = infer_medgemma(medgemma_objs, background, user_prompt)

                cleaned = clean_model_output(raw)
                parsed = parse_and_validate(cleaned)
                data = parsed.model_dump()

                writer.writerow({
                    "SAMPLE_ID": sample_id,
                    "prediction1": data["prediction1"],
                    "prob1": data["prob1"],
                    "prediction2": data["prediction2"],
                    "prob2": data["prob2"],
                    "sex": data["sex"],
                    "sex_used": data["sex_used"],
                    "key_genes": json.dumps(data["key_genes"]) if data["key_genes"] is not None else None,
                    "key_arm_changes": json.dumps(data["key_arm_changes"]) if data["key_arm_changes"] is not None else None,
                    "key_scnas": json.dumps(data["key_scnas"]) if data["key_scnas"] is not None else None,
                    "key_fusions": json.dumps(data["key_fusions"]) if data["key_fusions"] is not None else None,
                    "explanation": data["explanation"],
                    "ground_truth": ground_truth
                })

                if idx % 1 == 0:
                    fout.flush()
                if idx % 10 == 0:
                    logging.info(f"Processed {idx} samples (last SAMPLE_ID={sample_id})")

            except Exception as e:
                logging.error(f"Error on SAMPLE_ID={sample_id} index={idx}: {e}")

        fout.flush()

    logging.info("Run complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tumor inference runner (Azure/OpenAI or MedGemma).")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    main(args.config)
