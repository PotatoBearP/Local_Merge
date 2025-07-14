import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import re
from tqdm import tqdm
import json
from typing import List, Dict, Any


def extract_code_from_completion(completion: str) -> str:
    """
    Extracts the first valid Python function from the completion.
    """
    pattern = r"def\s+\w+\s*\(.*?\):[\s\S]+?(?=\ndef|\Z)"
    match = re.search(pattern, completion)
    return match.group(0).strip() if match else completion.strip()


def evaluate_code(problem: Dict[str, Any], code: str) -> bool:
    """
    Runs the test cases for a given problem against the candidate code.
    """
    namespace = {}
    try:
        exec(problem["prompt"] + "\n" + code, namespace)
        for test_case in problem["test"]:
            exec(test_case, namespace)
        return True
    except Exception:
        return False


def generate_completion(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):]


def evaluate_humaneval(
    model_name_or_path: str,
    batch_size: int = 1,
    device: str = "cuda",
    samples_per_task: int = 5,
    output_json: str = "humaneval_results.json",
    torch_dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    quantization_config: BitsAndBytesConfig = None,
    use_flash_attention_2: bool = False,
    max_sequence_length: int = 2048
) -> Dict[str, Any]:

    print(f"Loading model: {model_name_or_path}...")

    if load_in_4bit or load_in_8bit:
        if quantization_config is None:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        print("Using BitsAndBytes quantization...")

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "device_map": "auto" if device.startswith("cuda") else "cpu",
    }

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    if use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded with dtype: {model.dtype}")
    print(f"Model on device: {next(model.parameters()).device}")

    dataset = datasets.load_dataset("openai_humaneval", split="test")
    results = []

    pass_counts = []

    for problem in tqdm(dataset, desc=f"Evaluating HumanEval {samples_per_task}-shot per task"):
        problem_id = problem["task_id"]
        prompt = problem["prompt"]
        passed = 0
        completions: List[str] = []

        for _ in range(samples_per_task):
            completion = generate_completion(model, tokenizer, prompt, device)
            code = extract_code_from_completion(completion)
            completions.append(code)

            if evaluate_code(problem, code):
                passed += 1

        pass_at_1 = passed >= 1
        pass_counts.append(pass_at_1)

        results.append({
            "task_id": problem_id,
            "prompt": prompt,
            "samples": completions,
            "passed": passed,
            "total_samples": samples_per_task,
            "pass@1": pass_at_1
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    accuracy = sum(pass_counts) / len(pass_counts) * 100.0
    print(f"HumanEval pass@1 (with {samples_per_task} samples per task): {accuracy:.2f}%")

    return {
        "accuracy": accuracy,
        "results": results
    }
