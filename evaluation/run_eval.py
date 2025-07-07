import os
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluation.winogrande_eval import evaluate_winogrande
# from evaluation.truthfulqa_eval import evaluate_truthfulqa
# from evaluation.mmlu_eval import evaluate_mmlu
# from evaluation.humaneval_eval import evaluate_humaneval

def evaluate_model_on_tasks(
    model_path: str,
    tasks: str,
    output_path: str,
    batch_size: int = 4,
    max_tokens: int = 512,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Evaluate a causal LLM on selected tasks and save results to JSON.
    """
    print(f"Using device: {device}")
    if isinstance(tasks, str):
        tasks = [t.strip() for t in tasks.split(",") if t.strip()]

    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    task_results = {}
    for task in tasks:
        task = task.lower()
        if task == "winogrande":
            task_results["winogrande"] = evaluate_winogrande(model, tokenizer, batch_size, device)
        # elif task == "truthfulqa":
        #     task_results["truthfulqa"] = evaluate_truthfulqa(model, tokenizer, batch_size, device)
        # elif task == "mmlu":
        #     task_results["mmlu"] = evaluate_mmlu(model, tokenizer, batch_size, device)
        # elif task == "humaneval":
        #     task_results["humaneval"] = evaluate_humaneval(model, tokenizer, max_tokens, device)
        else:
            raise ValueError(f"Unsupported task: {task}")

    avg_score = sum(v for v in task_results.values() if isinstance(v, (int, float))) / len(task_results)
    task_results["average"] = avg_score

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(task_results, f, indent=2)

    return task_results
