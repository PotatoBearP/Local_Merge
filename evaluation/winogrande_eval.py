import torch
import datasets
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json

def compute_log_probs_conditional(model, tokenizer, prompts, continuations, device):
    """Compute log-probs for `continuations` conditioned on `prompts` (one-by-one for low memory)."""
    model.eval()
    results = []

    with torch.inference_mode():
        for prompt, continuation in zip(prompts, continuations):
            full_text = prompt + continuation

            enc = tokenizer(full_text, return_tensors="pt").to(device)
            input_ids = enc["input_ids"]
            prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

            # Target only continuation tokens
            target_ids = input_ids.clone()
            target_ids[:, :prompt_len] = -100  # ignore prompt in loss

            outputs = model(input_ids)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())

            # Only sum continuation losses
            loss_sum = loss[target_ids[:, 1:] != -100].sum()
            results.append(-loss_sum.item())

    return results

def evaluate_winogrande(
    model_name_or_path: str,
    batch_size: int = 4,
    device: str = "cuda",
    output_json: str = "winogrande_eval_results.json",
    torch_dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    quantization_config: BitsAndBytesConfig = None,
    use_flash_attention_2: bool = False,
    max_sequence_length: int = 2048
) -> float:

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

    dataset = datasets.load_dataset("winogrande", "winogrande_xl", split="test", trust_remote_code=True)
    correct = 0
    total = 0
    results = []
    random.seed(1234)

    shot_examples = random.sample(list(dataset), 5)

    def format_example(ex, include_answer=True):
        ans_text = ex['option1'] if ex['answer'] == '1' else ex['option2']
        if include_answer:
            return f"Q: {ex['sentence']}\nA: {ex['option1']} or {ex['option2']}\nAnswer: {ans_text}\n"
        else:
            return f"Q: {ex['sentence']}\nA: {ex['option1']} or {ex['option2']}\nAnswer:"

    shot_context = "".join([format_example(ex) for ex in shot_examples])
    model_device = next(model.parameters()).device

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating Winogrande 5-shot"):
        batch = dataset[i:i + batch_size]
        batch = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]

        prompts = [
            shot_context + f"Q: {ex['sentence']}\nA: {ex['option1']} or {ex['option2']}\nAnswer:"
            for ex in batch
        ]
        continuations_1 = [ex["option1"] for ex in batch]
        continuations_2 = [ex["option2"] for ex in batch]

        logp_1 = compute_log_probs_conditional(model, tokenizer, prompts, continuations_1, model_device)
        logp_2 = compute_log_probs_conditional(model, tokenizer, prompts, continuations_2, model_device)

        for j, ex in enumerate(batch):
            pred = "1" if logp_1[j] > logp_2[j] else "2"
            is_correct = pred == ex["answer"]
            if is_correct:
                correct += 1
            total += 1
            results.append({
                "question": ex['sentence'],
                "llm_choices": [ex['option1'], ex['option2']],
                "correct_choice": ex['option1'] if ex['answer'] == "1" else ex['option2'],
                "label": ex['answer'],
                "prediction": pred,
                "is_correct": is_correct,
                "logp_choice1": logp_1[j],
                "logp_choice2": logp_2[j]
            })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    accuracy = correct / total * 100.0
    print(f"Winogrande 5-shot Accuracy: {accuracy:.2f}%")
    return accuracy
