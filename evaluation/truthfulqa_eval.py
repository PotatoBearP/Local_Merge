import torch
import datasets
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json


def compute_log_probs_batch(model, tokenizer, prompts, continuations, device):
    model.eval()
    results = []

    full_texts = [p + c for p, c in zip(prompts, continuations)]
    encodings = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(device)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    prompt_lens = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]

    target_ids = input_ids.clone()
    for i, plen in enumerate(prompt_lens):
        target_ids[i, :plen] = -100  # ignore prompt tokens in loss

    with torch.inference_mode():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())

        for i in range(loss.size(0)):
            loss_sum = loss[i][shift_labels[i] != -100].sum()
            results.append(-loss_sum.item())

    return results


def evaluate_truthfulqa_mc2(
    model_name_or_path: str,
    batch_size: int = 4,
    device: str = "cuda",
    output_json: str = "truthfulqa_eval_results.json",
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

    dataset = datasets.load_dataset("truthful_qa", "mc2", split="validation")
    dataset = dataset.shuffle(seed=1234).select(range(int(0.5 * len(dataset))))  # Optional downsample for speed.

    correct = 0
    total = 0
    results = []

    model_device = next(model.parameters()).device

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating TruthfulQA MC2"):
        batch = dataset[i:i + batch_size]
        batch = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]

        prompts = [f"Q: {ex['question']}\nA:" for ex in batch]
        all_choices = [ex["mc2_targets"]["choices"] for ex in batch]
        correct_answers = [ex["mc2_targets"]["labels"][0] for ex in batch]

        # Compute log-probs for each choice.
        choice_logps = []
        for choices in all_choices:
            logps = compute_log_probs_batch(model, tokenizer, prompts * len(choices), choices, model_device)
            choice_logps.append(logps)

        for j, ex in enumerate(batch):
            choices = all_choices[j]
            logps = choice_logps[j]
            best_choice_idx = int(torch.tensor(logps).argmax())
            pred_choice = choices[best_choice_idx]
            gt_choice = correct_answers[j]

            is_correct = pred_choice == gt_choice
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": ex["question"],
                "llm_choices": choices,
                "correct_choice": gt_choice,
                "prediction": pred_choice,
                "is_correct": is_correct,
                "logp_choices": dict(zip(choices, logps)),
            })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    accuracy = correct / total * 100.0
    print(f"TruthfulQA MC2 0-shot Accuracy: {accuracy:.2f}%")
    return accuracy