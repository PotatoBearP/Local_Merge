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


def evaluate_mmlu(
    model_name_or_path: str,
    subject: str = "abstract_algebra",
    batch_size: int = 4,
    device: str = "cuda",
    output_json: str = "mmlu_eval_results.json",
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

    dataset = datasets.load_dataset("hendrycks_test", subject, split="validation")
    dataset = dataset.shuffle(seed=1234).select(range(int(0.5 * len(dataset))))  # Optional downsample

    correct = 0
    total = 0
    results = []

    # 5-shot prompt examples
    few_shot_examples = random.sample(list(dataset), 5)

    def format_example(ex, include_answer=True):
        text = f"Q: {ex['question']}\nA:\n(A) {ex['choices'][0]}\n(B) {ex['choices'][1]}\n(C) {ex['choices'][2]}\n(D) {ex['choices'][3]}\n"
        if include_answer:
            return text + f"Answer: {ex['answer']}\n"
        else:
            return text + "Answer:"

    few_shot_context = "".join([format_example(ex) for ex in few_shot_examples])
    model_device = next(model.parameters()).device

    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating MMLU {subject} 5-shot"):
        batch = dataset[i:i + batch_size]
        batch = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]

        prompts = [
            few_shot_context + format_example(ex, include_answer=False)
            for ex in batch
        ]

        choices = ["A", "B", "C", "D"]
        logp_all_choices = []

        for choice in choices:
            continuations = [f" {choice}" for _ in batch]
            logps = compute_log_probs_batch(model, tokenizer, prompts, continuations, model_device)
            logp_all_choices.append(logps)

        logp_all_choices = torch.tensor(logp_all_choices).T  # (batch_size, 4)

        for j, ex in enumerate(batch):
            best_choice_idx = int(logp_all_choices[j].argmax())
            pred_choice = choices[best_choice_idx]
            gt_choice = ex["answer"]

            is_correct = pred_choice == gt_choice
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": ex["question"],
                "choices": ex["choices"],
                "correct_answer": ex["answer"],
                "prediction": pred_choice,
                "is_correct": is_correct,
                "logp_choices": {choices[k]: logp_all_choices[j][k].item() for k in range(4)},
            })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    accuracy = correct / total * 100.0
    print(f"MMLU {subject} 5-shot Accuracy: {accuracy:.2f}%")
    return accuracy
