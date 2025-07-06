import torch
import datasets
import random
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
import json


def compute_log_probs(model, tokenizer, prompts, device, max_length=384):
    """Compute negative log-likelihood for each prompt in a batch."""
    # Tokenize on CPU, move to GPU after truncation
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())

    seq_loss = (loss * shift_mask).sum(dim=1)
    return -seq_loss  # Return log-probabilities (higher is better)


def evaluate_winogrande(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 1,
    device: str = "cuda",
    output_json: str = "winogrande_eval_results.json",
    fewshot_k: int = 1,
    max_eval: int = None,
    max_length: int = 384,
) -> float:
    """Efficient evaluation of causal LM on Winogrande with few-shot learning."""
    random.seed(1234)  # Set seed for reproducibility
    dataset = datasets.load_dataset("winogrande", "winogrande_xl", split="validation")
    if max_eval:
        dataset = dataset.select(range(min(max_eval, len(dataset))))

    model.eval()
    model.half()  
    model.to(device)

    # Format few-shot examples once
    shots = random.sample(list(dataset), fewshot_k)

    def format_example(ex):
        return f"Q: {ex['sentence']}\nA: {ex['option1']} or {ex['option2']}\nAnswer: {ex['option1'] if ex['answer'] == '1' else ex['option2']}\n"

    shot_context = "".join([format_example(ex) for ex in shots])

    correct = 0
    total = 0
    results = []

    for i in tqdm(range(fewshot_k, len(dataset), batch_size), desc="Evaluating Winogrande"):
        batch = dataset[i:i + batch_size]

        prompts_1 = [
            f"{shot_context}Q: {ex['sentence']}\nA: {ex['option1']} or {ex['option2']}\nAnswer: {ex['option1']}\n"
            for ex in batch
        ]
        prompts_2 = [
            f"{shot_context}Q: {ex['sentence']}\nA: {ex['option1']} or {ex['option2']}\nAnswer: {ex['option2']}\n"
            for ex in batch
        ]

        logp_1 = compute_log_probs(model, tokenizer, prompts_1, device, max_length)
        logp_2 = compute_log_probs(model, tokenizer, prompts_2, device, max_length)

        for j, ex in enumerate(batch):
            pred = "1" if logp_1[j] > logp_2[j] else "2"
            is_correct = pred == ex["answer"]
            correct += int(is_correct)
            total += 1
            results.append({
                "question": ex['sentence'],
                "llm_choices": [ex['option1'], ex['option2']],
                "correct_choice": ex['option1'] if ex['answer'] == "1" else ex['option2'],
                "label": ex['answer'],
                "prediction": pred,
                "is_correct": is_correct,
                "logp_choice1": logp_1[j].item(),
                "logp_choice2": logp_2[j].item()
            })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return correct / total * 100.0
