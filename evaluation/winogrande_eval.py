import torch
import datasets
import random
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
import json

def compute_log_probs(model, tokenizer, prompts, device):
    """Compute negative log-likelihood for each prompt in a batch."""
    encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

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
    batch_size: int = 4,
    device: str = "cuda",
    output_json: str = "winogrande_eval_results.json"
) -> float:
    """Evaluate causal LM on Winogrande 5-shot task."""
    dataset = datasets.load_dataset("winogrande", "winogrande_xl", split="validation")
    correct = 0
    total = 0
    results = []

    model.eval()
    model.to(device)

    shots = random.sample(list(dataset), 5)

    def format_example(ex):
        return f"Q: {ex['sentence']}\nA: {ex['option1']} or {ex['option2']}\nAnswer: {ex['option1'] if ex['answer']=='1' else ex['option2']}\n"

    shot_context = "".join([format_example(ex) for ex in shots])

    for i in tqdm(range(5, len(dataset), batch_size), desc="Evaluating Winogrande 5-shot"):
        batch = dataset[i:i + batch_size]

        prompts_1 = [
            shot_context + f"Q: {ex['sentence']}\nA: {ex['option1']} or {ex['option2']}\nAnswer: {ex['option1']}\n"
            for ex in batch
        ]
        prompts_2 = [
            shot_context + f"Q: {ex['sentence']}\nA: {ex['option1']} or {ex['option2']}\nAnswer: {ex['option2']}\n"
            for ex in batch
        ]

        logp_1 = compute_log_probs(model, tokenizer, prompts_1, device)
        logp_2 = compute_log_probs(model, tokenizer, prompts_2, device)

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
                "logp_choice1": logp_1[j].item(),
                "logp_choice2": logp_2[j].item()
            })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return correct / total * 100.0
