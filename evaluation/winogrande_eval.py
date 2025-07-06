import torch
import datasets
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import json
# Import for quantization
from accelerate import Accelerator
from accelerate import init_empty_weights # For device_map
from bitsandbytes import BitsAndBytesConfig # For 4-bit/8-bit quantization

# --- Keep your compute_log_probs as is for now ---
def compute_log_probs(model, tokenizer, prompts, device):
    """Compute negative log-likelihood for each prompt in a batch."""
    # Ensure attention_mask is handled for proper loss calculation
    encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    with torch.no_grad():
        # Ensure the model is in evaluation mode and on the correct device
        model.eval()
        model.to(device) # Ensure model is on device before inference

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous() # Mask for valid tokens

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Reshape for CrossEntropyLoss
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())

    # Only sum loss for actual tokens (non-padded, non-shifted-off-end)
    seq_loss = (loss * shift_mask).sum(dim=1)
    return -seq_loss # Return log-probabilities (higher is better)

def evaluate_winogrande(
    model_name_or_path: str, # Changed to model_name_or_path
    batch_size: int = 4, # You'll need to experiment with this
    device: str = "cuda",
    output_json: str = "winogrande_eval_results.json",
    # Add parameters for memory optimization
    torch_dtype: torch.dtype = torch.bfloat16, # Or torch.float16 if bfloat16 not supported
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    quantization_config: BitsAndBytesConfig = None,
    use_flash_attention_2: bool = False,
    max_sequence_length: int = 2048 # To help with truncation and managing length
) -> float:
    """Evaluate causal LM on Winogrande 5-shot task."""

    print(f"Loading model: {model_name_or_path}...")

    # Set up quantization configuration if enabled
    if load_in_4bit or load_in_8bit:
        if quantization_config is None:
            # Default quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_quant_type="nf4", # Or "fp4"
                bnb_4bit_compute_dtype=torch.bfloat16 if torch_dtype == torch.bfloat16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        print("Using BitsAndBytes quantization...")

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device, # Can be "auto" if you have multiple GPUs
        "trust_remote_code": True,
    }

    if load_in_4bit or load_in_8bit:
        model_kwargs["quantization_config"] = quantization_config
    if use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2...")

    if device.startswith("cuda") and not isinstance(model_kwargs["device_map"], dict):
         model_kwargs["device_map"] = {"": 0} if device == "cuda" else {"": int(device.split(":")[-1])}
    elif device == "cpu":
        model_kwargs["device_map"] = "cpu"
    else:
        if not isinstance(model_kwargs["device_map"], dict) and not device.startswith("cuda"):
            model_kwargs["device_map"] = "auto"


    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    print(f"Model loaded with dtype: {model.dtype}")
    print(f"Model on device: {next(model.parameters()).device}")

    dataset = datasets.load_dataset("winogrande", "winogrande_xl", split="validation")
    correct = 0
    total = 0
    results = []

    model.eval()

    shot_examples = random.sample(list(dataset), 5) 
    def format_example(ex, include_answer=True):
        ans_text = ex['option1'] if ex['answer']=='1' else ex['option2']
        if include_answer:
            return f"Q: {ex['sentence']}\nA: {ex['option1']} or {ex['option2']}\nAnswer: {ans_text}\n"
        else:
            return f"Q: {ex['sentence']}\nA: {ex['option1']} or {ex['option2']}\nAnswer:" # For generation tasks if needed

    shot_context = "".join([format_example(ex) for ex in shot_examples])

    # Iterate with a tqdm progress bar
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating Winogrande 5-shot"):
        batch = dataset[i:i + batch_size]

        prompts_1 = [
            shot_context + f"Q: {ex['sentence']}\nA: {ex['option1']} or {ex['option2']}\nAnswer: {ex['option1']}"
            for ex in batch
        ]
        prompts_2 = [
            shot_context + f"Q: {ex['sentence']}\nA: {ex['option1']} or {ex['option2']}\nAnswer: {ex['option2']}"
            for ex in batch
        ]

        prompts_1 = [p[:max_sequence_length] for p in prompts_1]
        prompts_2 = [p[:max_sequence_length] for p in prompts_2]


        logp_1 = compute_log_probs(model, tokenizer, prompts_1, next(model.parameters()).device) # Use model's actual device
        logp_2 = compute_log_probs(model, tokenizer, prompts_2, next(model.parameters()).device)

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

    accuracy = correct / total * 100.0
    print(f"Winogrande 5-shot Accuracy: {accuracy:.2f}%")
    return accuracy
