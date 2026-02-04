"""
Eva-4B-V2 Inference Script for EvasionBench

This script demonstrates how to use Eva-4B-V2 model to detect evasive answers
in earnings call Q&A sessions.

Usage:
    python eva4b_inference.py --num_samples 5
    python eva4b_inference.py --num_samples 10 --device cuda
"""

import argparse
import json
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


PROMPT_TEMPLATE = """You are a financial analyst. Your task is to Detect Evasive Answers in Financial Q&A

Question: {question}
Answer: {answer}

Response format:
```json
{{"label": "direct|intermediate|fully_evasive"}}
```

Answer in ```json content, no other text"""


def load_model(model_name: str = "FutureMa/Eva-4B-V2", device: str = "auto"):
    """Load Eva-4B-V2 model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    print(f"Model loaded on device: {model.device}")
    return model, tokenizer


def load_evasionbench(num_samples: int = 5):
    """Load EvasionBench dataset from HuggingFace."""
    print(f"Loading EvasionBench dataset...")
    dataset = load_dataset("FutureMa/EvasionBench", split="train")
    print(f"Dataset loaded: {len(dataset)} samples")

    # Sample subset
    if num_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(num_samples))

    return dataset


def parse_response(response: str) -> dict:
    """Parse JSON response from model output."""
    # Try to extract JSON from markdown code block
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to extract raw JSON
    json_match = re.search(r'\{[^{}]*"label"[^{}]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return {"label": "unknown"}


def predict(model, tokenizer, question: str, answer: str) -> dict:
    """Run inference on a single Q&A pair."""
    prompt = PROMPT_TEMPLATE.format(question=question, answer=answer)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return parse_response(response), response


def main():
    parser = argparse.ArgumentParser(description="Eva-4B-V2 Evasion Detection")
    parser.add_argument("--model", type=str, default="FutureMa/Eva-4B-V2",
                        help="Model name or path")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cpu)")
    args = parser.parse_args()

    # Load model and dataset
    model, tokenizer = load_model(args.model, args.device)
    dataset = load_evasionbench(args.num_samples)

    print(f"\n{'='*60}")
    print(f"Running inference on {len(dataset)} samples")
    print(f"{'='*60}\n")

    correct = 0
    total = 0

    for i, sample in enumerate(dataset):
        uid = sample["uid"]
        question = sample["question"]
        answer = sample["answer"]
        gold_label = sample["eva4b_label"]

        # Truncate for display
        q_display = question[:100] + "..." if len(question) > 100 else question
        a_display = answer[:100] + "..." if len(answer) > 100 else answer

        print(f"Sample {i+1}/{len(dataset)}")
        print(f"  UID: {uid[:16]}...")
        print(f"  Question: {q_display}")
        print(f"  Answer: {a_display}")

        result, raw_response = predict(model, tokenizer, question, answer)
        pred_label = result.get("label", "unknown")

        is_correct = pred_label == gold_label
        correct += int(is_correct)
        total += 1

        status = "✓" if is_correct else "✗"
        print(f"  Gold: {gold_label} | Pred: {pred_label} {status}")
        print()

    # Summary
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"{'='*60}")
    print(f"Results: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
