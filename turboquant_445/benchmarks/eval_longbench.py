"""
LongBench evaluation for the Transformers-4.45.2-compatible TurboQuant package.

This runner is intentionally scoped to:
- meta-llama/Llama-3.1-8B-Instruct
- mistralai/Mistral-7B-Instruct-v0.3

It reuses KIVI's LongBench prompt templates and generation budgets so scores are
more comparable across the two methods.
"""

import argparse
import gc
import json
import os
import re
import string
import sys
import zipfile
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant_445.cache import TurboQuantCache

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
KIVI_CONFIG_DIR = REPO_ROOT / "KIVI" / "config"
DEFAULT_DATASET_PATH = "/home/nikita/benchmarking-kv-cache/datasets/LongBench/LongBench.py"
SUPPORTED_MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
}

TASK_CATEGORIES = {
    "Single-Doc QA": ["qasper", "multifieldqa_en"],
    "Multi-Doc QA": ["hotpotqa", "2wikimqa"],
    #
    # "Summarization": ["multi_news"],
    "Few-shot": ["triviaqa"]
    # "Synthetic": ["passage_count", "passage_retrieval_en"],
    # "Code": ["lcc", "repobench-p"],
}
ALL_TASKS = [task for tasks in TASK_CATEGORIES.values() for task in tasks]

METRIC_BY_TASK = {
    "qasper": "f1",
    "multifieldqa_en": "f1",
    "hotpotqa": "f1",
    "2wikimqa": "f1",
    "gov_report": "rouge_l",
    "multi_news": "rouge_l",
    "trec": "accuracy",
    "triviaqa": "f1",
    "passage_count": "accuracy",
    "passage_retrieval_en": "accuracy",
    "lcc": "prefix_match",
    "repobench-p": "prefix_match",
}


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def compute_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(normalize_text(prediction) == normalize_text(reference))
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_accuracy(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def compute_prefix_match(prediction: str, reference: str) -> float:
    pred_lines = prediction.strip().splitlines()
    ref_lines = reference.strip().splitlines()
    if not ref_lines:
        return 1.0 if not pred_lines else 0.0
    match_count = 0
    for p, r in zip(pred_lines, ref_lines):
        if p.strip() == r.strip():
            match_count += 1
        else:
            break
    return match_count / len(ref_lines)


METRIC_FN = {
    "f1": compute_f1,
    "rouge_l": compute_rouge_l,
    "accuracy": compute_accuracy,
    "prefix_match": compute_prefix_match,
}


def score_prediction(task: str, prediction: str, references: list[str]) -> float:
    fn = METRIC_FN[METRIC_BY_TASK[task]]
    return max(fn(prediction, ref) for ref in references)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_results_payload(model: str, bit_widths: list, tasks: list[str], all_results: dict) -> dict:
    return {
        "model": model,
        "bit_widths": [str(bw) for bw in bit_widths],
        "tasks": tasks,
        "results": {task: {str(bw): score for bw, score in scores.items()} for task, scores in all_results.items()},
    }


def write_results_payload(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_prediction_output_dir(model_name: str) -> Path:
    return REPO_ROOT / "turboquant_445" / "pred" / Path(model_name)


def write_task_predictions(prediction_dir: Path, task: str, records: list[dict]) -> Path:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{task}.jsonl"
    with prediction_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return prediction_path


def load_longbench_dataset(dataset_repo_or_path: str | None, dataset_name: str):
    if dataset_repo_or_path and os.path.exists(dataset_repo_or_path):
        base_path = dataset_repo_or_path
        if os.path.isfile(base_path) and (base_path.endswith(".py") or base_path.endswith(".zip")):
            base_path = os.path.dirname(base_path)

        data_dir = os.path.join(base_path, "data")
        local_jsonl = os.path.join(data_dir, f"{dataset_name}.jsonl")
        if os.path.exists(local_jsonl):
            return load_dataset("json", data_files=local_jsonl, split="train")

        local_zip = os.path.join(base_path, "data.zip")
        if os.path.exists(local_zip):
            extract_dir = os.path.join(base_path, ".extracted_longbench", "data")
            extracted_jsonl = os.path.join(extract_dir, f"{dataset_name}.jsonl")
            if not os.path.exists(extracted_jsonl):
                os.makedirs(extract_dir, exist_ok=True)
                member_name = f"data/{dataset_name}.jsonl"
                with zipfile.ZipFile(local_zip) as zip_file:
                    zip_file.extract(member_name, path=os.path.join(base_path, ".extracted_longbench"))
            return load_dataset("json", data_files=extracted_jsonl, split="train")

    return load_dataset(dataset_repo_or_path or "zai-org/LongBench", dataset_name, split="test")


def build_chat_prompt(tokenizer, prompt: str, model_name: str) -> str:
    model_name_lower = model_name.lower()
    if model_name_lower not in {
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/mistral-7b-instruct-v0.3",
    }:
        raise ValueError(f"Unsupported model for this runner: {model_name}")
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def maybe_truncate_prompt(tokenizer, prompt: str, max_prompt_tokens: int | None) -> str:
    if max_prompt_tokens is None:
        return prompt
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) <= max_prompt_tokens:
        return prompt
    half = max_prompt_tokens // 2
    return (
        tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
        + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    )


def load_task_examples(task: str, dataset_path: str | None, tokenizer, model_name: str, max_prompt_tokens: int | None):
    dataset2prompt = load_json(KIVI_CONFIG_DIR / "dataset2prompt.json")
    ds = load_longbench_dataset(dataset_path, task)
    examples = []
    for row in ds:
        answers = row.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        if not answers:
            answer = row.get("answer", "")
            answers = [answer] if answer else [""]
        prompt = dataset2prompt[task].format(**row)
        prompt = maybe_truncate_prompt(tokenizer, prompt, max_prompt_tokens)
        prompt = build_chat_prompt(tokenizer, prompt, model_name)
        examples.append({
            "prompt": prompt,
            "answers": answers,
        })
    return examples


def get_model_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        embed_device = model.hf_device_map.get("model.embed_tokens")
        if embed_device is not None:
            return torch.device(embed_device)
    return next(model.parameters()).device


def truncate_to_max_length(input_ids: torch.Tensor, max_ctx: int, gen_budget: int) -> torch.Tensor:
    allowed = max_ctx - gen_budget
    if input_ids.shape[1] > allowed:
        input_ids = input_ids[:, :allowed]
    return input_ids


@torch.no_grad()
def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int, max_ctx: int) -> str:
    device = get_model_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = truncate_to_max_length(inputs["input_ids"].to(device), max_ctx, max_new_tokens)
    attention_mask = torch.ones_like(input_ids)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        return_legacy_cache=False,
    )
    generated = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def generate_quantized(model, tokenizer, prompt: str, max_new_tokens: int, max_ctx: int, bit_width: int) -> str:
    device = get_model_input_device(model)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers

    num_outlier_channels = 0
    outlier_bits = 0
    if bit_width == 3:
        num_outlier_channels = 16 if head_dim == 64 else min(32, head_dim // 4)
        outlier_bits = 4

    cache = TurboQuantCache(
        head_dim=head_dim,
        bit_width=bit_width,
        num_layers=num_layers,
        num_outlier_channels=num_outlier_channels,
        outlier_bits=outlier_bits,
        device=device,
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = truncate_to_max_length(inputs["input_ids"].to(device), max_ctx, max_new_tokens)
    attention_mask = torch.ones_like(input_ids)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        past_key_values=cache,
        use_cache=True,
        return_legacy_cache=False,
    )
    generated = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_task(model, tokenizer, task: str, examples: list[dict], bit_widths: list, max_ctx: int, generation_tokens: dict[str, int]) -> tuple[dict, list[dict]]:
    max_gen = generation_tokens[task]
    results = {bw: [] for bw in bit_widths}
    prediction_records = []

    for idx, ex in enumerate(examples):
        example_predictions = {}
        example_scores = {}
        for bw in bit_widths:
            if bw == "baseline":
                pred = generate_baseline(model, tokenizer, ex["prompt"], max_gen, max_ctx)
            else:
                pred = generate_quantized(model, tokenizer, ex["prompt"], max_gen, max_ctx, bw)
            score = score_prediction(task, pred, ex["answers"])
            results[bw].append(score)
            example_predictions[str(bw)] = pred
            example_scores[str(bw)] = score
            gc.collect()
        prediction_records.append(
            {
                "task": task,
                "index": idx,
                "answers": ex["answers"],
                "predictions": example_predictions,
                "scores": example_scores,
            }
        )
        sys.stdout.write(f"\r  {task}: {idx + 1}/{len(examples)}")
        sys.stdout.flush()
    print()

    return {bw: sum(scores) / len(scores) if scores else 0.0 for bw, scores in results.items()}, prediction_records


def format_table(all_results: dict, bit_widths: list) -> str:
    labels = [f"{bw}-bit" if bw != "baseline" else "baseline" for bw in bit_widths]
    header = f"{'Task':<30} | " + " | ".join(f"{label:>10}" for label in labels)
    sep = "-" * len(header)
    lines = [sep, header, sep]
    cat_avgs = {bw: [] for bw in bit_widths}

    for category, tasks in TASK_CATEGORIES.items():
        lines.append(f"  [{category}]")
        for task in tasks:
            if task not in all_results:
                continue
            row_label = f"    {task} ({METRIC_BY_TASK[task]})"
            vals = []
            for bw in bit_widths:
                score = all_results[task].get(bw, 0.0) * 100
                vals.append(f"{score:>10.1f}")
                cat_avgs[bw].append(score)
            lines.append(f"{row_label:<30} | " + " | ".join(vals))
        lines.append("")

    lines.append(sep)
    avg_vals = []
    for bw in bit_widths:
        scores = cat_avgs[bw]
        avg_vals.append(f"{(sum(scores) / len(scores) if scores else 0.0):>10.1f}")
    lines.append(f"{'Overall Average':<30} | " + " | ".join(avg_vals))
    lines.append(sep)
    return "\n".join(lines)


def parse_bit_widths(s: str) -> list:
    result = []
    for part in s.split(","):
        part = part.strip()
        result.append("baseline" if part == "baseline" else int(part))
    return result


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def main():
    parser = argparse.ArgumentParser(description="LongBench evaluation for turboquant_445")
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--bit-widths", default="3,4,baseline")
    parser.add_argument("--tasks", default=None, help="Comma-separated task subset to run")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda:0 or cpu")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--output", default="results/longbench_results_445.json")
    args = parser.parse_args()

    bit_widths = parse_bit_widths(args.bit_widths)
    tasks = args.tasks.split(",") if args.tasks else ALL_TASKS
    unsupported = [task for task in tasks if task not in ALL_TASKS]
    if unsupported:
        raise ValueError(f"Unsupported tasks requested: {unsupported}")

    generation_tokens = load_json(KIVI_CONFIG_DIR / "dataset2maxlen.json")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = get_torch_dtype(args.dtype)
    if device == "cpu" and torch_dtype != torch.float32:
        torch_dtype = torch.float32

    print("=" * 72)
    print("  LongBench Evaluation — TurboQuant 4.45.2")
    print("=" * 72)
    print(f"  Model:       {args.model}")
    print(f"  Bit widths:  {bit_widths}")
    print(f"  Device:      {device}")
    print(f"  Dtype:       {torch_dtype}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        pretrained_model_name_or_path=args.model,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if device == "cpu":
        model_kwargs["device_map"] = "cpu"
    elif args.device:
        model_kwargs["device_map"] = {"": device}
    elif torch.cuda.device_count() > 1:
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "cuda:0"

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.eval()

    max_ctx = getattr(model.config, "max_position_embeddings", 32768)
    print(f"  max_position_embeddings={max_ctx}")
    print(f"  layers={model.config.num_hidden_layers}, heads={model.config.num_attention_heads}")
    print()

    all_results = {}
    output_path = Path(args.output)
    prediction_output_dir = get_prediction_output_dir(args.model)
    for task in tasks:
        print(f"Loading {task}...")
        examples = load_task_examples(
            task=task,
            dataset_path=args.dataset_path,
            tokenizer=tokenizer,
            model_name=args.model,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        print(f"  {len(examples)} examples loaded")
        task_results, prediction_records = evaluate_task(model, tokenizer, task, examples, bit_widths, max_ctx, generation_tokens)
        all_results[task] = task_results
        prediction_path = write_task_predictions(prediction_output_dir, task, prediction_records)
        write_results_payload(output_path, build_results_payload(args.model, bit_widths, tasks, all_results))
        for bw in bit_widths:
            label = f"{bw}-bit" if bw != "baseline" else "baseline"
            print(f"    {label}: {task_results[bw] * 100:.1f}")
        print(f"    predictions: {prediction_path}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print(format_table(all_results, bit_widths))
    write_results_payload(output_path, build_results_payload(args.model, bit_widths, tasks, all_results))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
