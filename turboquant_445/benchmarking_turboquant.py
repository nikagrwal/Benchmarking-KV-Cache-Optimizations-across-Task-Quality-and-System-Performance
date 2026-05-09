import argparse
import json
import os
import platform
import statistics
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant_445.cache import TurboQuantCache


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
KIVI_CONFIG_DIR = REPO_ROOT / "KIVI" / "config"
DEFAULT_DATASET_PATH = str(REPO_ROOT / "datasets" / "LongBench" / "LongBench.py")
DEFAULT_DATASET_NAMES = ("gov_report",)


def resolve_model_path(model_name_or_path: str) -> str:
    model2path_path = KIVI_CONFIG_DIR / "model2path.json"
    if model2path_path.exists():
        with model2path_path.open("r", encoding="utf-8") as f:
            model2path = json.load(f)
        return model2path.get(model_name_or_path, model_name_or_path)
    return model_name_or_path


def model_tag(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1]


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def cuda_time_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, float(start.elapsed_time(end))


def compute_last_token_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    pretraining_tp = getattr(model.config, "pretraining_tp", 1)
    if pretraining_tp > 1:
        lm_head_slices = model.lm_head.weight.split(
            model.config.vocab_size // pretraining_tp,
            dim=0,
        )
        logits = [torch.nn.functional.linear(hidden_states, lm_head_slices[i]) for i in range(pretraining_tp)]
        return torch.cat(logits, dim=-1).float()
    return model.lm_head(hidden_states).float()


@dataclass
class SampleMetrics:
    context_tokens: int
    ttft_ms: float
    output_throughput_toks_per_s: Optional[float]
    prefill_kv_memory_mb: float
    cache_size_mb: float


def build_model_and_tokenizer(
    model_name: str,
    cache_dir: Optional[str],
    local_files_only: bool,
    dtype_str: str,
):
    model_name = resolve_model_path(model_name)
    torch_dtype = get_torch_dtype(dtype_str)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        pretrained_model_name_or_path=model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if torch.cuda.device_count() > 1:
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "cuda:0"

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.eval()
    return model, tokenizer, model_name


def get_model_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        embed_device = model.hf_device_map.get("model.embed_tokens")
        if embed_device is not None:
            return torch.device(embed_device)
    return next(model.parameters()).device


def get_decoder_model(model):
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    raise ValueError(f"Unsupported causal LM architecture: {type(model).__name__}")


def build_chat_prompt(tokenizer, prompt: str, model_name: str) -> str:
    model_name_lower = model_name.lower()
    if "llama-3" in model_name_lower and "instruct" in model_name_lower:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if "mistral-7b-instruct-v0.3" in model_name_lower or "mistral-v0.2-instruct" in model_name_lower:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def load_longbench_dataset(dataset_repo_or_path: Optional[str], dataset_name: str):
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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_max_new_tokens() -> Dict[str, int]:
    dataset2maxlen = load_json(KIVI_CONFIG_DIR / "dataset2maxlen.json")
    return {name: int(value) for name, value in dataset2maxlen.items()}


def normalize_dataset_names(dataset_names: Any) -> Tuple[str, ...]:
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    normalized: List[str] = []
    for dataset_name in dataset_names:
        normalized.extend(part.strip() for part in str(dataset_name).split(",") if part.strip())
    return tuple(normalized)


def maybe_truncate_prompt(tokenizer, prompt: str, max_prompt_tokens: Optional[int]) -> str:
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


def load_prompts(
    dataset_name: str,
    dataset_path: Optional[str],
    tokenizer,
    model_name: str,
    max_prompt_tokens: Optional[int],
) -> List[Dict[str, Any]]:
    dataset2prompt = load_json(KIVI_CONFIG_DIR / "dataset2prompt.json")
    if dataset_name not in dataset2prompt:
        raise ValueError(f"Unknown dataset_name={dataset_name}")

    prompt_format = dataset2prompt[dataset_name]
    ds = load_longbench_dataset(dataset_path, dataset_name)
    prompts = []
    for row in ds:
        context_tokens = len(
            tokenizer(
                row["context"],
                truncation=False,
                return_tensors="pt",
            ).input_ids[0]
        )
        prompt = prompt_format.format(**row)
        prompt = maybe_truncate_prompt(tokenizer, prompt, max_prompt_tokens)
        prompt = build_chat_prompt(tokenizer, prompt, model_name)
        prompts.append(
            {
                "prompt": prompt,
                "context_tokens": context_tokens,
            }
        )
    return prompts


def get_auto_outlier_channels(bit_width: int, head_dim: int) -> Tuple[int, int]:
    if bit_width == 3:
        return (16 if head_dim == 64 else min(32, head_dim // 4), 4)
    return 0, 0


def build_turboquant_cache(
    model,
    bit_width: int,
    num_outlier_channels: int,
    outlier_bits: int,
    device: torch.device,
    use_packing: bool,
) -> TurboQuantCache:
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    return TurboQuantCache(
        head_dim=head_dim,
        bit_width=bit_width,
        num_layers=model.config.num_hidden_layers,
        num_outlier_channels=num_outlier_channels,
        outlier_bits=outlier_bits,
        device=device,
        use_packing=use_packing,
    )


@torch.inference_mode()
def benchmark_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    context_tokens: int,
    bit_width: int,
    num_outlier_channels: int,
    outlier_bits: int,
    use_packing: bool,
) -> SampleMetrics:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_device = get_model_input_device(model)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    cache = build_turboquant_cache(
        model=model,
        bit_width=bit_width,
        num_outlier_channels=num_outlier_channels,
        outlier_bits=outlier_bits,
        device=input_device,
        use_packing=use_packing,
    )
    decoder_model = get_decoder_model(model)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(input_device)

    def do_prefill():
        return decoder_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

    outputs, prefill_latency_ms = cuda_time_ms(do_prefill)
    past_key_values = outputs.past_key_values
    cache_size_mb = past_key_values.get_memory_bytes() / (1024 ** 2)

    torch.cuda.synchronize()
    mem_after_prefill = torch.cuda.memory_allocated(input_device)
    prefill_kv_memory_mb = max(0, mem_after_prefill - mem_before) / (1024 ** 2)

    last_hidden_state = outputs[0][:, -1:, :].contiguous()
    first_token = torch.argmax(compute_last_token_logits(model, last_hidden_state)[:, -1, :], dim=-1, keepdim=True)
    del last_hidden_state
    del outputs

    generated_tokens = 1
    current_input_ids = first_token
    current_attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones(
                (attention_mask.shape[0], 1),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            ),
        ],
        dim=1,
    )

    decode_step_times_ms: List[float] = []
    should_continue_decoding = first_token.item() != tokenizer.eos_token_id

    for step_idx in range(max(0, max_new_tokens - 1)):
        if not should_continue_decoding:
            break

        def do_decode():
            return decoder_model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        step_outputs, step_ms = cuda_time_ms(do_decode)
        decode_step_times_ms.append(step_ms)
        if step_idx == 0:
            ttft_ms = prefill_latency_ms + step_ms

        step_hidden_state = step_outputs[0][:, -1:, :].contiguous()
        next_token = torch.argmax(compute_last_token_logits(model, step_hidden_state)[:, -1, :], dim=-1, keepdim=True)
        del step_hidden_state
        generated_tokens += 1
        should_continue_decoding = next_token.item() != tokenizer.eos_token_id
        past_key_values = step_outputs.past_key_values
        current_input_ids = next_token
        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones(
                    (current_attention_mask.shape[0], 1),
                    device=current_attention_mask.device,
                    dtype=current_attention_mask.dtype,
                ),
            ],
            dim=1,
        )
        del step_outputs

    if not decode_step_times_ms:
        ttft_ms = prefill_latency_ms

    decode_time_ms = sum(decode_step_times_ms)
    output_throughput = None
    if decode_time_ms > 0:
        output_throughput = len(decode_step_times_ms) / (decode_time_ms / 1000.0)

    del past_key_values
    torch.cuda.synchronize()

    return SampleMetrics(
        context_tokens=context_tokens,
        ttft_ms=ttft_ms,
        output_throughput_toks_per_s=output_throughput,
        prefill_kv_memory_mb=prefill_kv_memory_mb,
        cache_size_mb=cache_size_mb,
    )


def summarize_basic(results: List[SampleMetrics]) -> Dict[str, Any]:
    def values(name: str) -> List[float]:
        vals = []
        for result in results:
            value = getattr(result, name)
            if value is not None:
                vals.append(value)
        return vals

    def mean_or_none(name: str) -> Optional[float]:
        vals = values(name)
        return float(statistics.mean(vals)) if vals else None

    return {
        "mean_ttft_ms": mean_or_none("ttft_ms"),
        "mean_output_throughput_toks_per_s": mean_or_none("output_throughput_toks_per_s"),
        "mean_prefill_kv_memory_mb": mean_or_none("prefill_kv_memory_mb"),
        "mean_cache_size_mb": mean_or_none("cache_size_mb"),
    }


def get_context_length_buckets() -> List[Tuple[str, int, Optional[int]]]:
    return [
        ("0_4k", 0, 4000),
        ("4k_8k", 4000, 8000),
        ("8k_plus", 8000, None),
    ]


def summarize(results: List[SampleMetrics]) -> Dict[str, Any]:
    summary = summarize_basic(results)
    bucket_summaries: Dict[str, Any] = {}

    for bucket_name, lower, upper in get_context_length_buckets():
        bucket_results = [
            result
            for result in results
            if result.context_tokens >= lower and (upper is None or result.context_tokens < upper)
        ]
        bucket_summary = summarize_basic(bucket_results)
        bucket_summary["min_context_tokens"] = (
            min(result.context_tokens for result in bucket_results) if bucket_results else None
        )
        bucket_summary["max_context_tokens"] = (
            max(result.context_tokens for result in bucket_results) if bucket_results else None
        )
        bucket_summaries[bucket_name] = bucket_summary

    summary["context_length_buckets"] = bucket_summaries
    return summary


def summarize_by_dataset(results_by_dataset: Dict[str, List[SampleMetrics]]) -> Dict[str, Any]:
    return {
        dataset_name: summarize(dataset_results) if dataset_results else {}
        for dataset_name, dataset_results in results_by_dataset.items()
    }


def get_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
            }
        )
    return info


def build_output_paths(args, resolved_model_name: str, effective_bits: float) -> Dict[str, str]:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bit_label = f"{effective_bits:g}bits" if effective_bits != args.bit_width else f"{args.bit_width}bits"
    packing_label = "_packed" if args.use_packing else ""
    run_dir = os.path.join(
        args.output_dir,
        f"{model_tag(resolved_model_name)}_turboquant_{bit_label}{packing_label}_{args.dtype}_{run_timestamp}",
    )
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_timestamp": run_timestamp,
        "run_dir": run_dir,
        "samples_jsonl": os.path.join(run_dir, "all_datasets.jsonl"),
        "results_json": os.path.join(run_dir, "results.json"),
    }


def main():
    parser = argparse.ArgumentParser(description="System performance benchmark for turboquant_445")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cached_models")
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    parser.add_argument("--dataset_names", type=str, nargs="+", default=list(DEFAULT_DATASET_NAMES))
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_end", type=int, default=None)
    parser.add_argument("--sample_indices", type=int, nargs="+", default=None)

    parser.add_argument("--bit_width", type=int, default=3)
    parser.add_argument("--num_outlier_channels", type=int, default=None)
    parser.add_argument("--outlier_bits", type=int, default=None)
    parser.add_argument("--use_packing", action="store_true")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    model, tokenizer, resolved_model_name = build_model_and_tokenizer(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        dtype_str=args.dtype,
    )

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    auto_outliers, auto_outlier_bits = get_auto_outlier_channels(args.bit_width, head_dim)
    num_outlier_channels = auto_outliers if args.num_outlier_channels is None else args.num_outlier_channels
    outlier_bits = auto_outlier_bits if args.outlier_bits is None else args.outlier_bits
    effective_bits = (
        ((head_dim - num_outlier_channels) * args.bit_width + num_outlier_channels * outlier_bits) / head_dim
        if num_outlier_channels > 0 and outlier_bits > args.bit_width
        else float(args.bit_width)
    )

    dataset_max_new_tokens = load_dataset_max_new_tokens()
    dataset_names = normalize_dataset_names(args.dataset_names)
    unknown_datasets = sorted(set(dataset_names) - set(dataset_max_new_tokens))
    if unknown_datasets:
        raise ValueError(f"Unknown dataset names requested: {unknown_datasets}")

    output_paths = build_output_paths(args, resolved_model_name, effective_bits)

    print("=" * 72)
    print("  System Performance Benchmark - TurboQuant 4.45.2")
    print("=" * 72)
    print(f"  Model:          {resolved_model_name}")
    print(f"  Bit width:      {args.bit_width}")
    print(f"  Outlier config: {num_outlier_channels} channels @ {outlier_bits} bits")
    print(f"  Effective bits: {effective_bits:g}")
    print(f"  Dtype:          {args.dtype}")
    print(f"  Output:         {output_paths['run_dir']}")
    print()

    results: List[SampleMetrics] = []
    results_by_dataset: Dict[str, List[SampleMetrics]] = {name: [] for name in dataset_names}
    sample_records: List[Dict[str, Any]] = []
    skipped_oom = 0
    warmup_done = False

    for dataset_name in dataset_names:
        prompts = load_prompts(
            dataset_name=dataset_name,
            dataset_path=args.dataset_path,
            tokenizer=tokenizer,
            model_name=resolved_model_name,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        if not prompts:
            raise ValueError(f"No prompts were loaded for dataset={dataset_name}")

        if args.sample_indices is not None:
            selected_indices = set(args.sample_indices)
            prompts = [prompt for i, prompt in enumerate(prompts) if i in selected_indices]
        else:
            prompts = prompts[args.sample_start:args.sample_end]

        if not prompts:
            raise ValueError(f"No prompts remain after filtering for dataset={dataset_name}")

        max_new_tokens = args.max_new_tokens or dataset_max_new_tokens[dataset_name]

        if not warmup_done:
            _ = benchmark_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompts[0]["prompt"],
                max_new_tokens=min(8, max_new_tokens),
                context_tokens=prompts[0]["context_tokens"],
                bit_width=args.bit_width,
                num_outlier_channels=num_outlier_channels,
                outlier_bits=outlier_bits,
                use_packing=args.use_packing,
            )
            warmup_done = True

        for i, prompt_record in enumerate(prompts):
            prompt = prompt_record["prompt"]
            try:
                metrics = benchmark_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    context_tokens=prompt_record["context_tokens"],
                    bit_width=args.bit_width,
                    num_outlier_channels=num_outlier_channels,
                    outlier_bits=outlier_bits,
                    use_packing=args.use_packing,
                )
                results.append(metrics)
                results_by_dataset[dataset_name].append(metrics)
                sample_record = {
                    "index": i,
                    "dataset": dataset_name,
                    "max_new_tokens": max_new_tokens,
                    "prompt": prompt,
                    "status": "ok",
                    **asdict(metrics),
                }
                sample_records.append(sample_record)

                out_tput_str = (
                    f"{metrics.output_throughput_toks_per_s:.2f}"
                    if metrics.output_throughput_toks_per_s is not None
                    else "NA"
                )
                print(
                    f"[{dataset_name} {i + 1}/{len(prompts)}] "
                    f"ttft={metrics.ttft_ms:.2f} ms | "
                    f"out_tput={out_tput_str} tok/s | "
                    f"prefill_kv_mem={metrics.prefill_kv_memory_mb:.2f} MB | "
                    f"cache_size={metrics.cache_size_mb:.2f} MB"
                )

            except torch.cuda.OutOfMemoryError:
                skipped_oom += 1
                sample_records.append(
                    {
                        "index": i,
                        "dataset": dataset_name,
                        "max_new_tokens": max_new_tokens,
                        "prompt": prompt,
                        "status": "oom",
                    }
                )
                print(f"[{dataset_name} {i + 1}/{len(prompts)}] CUDA OOM, skipping sample.")
                torch.cuda.empty_cache()

    summary = {
        "overall": summarize(results) if results else {},
        "by_dataset": summarize_by_dataset(results_by_dataset),
    }
    payload = {
        "config": {
            **vars(args),
            "resolved_model_name": resolved_model_name,
            "effective_bits": effective_bits,
            "head_dim": head_dim,
            "num_hidden_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "resolved_num_outlier_channels": num_outlier_channels,
            "resolved_outlier_bits": outlier_bits,
        },
        "run_timestamp": output_paths["run_timestamp"],
        "datasets": list(dataset_names),
        "dataset_max_new_tokens": dataset_max_new_tokens,
        "system_info": get_system_info(),
        "summary": summary,
        "num_skipped_oom": skipped_oom,
        "samples": sample_records,
    }

    with open(output_paths["samples_jsonl"], "w", encoding="utf-8") as f:
        for sample_record in sample_records:
            json.dump(sample_record, f, ensure_ascii=False)
            f.write("\n")

    with open(output_paths["results_json"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved per-sample results to: {output_paths['samples_jsonl']}")
    print(f"Saved summary results to: {output_paths['results_json']}")


if __name__ == "__main__":
    main()
