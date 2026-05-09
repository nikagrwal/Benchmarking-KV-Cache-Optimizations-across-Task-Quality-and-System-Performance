import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from transformers import DynamicCache

from benchmarking_snapkv_kvpress import (
    DATASET_NAMES,
    SampleMetrics,
    build_model_and_tokenizer,
    cleanup_cuda,
    compute_cache_size_mb,
    compute_last_token_logits,
    cuda_time_ms,
    get_model_input_device,
    is_cuda_oom,
    load_dataset_max_new_tokens,
    load_prompts,
    model_tag,
    summarize,
    summarize_by_dataset,
    get_system_info,
)
from kvpress import CAMPress, KnormPress, SnapKVPress, StreamingLLMPress, TOVAPress


def build_cam_press(args):
    if args.press_name == "cam_knorm":
        base_press = KnormPress()
    elif args.press_name == "cam_snapkv":
        base_press = SnapKVPress(window_size=args.window_size, kernel_size=args.kernel_size)
    elif args.press_name == "cam_streaming_llm":
        base_press = StreamingLLMPress()
    elif args.press_name == "cam_tova":
        base_press = TOVAPress()
    else:
        raise ValueError(f"Unsupported press_name={args.press_name}")

    return CAMPress(
        base_press=base_press,
        compression_interval=args.compression_interval,
        target_size=args.target_size,
        hidden_states_buffer_size=args.hidden_states_buffer_size,
        merge_budget=args.merge_budget,
    )


@torch.inference_mode()
def benchmark_one(
    model,
    tokenizer,
    press: CAMPress,
    prompt: str,
    max_new_tokens: int,
    context_tokens: int,
) -> SampleMetrics:
    cleanup_cuda()
    press.reset()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = get_model_input_device(model)
    input_ids = inputs["input_ids"].to(input_device)
    prompt_length = input_ids.shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(input_device)
    cache = DynamicCache()

    with press(model):
        def do_prefill():
            return model.model(
                input_ids=input_ids,
                past_key_values=cache,
                return_dict=True,
                use_cache=True,
            )

        decoder_outputs, prefill_latency_ms = cuda_time_ms(do_prefill)
        last_hidden_state = decoder_outputs[0][:, -1:, :].contiguous()
        del decoder_outputs

        torch.cuda.synchronize()
        mem_after_prefill = torch.cuda.memory_allocated(input_device)
        prefill_kv_memory_mb = max(0, mem_after_prefill - mem_before) / (1024**2)

        def pick_first_token():
            logits = compute_last_token_logits(model, last_hidden_state)
            return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        first_token, _ = cuda_time_ms(pick_first_token)

        generated_tokens = 1
        current_input_ids = first_token
        current_position_ids = torch.tensor([[prompt_length]], device=input_device, dtype=torch.long)
        decode_step_times_ms: List[float] = []
        eos_token_ids = tokenizer.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = []
        elif not isinstance(eos_token_ids, list):
            eos_token_ids = [eos_token_ids]
        should_continue_decoding = int(first_token.item()) not in eos_token_ids

        if max_new_tokens > 1 and should_continue_decoding:
            def do_decode():
                return model(
                    input_ids=current_input_ids,
                    past_key_values=cache,
                    position_ids=current_position_ids,
                    use_cache=True,
                    return_dict=True,
                )

            step_outputs, first_decode_ms = cuda_time_ms(do_decode)
            decode_step_times_ms.append(first_decode_ms)
            ttft_ms = prefill_latency_ms + first_decode_ms

            next_token = torch.argmax(step_outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens += 1
            should_continue_decoding = int(next_token.item()) not in eos_token_ids
            current_input_ids = next_token
            current_position_ids = current_position_ids + 1
        else:
            ttft_ms = prefill_latency_ms

        for _ in range(max(0, max_new_tokens - generated_tokens)):
            if not should_continue_decoding:
                break

            def do_decode():
                return model(
                    input_ids=current_input_ids,
                    past_key_values=cache,
                    position_ids=current_position_ids,
                    use_cache=True,
                    return_dict=True,
                )

            step_outputs, step_ms = cuda_time_ms(do_decode)
            decode_step_times_ms.append(step_ms)

            next_token = torch.argmax(step_outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens += 1
            should_continue_decoding = int(next_token.item()) not in eos_token_ids
            current_input_ids = next_token
            current_position_ids = current_position_ids + 1

    decode_time_ms = sum(decode_step_times_ms)
    output_throughput = None
    if decode_time_ms > 0 and decode_step_times_ms:
        output_throughput = len(decode_step_times_ms) / (decode_time_ms / 1000.0)

    cache_size_mb = compute_cache_size_mb(cache)
    compressed_cache_tokens = cache.get_seq_length() if hasattr(cache, "get_seq_length") else None

    del last_hidden_state
    del cache
    torch.cuda.synchronize()

    return SampleMetrics(
        context_tokens=context_tokens,
        ttft_ms=ttft_ms,
        output_throughput_toks_per_s=output_throughput,
        prefill_kv_memory_mb=prefill_kv_memory_mb,
        cache_size_mb=cache_size_mb,
        compressed_cache_tokens=compressed_cache_tokens,
    )


def build_output_paths(args) -> Dict[str, str]:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extra = ""
    if args.press_name == "cam_snapkv":
        extra = f"_win{args.window_size}_kernel{args.kernel_size}"
    run_dir = os.path.join(
        args.output_dir,
        (
            f"{model_tag(args.model_name)}_kvpress_{args.press_name}"
            f"_target{args.target_size}_interval{args.compression_interval}{extra}_{args.dtype}_{run_timestamp}"
        ),
    )
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_timestamp": run_timestamp,
        "run_dir": run_dir,
        "samples_jsonl": os.path.join(run_dir, "all_datasets.jsonl"),
        "results_json": os.path.join(run_dir, "results.json"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cached_models")
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--dataset_path", type=str, default="zai-org/LongBench")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    parser.add_argument("--skip_warmup", action="store_true")
    parser.add_argument("--dataset_names", type=str, nargs="+", default=list(DATASET_NAMES))
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_end", type=int, default=None)
    parser.add_argument("--sample_indices", type=int, nargs="+", default=None)

    parser.add_argument(
        "--press_name",
        type=str,
        default="cam_knorm",
        choices=["cam_knorm", "cam_snapkv", "cam_streaming_llm", "cam_tova"],
    )
    parser.add_argument("--compression_interval", type=int, default=32)
    parser.add_argument("--target_size", type=int, default=1024)
    parser.add_argument("--hidden_states_buffer_size", type=int, default=64)
    parser.add_argument("--merge_budget", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if args.press_name == "cam_snapkv" and args.compression_interval <= args.window_size:
        raise ValueError("--compression_interval must be greater than --window_size for cam_snapkv")

    model, tokenizer = build_model_and_tokenizer(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        dtype_str=args.dtype,
    )
    press = build_cam_press(args)

    dataset_max_new_tokens = load_dataset_max_new_tokens()
    dataset_names = tuple(args.dataset_names)
    unknown_datasets = sorted(set(dataset_names) - set(DATASET_NAMES))
    if unknown_datasets:
        raise ValueError(f"Unknown dataset names requested: {unknown_datasets}")
    output_paths = build_output_paths(args)

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
            model_name=args.model_name,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        if args.sample_indices is not None:
            selected_indices = set(args.sample_indices)
            prompts = [prompt for i, prompt in enumerate(prompts) if i in selected_indices]
        else:
            prompts = prompts[args.sample_start : args.sample_end]

        if not prompts:
            raise ValueError(f"No prompts remain after filtering for dataset={dataset_name}")

        max_new_tokens = args.max_new_tokens or dataset_max_new_tokens[dataset_name]

        if not warmup_done and not args.skip_warmup:
            try:
                _ = benchmark_one(
                    model,
                    tokenizer,
                    press,
                    prompts[0]["prompt"],
                    max_new_tokens=min(max(args.compression_interval + 1, 8), max_new_tokens),
                    context_tokens=prompts[0]["context_tokens"],
                )
                warmup_done = True
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if not is_cuda_oom(error):
                    raise
                print("[warmup] CUDA OOM, skipping warmup and continuing with measured samples.")
                cleanup_cuda()
                warmup_done = True

        for i, prompt_record in enumerate(prompts):
            prompt = prompt_record["prompt"]
            try:
                metrics = benchmark_one(
                    model,
                    tokenizer,
                    press,
                    prompt,
                    max_new_tokens,
                    context_tokens=prompt_record["context_tokens"],
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
                    f"cache={metrics.cache_size_mb:.2f} MB | "
                    f"compressed_tokens={metrics.compressed_cache_tokens}"
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if not is_cuda_oom(error):
                    raise
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
                cleanup_cuda()

    summary = {
        "overall": summarize(results) if results else {},
        "by_dataset": summarize_by_dataset(results_by_dataset),
    }
    payload = {
        "config": vars(args),
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
