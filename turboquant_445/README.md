# TurboQuant 4.45.2

This is a side-by-side TurboQuant rewrite targeting `transformers==4.45.2`.

## What changed

- Replaced the old `DynamicLayer`-based cache integration with a `DynamicCache`-native implementation.
- Kept the original quantization math from the existing TurboQuant project.
- Left the original `turboquant/` folder untouched.

## Usage

From this folder, run:

```bash
PYTHONPATH=/home/nikita/benchmarking-kv-cache/turboquant_445 python -m benchmarks.local
```
