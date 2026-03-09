# Benchmark scripts

Scripts for the wall-clock time and peak incremental resident set size (RSS) benchmarks reported in the paper.

## Scripts

| Script | Measures | Method |
| ------ | -------- | ------ |
| `bench_time_predict.py` | Wall-clock time | aimz `predict` |
| `bench_time_predictive.py` | Wall-clock time | NumPyro `Predictive` |
| `bench_memory_predict.py` | Peak incremental RSS | aimz `predict` |
| `bench_memory_predictive.py` | Peak incremental RSS | NumPyro `Predictive` |

## Environment

- Python 3.13.2
- aimz v0.10.0
- flax v0.12.5
- optax v0.2.6

All benchmarks were run on an **AWS EC2 g5.24xlarge** instance.

## Usage

Each script accepts `--n` to set the dataset size. The `predict` scripts also accept `--output-dir` for the Zarr store location.

```bash
# Wall-clock time
python bench_time_predict.py --n 1000000
python bench_time_predictive.py --n 1000000

# Peak incremental RSS
python bench_memory_predict.py --n 1000000
python bench_memory_predictive.py --n 1000000
```

> **Note:** NumPyro `Predictive` scripts will run out of memory for large $n$ (typically $n \geq 10\text{M}$).
