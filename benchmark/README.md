# Benchmarks

Performance and correctness benchmarks for MichiBoost.jl, compared head-to-head against [CatBoost.jl](https://github.com/beacon-biosignals/CatBoost.jl) (a PythonCall wrapper around reference CatBoost). Each script answers one specific question about the library; results from all four together are what back the performance claims in the top-level README.

## Setup

The `benchmark/` directory has its own `Project.toml` that pulls in CatBoost (via PythonCall + CondaPkg), BenchmarkTools, CSV, MLDatasets, and a few other dependencies. From the repo root:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
```

The first run installs the CatBoost Python package via CondaPkg. Subsequent runs skip this.

All benchmarks respect `Threads.nthreads()` and pass the same thread count to CatBoost via `thread_count`, so both libraries run on the same CPU budget. Launch Julia with `-t N` to control it.

## The four questions

| File | Question it answers |
| --- | --- |
| `correctness.jl`  | Is MichiBoost's model quality on par with CatBoost on held-out data? |
| `speed_sweep.jl`  | At what data sizes is MichiBoost faster, and where is the crossover? |
| `scaling.jl`      | How does MichiBoost scale with threads on a real public dataset? |
| `feature_costs.jl`| What does each advertised feature (CV, early stopping, SHAP, RSM, sample weights, save/load) cost? |

Everything else (shared data generators, metric helpers, AUC, formatting) lives in `common.jl`, which each script `include`s.

## Running

### 1. Correctness

Compares held-out task metrics (RMSE, MAE, R², log-loss, AUC, accuracy) for regression, binary, multiclass (k=3 and k=10), and categorical tasks. Assertions require MichiBoost to stay within a fixed tolerance of CatBoost (e.g., `MB_RMSE ≤ 1.15 × CB_RMSE`, `AUC ≥ CB_AUC − 0.05`). This replaces correlation/agreement gates, which can pass while the underlying model is poor.

```bash
julia --project=benchmark -t 4 benchmark/correctness.jl
```

Each section prints a three-column table of MichiBoost / CatBoost / delta, followed by a `Test Passed` or failure from the `@testset`.

### 2. Speed sweep

Training and inference time across a grid of `(n_train, n_features)` sizes for regression, binary, and multiclass, plus a categorical-cardinality sweep and an inference batch-size sweep. The goal is to make the crossover point visible rather than reporting a single number.

```bash
# Full sweep (~5-15 min depending on hardware)
julia --project=benchmark -t 4 benchmark/speed_sweep.jl

# Quick run for dev loops: smaller grid, fewer batch sizes
julia --project=benchmark -t 4 benchmark/speed_sweep.jl --quick

# Single section
julia --project=benchmark -t 4 benchmark/speed_sweep.jl --section=regression
# sections: regression | binary | multiclass | categorical | inference
```

Each row prints `CatBoost ms`, `MichiBoost ms`, and a speedup tag (`MB 2.3x` means MichiBoost was 2.3× faster; `CB 1.5x` means CatBoost won).

### 3. Scaling

Runs a multiclass training + inference benchmark on the UCI [Covertype](https://archive.ics.uci.edu/dataset/31/covertype) dataset (581,012 rows × 54 features, 7 classes). The dataset is downloaded and unzipped into `benchmark/data/` on first run.

```bash
# Single run at the current thread count
julia --project=benchmark -t 4 benchmark/scaling.jl

# Threading curve: spawns one child Julia process per thread count (1, 2, 4, 8)
# and aggregates the results. Use a smaller subset for quicker iteration.
julia --project=benchmark benchmark/scaling.jl --sweep
julia --project=benchmark benchmark/scaling.jl --sweep --n=20000
```

The sweep path spawns fresh child processes rather than changing threads in-process (which Julia does not support), so each measurement starts from a clean thread pool. Each child emits a parseable `RESULT …` line that the parent aggregates.

### 4. Feature costs

Times each advertised feature against a direct comparison where CatBoost has one (CV, early stopping, SHAP) and against a no-op baseline where it does not (RSM toggle, sample-weight overhead, save/load).

```bash
julia --project=benchmark -t 4 benchmark/feature_costs.jl
```

RSM reports the percent change between `rsm=1.0` and `rsm=0.5`; sample-weight overhead reports the percent change vs. unweighted training on the same data.

### 5. Profile (diagnostic, not a benchmark)

`profile.jl` runs a single training under Julia's sampling profiler and allocation tracker. Used to investigate the threading ceiling and per-iteration allocation rate. Output reports wall-time totals, hot functions (across all threads and filtered to the main thread to surface serial regions), and top allocating types.

```bash
julia --project=benchmark -t 4 benchmark/profile.jl
julia --project=benchmark -t 4 benchmark/profile.jl --n=50000 --task=regression
```

Full flat and tree views are written to `benchmark/profile_flat.txt` and `benchmark/profile_tree.txt`.

## Running everything

```bash
for f in correctness speed_sweep scaling feature_costs; do
  julia --project=benchmark -t 4 benchmark/$f.jl
done
```

Total runtime varies with hardware; on a modern 4-core laptop expect ~20-30 minutes for the full set with default options, or a few minutes with `--quick` / smaller `--n=`.

## Results

The tables below will be populated after running the suite. Re-run locally to reproduce; numbers are machine-dependent (CPU, memory bandwidth, thread count, BLAS, CatBoost version).

### Correctness

| Task                              | Metric   | MichiBoost | CatBoost |       Δ |
| --------------------------------- | -------- | ---------: | -------: | ------: |
| Regression (RMSE loss), n=1600×20 | RMSE     |     2.2281 |   2.2686 | -0.0405 |
|                                   | MAE      |     1.7112 |   1.7295 | -0.0182 |
|                                   | R²       |     0.7628 |   0.7542 | +0.0087 |
| Regression (MAE loss), n=1600×20  | MAE      |     1.6486 |   1.7144 | -0.0658 |
|                                   | RMSE     |     2.1820 |   2.2469 | -0.0650 |
| Binary classification, n=3200×15  | LogLoss  |     0.4653 |   0.4702 | -0.0049 |
|                                   | AUC      |     0.8785 |   0.8874 | -0.0089 |
|                                   | Accuracy |     0.7925 |   0.7938 | -0.0012 |
| Multiclass k=3, n=2400×10         | LogLoss  |     0.5541 |   0.7205 | -0.1664 |
|                                   | Accuracy |     0.8033 |   0.8067 | -0.0033 |
| Multiclass k=10, n=4000×15        | LogLoss  |     0.4152 |   0.5644 | -0.1492 |
|                                   | Accuracy |     0.8990 |   0.8690 | +0.0300 |
| Categorical (cardinality=20)      | LogLoss  |     0.6965 |   0.6931 | +0.0035 |
|                                   | AUC      |     0.4954 |   0.5186 | -0.0233 |
|                                   | Accuracy |     0.4925 |   0.5175 | -0.0250 |

### Speed sweep

100 iterations, depth 6, learning_rate 0.03, **4 threads**. Columns report training wall-clock time; `n` is the training subset size.

**Training: Regression (RMSE)**

|     n |  p | CatBoost | MichiBoost |    Ratio |
| ----: | -: | -------: | ---------: | -------: |
|   500 | 10 |   89.2ms |     26.8ms | MB 3.33× |
|  2000 | 20 |  128.6ms |     60.5ms | MB 2.13× |
| 10000 | 50 |  281.4ms |    230.6ms | MB 1.22× |
| 50000 | 50 |  458.0ms |    689.2ms | CB 1.50× |

**Training: Binary classification (Logloss)**

|     n |  p | CatBoost | MichiBoost |    Ratio |
| ----: | -: | -------: | ---------: | -------: |
|   500 | 10 |   84.5ms |     24.0ms | MB 3.52× |
|  2000 | 20 |  141.8ms |     53.6ms | MB 2.65× |
| 10000 | 50 |  323.3ms |    224.9ms | MB 1.44× |
| 50000 | 50 |  517.5ms |    745.5ms | CB 1.44× |

**Training: Multiclass (3 classes)**

|     n |  p | CatBoost | MichiBoost |    Ratio |
| ----: | -: | -------: | ---------: | -------: |
|   500 | 10 |  130.4ms |     64.1ms | MB 2.03× |
|  2000 | 20 |  203.7ms |    138.8ms | MB 1.47× |
| 10000 | 50 |  540.7ms |    662.1ms | CB 1.22× |
| 50000 | 50 |  973.5ms |   2006.5ms | CB 2.06× |

**Training: Categorical (n=5000, 5 categorical + 5 numerical features)**

| Cardinality | CatBoost | MichiBoost |    Ratio |
| ----------: | -------: | ---------: | -------: |
|           5 |  123.6ms |    146.2ms | CB 1.18× |
|          50 |  121.4ms |    144.4ms | CB 1.19× |
|         500 |  135.2ms |     87.9ms | MB 1.54× |

**Inference: batch-size sweep (regression, n_train=10000, p=20)**

| Batch | CatBoost | MichiBoost |     Ratio |
| ----: | -------: | ---------: | --------: |
|     1 |  0.230ms |    0.008ms | MB 29.04× |
|   100 |  0.246ms |    0.025ms |  MB 9.73× |
|  1000 |  0.348ms |    0.085ms |  MB 4.11× |
| 10000 |  0.990ms |    2.046ms |  CB 2.07× |

### Scaling (Covertype)

Covertype subset: 50,000 samples (40k train / 10k test), 54 features, 7 classes. 100 iterations, depth 6, learning_rate 0.03. Test accuracy: 74.07% (MichiBoost) / 71.82% (CatBoost).

**Training time and self-speedup**

| Threads | MichiBoost | MB speedup vs 1t | CatBoost | CB speedup vs 1t |    Ratio |
| ------: | ---------: | ---------------: | -------: | ---------------: | -------: |
|       1 |   6309.0ms |            1.00× | 1911.8ms |            1.00× | CB 3.30× |
|       2 |   3380.2ms |            1.87× | 1026.7ms |            1.86× | CB 3.29× |
|       4 |   2041.6ms |            3.09× |  612.1ms |            3.12× | CB 3.34× |
|       8 |   1655.5ms |            3.81× |  506.7ms |            3.77× | CB 3.27× |

**Inference time (median ms)**

| Threads | MichiBoost | CatBoost |
| ------: | ---------: | -------: |
|       1 |      8.211 |    2.543 |
|       2 |      7.229 |    2.487 |
|       4 |      6.569 |    2.505 |
|       8 |      5.388 |    2.494 |

### Feature costs

Fixture: `n_train=1600`, `n_features=20`, default hyperparameters (100 iters, depth 6, learning_rate 0.03).

4 threads. Fixture: `n_train=1600`, `n_features=20`, default hyperparameters (100 iters, depth 6, learning_rate 0.03).

| Feature                           | CatBoost | MichiBoost |    Ratio |
| --------------------------------- | -------: | ---------: | -------: |
| Cross-validation (5-fold, RMSE)   | 1067.1ms |    274.9ms | MB 3.88× |
| Cross-validation (5-fold, LogLoss)| 1183.2ms |    234.3ms | MB 5.05× |
| Early stopping (max 200, pat=20)  |  224.5ms |     99.8ms | MB 2.25× |
| SHAP values (n_te=400)            |   13.6ms |      4.6ms | MB 2.93× |

| Internal toggle (MichiBoost only) | Baseline | Variant |      Δ |
| --------------------------------- | -------: | ------: | -----: |
| RSM: rsm=1.0 → rsm=0.5             |   50.9ms |  31.0ms | −39.1% |
| Sample weights: none → uniform     |   52.5ms |  54.1ms |  +2.9% |

Save/load: `save` 0.1ms, `load` 0.2ms (Julia `Serialization`).

## Caveats

- **CatBoost has Python/PythonCall overhead** on every call boundary. For very small datasets and single-row inference this dominates; for large training runs it's negligible. The comparison is still meaningful — what a user actually pays when calling either library from Julia — but the micro-benchmark regime flatters MichiBoost on tiny inputs.
- **Synthetic data is easy.** Three of the four benchmarks use it for controllability, but it lets both libraries reach near-perfect accuracy where real data wouldn't. `scaling.jl` runs on real Covertype to cross-check.
- **Metrics, not agreement.** Correctness uses task metrics (RMSE, AUC, log-loss) on held-out data rather than output correlation with CatBoost. A model can agree with CatBoost 85% of the time and still be bad.
- **Reproducibility.** All scripts seed `MersenneTwister(SEED=9)` and pass `random_seed=9` to both libraries, but CatBoost's internal determinism depends on thread count; results at 1 and 8 threads will differ slightly even with the same seed.
