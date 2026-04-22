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
|   500 | 10 |   82.7ms |     24.0ms | MB 3.45× |
|  2000 | 20 |  114.7ms |     58.6ms | MB 1.96× |
| 10000 | 50 |  259.9ms |    208.5ms | MB 1.25× |
| 50000 | 50 |  434.7ms |    668.5ms | CB 1.54× |

**Training: Binary classification (Logloss)**

|     n |  p | CatBoost | MichiBoost |    Ratio |
| ----: | -: | -------: | ---------: | -------: |
|   500 | 10 |   88.8ms |     23.4ms | MB 3.79× |
|  2000 | 20 |  124.0ms |     61.6ms | MB 2.01× |
| 10000 | 50 |  307.8ms |    220.8ms | MB 1.39× |
| 50000 | 50 |  532.5ms |    710.6ms | CB 1.33× |

**Training: Multiclass (3 classes)**

|     n |  p | CatBoost | MichiBoost |    Ratio |
| ----: | -: | -------: | ---------: | -------: |
|   500 | 10 |  126.2ms |     82.6ms | MB 1.53× |
|  2000 | 20 |  215.0ms |    157.9ms | MB 1.36× |
| 10000 | 50 |  548.0ms |    719.0ms | CB 1.31× |
| 50000 | 50 |  953.4ms |   2547.2ms | CB 2.67× |

**Training: Categorical (n=5000, 5 categorical + 5 numerical features)**

| Cardinality | CatBoost | MichiBoost |    Ratio |
| ----------: | -------: | ---------: | -------: |
|           5 |  126.5ms |    135.2ms | CB 1.07× |
|          50 |  120.9ms |    133.5ms | CB 1.10× |
|         500 |  122.0ms |     83.3ms | MB 1.46× |

**Inference: batch-size sweep (regression, n_train=10000, p=20)**

| Batch | CatBoost | MichiBoost |     Ratio |
| ----: | -------: | ---------: | --------: |
|     1 |  0.223ms |    0.008ms | MB 28.45× |
|   100 |  0.246ms |    0.025ms |  MB 9.68× |
|  1000 |  0.354ms |    0.080ms |  MB 4.44× |
| 10000 |  0.996ms |    2.055ms |  CB 2.06× |

### Scaling (Covertype)

Covertype subset: 50,000 samples (40k train / 10k test), 54 features, 7 classes. 100 iterations, depth 6, learning_rate 0.03. Test accuracy: 74.07% (MichiBoost) / 71.82% (CatBoost).

**Training time and self-speedup**

| Threads | MichiBoost | MB speedup vs 1t | CatBoost | CB speedup vs 1t |    Ratio |
| ------: | ---------: | ---------------: | -------: | ---------------: | -------: |
|       1 |    7597.0ms |            1.00× | 1774.4ms |            1.00× | CB 4.28× |
|       2 |    4293.0ms |            1.77× |  959.5ms |            1.85× | CB 4.47× |
|       4 |    2682.8ms |            2.83× |  597.3ms |            2.97× | CB 4.49× |
|       8 |    2062.5ms |            3.68× |  502.0ms |            3.54× | CB 4.11× |

**Inference time (median ms)**

| Threads | MichiBoost | CatBoost |
| ------: | ---------: | -------: |
|       1 |      8.020 |    2.482 |
|       2 |      7.215 |    2.453 |
|       4 |      6.543 |    2.466 |
|       8 |      5.433 |    2.482 |

### Feature costs

Fixture: `n_train=1600`, `n_features=20`, default hyperparameters (100 iters, depth 6, learning_rate 0.03).

4 threads. Fixture: `n_train=1600`, `n_features=20`, default hyperparameters (100 iters, depth 6, learning_rate 0.03).

| Feature                           | CatBoost | MichiBoost |    Ratio |
| --------------------------------- | -------: | ---------: | -------: |
| Cross-validation (5-fold, RMSE)   | 1107.4ms |    251.2ms | MB 4.41× |
| Cross-validation (5-fold, LogLoss)| 1281.3ms |    243.1ms | MB 5.27× |
| Early stopping (max 200, pat=20)  |  206.0ms |     90.1ms | MB 2.29× |
| SHAP values (n_te=400)            |   12.9ms |      2.9ms | MB 4.51× |

| Internal toggle (MichiBoost only) | Baseline | Variant |      Δ |
| --------------------------------- | -------: | ------: | -----: |
| RSM: rsm=1.0 → rsm=0.5             |   45.2ms |  30.1ms | −33.5% |
| Sample weights: none → uniform     |   46.4ms |  45.9ms |  −1.2% |

Save/load: `save` 0.1ms, `load` 0.2ms (Julia `Serialization`).

## Caveats

- **CatBoost has Python/PythonCall overhead** on every call boundary. For very small datasets and single-row inference this dominates; for large training runs it's negligible. The comparison is still meaningful — what a user actually pays when calling either library from Julia — but the micro-benchmark regime flatters MichiBoost on tiny inputs.
- **Synthetic data is easy.** Three of the four benchmarks use it for controllability, but it lets both libraries reach near-perfect accuracy where real data wouldn't. `scaling.jl` runs on real Covertype to cross-check.
- **Metrics, not agreement.** Correctness uses task metrics (RMSE, AUC, log-loss) on held-out data rather than output correlation with CatBoost. A model can agree with CatBoost 85% of the time and still be bad.
- **Reproducibility.** All scripts seed `MersenneTwister(SEED=9)` and pass `random_seed=9` to both libraries, but CatBoost's internal determinism depends on thread count; results at 1 and 8 threads will differ slightly even with the same seed.
