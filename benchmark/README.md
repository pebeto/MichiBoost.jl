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

_Latest run:_ _(pending)_
_Hardware:_ _(pending)_
_Threads:_ _(pending)_

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
| Categorical (cardinality=20)      | LogLoss  |     0.6943 |   0.6931 | +0.0013 |
|                                   | AUC      |     0.5080 |   0.5186 | -0.0106 |
|                                   | Accuracy |     0.5062 |   0.5175 | -0.0112 |

### Speed sweep

100 iterations, depth 6, learning_rate 0.03, 4 threads. Columns report training wall-clock time; `n` is the training subset size.

**Training: Regression (RMSE)**

|     n |  p | CatBoost | MichiBoost |    Ratio |
| ----: | -: | -------: | ---------: | -------: |
|   500 | 10 |  149.4ms |     65.8ms | MB 2.27× |
|  2000 | 20 |  270.7ms |    173.4ms | MB 1.56× |
| 10000 | 50 |  706.8ms |    712.6ms | CB 1.01× |
| 50000 | 50 | 1332.1ms |   2505.5ms | CB 1.88× |

**Training: Binary classification (Logloss)**

|     n |  p | CatBoost | MichiBoost |    Ratio |
| ----: | -: | -------: | ---------: | -------: |
|   500 | 10 |  152.6ms |     64.0ms | MB 2.38× |
|  2000 | 20 |  282.1ms |    173.4ms | MB 1.63× |
| 10000 | 50 |  774.5ms |    735.8ms | MB 1.05× |
| 50000 | 50 | 1532.5ms |   2610.1ms | CB 1.70× |

**Training: Multiclass (3 classes)**

|     n |  p | CatBoost | MichiBoost |    Ratio |
| ----: | -: | -------: | ---------: | -------: |
|   500 | 10 |  293.2ms |    225.0ms | MB 1.30× |
|  2000 | 20 |  570.2ms |    612.2ms | CB 1.07× |
| 10000 | 50 | 1672.2ms |   2846.6ms | CB 1.70× |
| 50000 | 50 | 3263.8ms |   9590.3ms | CB 2.94× |

**Training: Categorical (n=5000, 5 categorical + 5 numerical features)**

| Cardinality | CatBoost | MichiBoost |    Ratio |
| ----------: | -------: | ---------: | -------: |
|           5 |  233.2ms |   1423.8ms | CB 6.10× |
|          50 |  208.8ms |    790.8ms | CB 3.79× |
|         500 |  208.5ms |    235.1ms | CB 1.13× |

**Inference: batch-size sweep (regression, n_train=10000, p=20)**

| Batch | CatBoost | MichiBoost |     Ratio |
| ----: | -------: | ---------: | --------: |
|     1 |  0.226ms |    0.010ms | MB 22.49× |
|   100 |  0.244ms |    0.032ms |  MB 7.51× |
|  1000 |  0.347ms |    0.690ms |  CB 1.99× |
| 10000 |  1.015ms |    6.961ms |  CB 6.86× |

### Scaling (Covertype)

Covertype subset: 50,000 samples (40k train / 10k test), 54 features, 7 classes. 100 iterations, depth 6, learning_rate 0.03. Test accuracy: 74.07% (MichiBoost) / 71.82% (CatBoost).

Single-thread run:

| Metric          |   CatBoost | MichiBoost |    Ratio |
| --------------- | ---------: | ---------: | -------: |
| Training        |   1837.2ms |  10463.5ms | CB 5.70× |
| Inference       |    2.482ms |    8.759ms | CB 3.53× |
| Test accuracy   |     71.82% |     74.07% |   +2.25pp |

Threading curve (`--sweep`): _pending re-run after the latest perf changes._

### Feature costs

Fixture: `n_train=1600`, `n_features=20`, default hyperparameters (100 iters, depth 6, learning_rate 0.03).

| Feature                           | CatBoost | MichiBoost |    Ratio |
| --------------------------------- | -------: | ---------: | -------: |
| Cross-validation (5-fold, RMSE)   | 1100.3ms |    734.5ms | MB 1.50× |
| Cross-validation (5-fold, LogLoss)| 1224.9ms |    766.2ms | MB 1.60× |
| Early stopping (max 200, pat=20)  |  497.2ms |    303.2ms | MB 1.64× |
| SHAP values (n_te=400)            |   14.0ms |     10.2ms | MB 1.38× |

| Internal toggle (MichiBoost only) | Baseline | Variant |      Δ |
| --------------------------------- | -------: | ------: | -----: |
| RSM: rsm=1.0 → rsm=0.5             |  154.8ms |  90.7ms | −41.4% |
| Sample weights: none → uniform     |  155.2ms | 155.1ms |  −0.1% |

Save/load: `save` 0.1ms, `load` 0.2ms (Julia `Serialization`).

## Caveats

- **CatBoost has Python/PythonCall overhead** on every call boundary. For very small datasets and single-row inference this dominates; for large training runs it's negligible. The comparison is still meaningful — what a user actually pays when calling either library from Julia — but the micro-benchmark regime flatters MichiBoost on tiny inputs.
- **Synthetic data is easy.** Three of the four benchmarks use it for controllability, but it lets both libraries reach near-perfect accuracy where real data wouldn't. `scaling.jl` runs on real Covertype to cross-check.
- **Metrics, not agreement.** Correctness uses task metrics (RMSE, AUC, log-loss) on held-out data rather than output correlation with CatBoost. A model can agree with CatBoost 85% of the time and still be bad.
- **Reproducibility.** All scripts seed `MersenneTwister(SEED=9)` and pass `random_seed=9` to both libraries, but CatBoost's internal determinism depends on thread count; results at 1 and 8 threads will differ slightly even with the same seed.
