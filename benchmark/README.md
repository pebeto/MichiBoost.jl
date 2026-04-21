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

100 iterations, depth 6, learning_rate 0.03, single thread. Columns report training wall-clock time; `n` is the training subset size.

**Training: Regression (RMSE)**

|     n |  p | CatBoost | MichiBoost |    Ratio |
| ----: | -: | -------: | ---------: | -------: |
|   500 | 10 |  151.5ms |     58.5ms | MB 2.59× |
|  2000 | 20 |  274.2ms |    157.0ms | MB 1.75× |
| 10000 | 50 |  714.8ms |    646.1ms | MB 1.11× |
| 50000 | 50 | 1349.7ms |   1941.9ms | CB 1.44× |

**Training: Binary classification (Logloss)**

|     n |  p | CatBoost | MichiBoost |    Ratio |
| ----: | -: | -------: | ---------: | -------: |
|   500 | 10 |  156.2ms |     57.7ms | MB 2.71× |
|  2000 | 20 |  297.7ms |    153.5ms | MB 1.94× |
| 10000 | 50 |  780.3ms |    672.5ms | MB 1.16× |
| 50000 | 50 | 1518.4ms |   2077.4ms | CB 1.37× |

**Training: Multiclass (3 classes)**

|     n |  p | CatBoost | MichiBoost |    Ratio |
| ----: | -: | -------: | ---------: | -------: |
|   500 | 10 |  292.9ms |    213.0ms | MB 1.38× |
|  2000 | 20 |  566.2ms |    555.6ms | MB 1.02× |
| 10000 | 50 | 1665.1ms |   2375.7ms | CB 1.43× |
| 50000 | 50 | 3140.8ms |   7322.6ms | CB 2.33× |

**Training: Categorical (n=5000, 5 categorical + 5 numerical features)**

| Cardinality | CatBoost | MichiBoost |    Ratio |
| ----------: | -------: | ---------: | -------: |
|           5 |  231.2ms |   1405.1ms | CB 6.08× |
|          50 |  208.7ms |    719.8ms | CB 3.45× |
|         500 |  208.3ms |    177.5ms | MB 1.17× |

**Inference: batch-size sweep (regression, n_train=10000, p=20)**

| Batch | CatBoost | MichiBoost |     Ratio |
| ----: | -------: | ---------: | --------: |
|     1 |  0.232ms |    0.009ms | MB 27.23× |
|   100 |  0.248ms |    0.029ms |  MB 8.59× |
|  1000 |  0.364ms |    0.691ms |  CB 1.90× |
| 10000 |  0.978ms |    7.028ms |  CB 7.19× |

### Scaling (Covertype)

Covertype subset: 50,000 samples (40k train / 10k test), 54 features, 7 classes. 100 iterations, depth 6, learning_rate 0.03. Test accuracy: 74.07% (MichiBoost) / 71.82% (CatBoost).

**Training time and self-speedup**

| Threads | MichiBoost | MB speedup vs 1t | CatBoost | CB speedup vs 1t |     Ratio |
| ------: | ---------: | ---------------: | -------: | ---------------: | --------: |
|       1 |    7590.7ms |            1.00× | 1831.2ms |            1.00× |  CB 4.15× |
|       2 |    4388.3ms |            1.73× | 1008.5ms |            1.82× |  CB 4.35× |
|       4 |    2962.2ms |            2.56× |  615.5ms |            2.98× |  CB 4.81× |
|       8 |    2133.4ms |            3.56× |  521.7ms |            3.51× |  CB 4.09× |

**Inference time (median ms)**

| Threads | MichiBoost | CatBoost |
| ------: | ---------: | -------: |
|       1 |      8.128 |    2.489 |
|       2 |      7.040 |    2.512 |
|       4 |      6.209 |    2.515 |
|       8 |      6.298 |    2.484 |

### Feature costs

Fixture: `n_train=1600`, `n_features=20`, default hyperparameters (100 iters, depth 6, learning_rate 0.03).

| Feature                           | CatBoost | MichiBoost |    Ratio |
| --------------------------------- | -------: | ---------: | -------: |
| Cross-validation (5-fold, RMSE)   |  973.1ms |    677.5ms | MB 1.44× |
| Cross-validation (5-fold, LogLoss)| 1221.3ms |    698.0ms | MB 1.75× |
| Early stopping (max 200, pat=20)  |  502.6ms |    270.9ms | MB 1.86× |
| SHAP values (n_te=400)            |   14.0ms |     10.0ms | MB 1.41× |

| Internal toggle (MichiBoost only) | Baseline | Variant |      Δ |
| --------------------------------- | -------: | ------: | -----: |
| RSM: rsm=1.0 → rsm=0.5             |  139.9ms |  74.7ms | −46.6% |
| Sample weights: none → uniform     |  138.6ms | 140.8ms |  +1.6% |

Save/load: `save` 0.1ms, `load` 0.2ms (Julia `Serialization`).

## Caveats

- **CatBoost has Python/PythonCall overhead** on every call boundary. For very small datasets and single-row inference this dominates; for large training runs it's negligible. The comparison is still meaningful — what a user actually pays when calling either library from Julia — but the micro-benchmark regime flatters MichiBoost on tiny inputs.
- **Synthetic data is easy.** Three of the four benchmarks use it for controllability, but it lets both libraries reach near-perfect accuracy where real data wouldn't. `scaling.jl` runs on real Covertype to cross-check.
- **Metrics, not agreement.** Correctness uses task metrics (RMSE, AUC, log-loss) on held-out data rather than output correlation with CatBoost. A model can agree with CatBoost 85% of the time and still be bad.
- **Reproducibility.** All scripts seed `MersenneTwister(SEED=9)` and pass `random_seed=9` to both libraries, but CatBoost's internal determinism depends on thread count; results at 1 and 8 threads will differ slightly even with the same seed.
