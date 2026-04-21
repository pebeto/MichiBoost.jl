# Training/inference time across a grid of dataset sizes and feature counts.
# Answers "at what scale is MichiBoost faster than CatBoost, and where is the
# crossover point?" — the README claim only holds at small sizes, so the point
# is to make the boundary visible.
#
#   julia --project=benchmark -t 4 benchmark/speed_sweep.jl
#   julia --project=benchmark -t 4 benchmark/speed_sweep.jl --quick
#   julia --project=benchmark -t 4 benchmark/speed_sweep.jl --section=regression

include("common.jl")

using BenchmarkTools

const QUICK = "--quick" in ARGS
const SECTION = let s = findfirst(a -> startswith(a, "--section="), ARGS)
    s === nothing ? "all" : split(ARGS[s], "=")[2]
end

# (n_train_samples, n_features)
const GRID = QUICK ?
    [(500, 10), (5_000, 20)] :
    [(500, 10), (2_000, 20), (10_000, 50), (50_000, 50)]

const BATCH_SIZES = QUICK ? [1, 1_000] : [1, 100, 1_000, 10_000]
const CAT_CARDINALITIES = QUICK ? [5, 100] : [5, 50, 500]

print_header(title) = begin
    println("\n", "="^72)
    println(title)
    println("="^72)
end

print_grid_row(n, p, t_cb, t_mb) = begin
    su = t_cb / t_mb
    tag = su >= 1.0 ? "MB $(round(su; digits=2))x" : "CB $(round(1/su; digits=2))x"
    println("  n=$(lpad(n, 6))  p=$(lpad(p, 3))  CatBoost $(rpad(round(t_cb; digits=1), 8))ms  MichiBoost $(rpad(round(t_mb; digits=1), 8))ms  $tag")
end

function bench_fit(X_tr, y_tr, loss, is_reg)
    cb_pool = CatBoost.Pool(data=np.array(X_tr), label=np.array(y_tr))
    mb_pool = MichiBoost.Pool(X_tr; label=y_tr)

    cb_fn = is_reg ?
        () -> CatBoost.CatBoostRegressor(iterations=ITERS, learning_rate=LR, depth=DEPTH,
                  random_seed=SEED, verbose=false, loss_function=loss, thread_count=N_THREADS) :
        () -> CatBoost.CatBoostClassifier(iterations=ITERS, learning_rate=LR, depth=DEPTH,
                  random_seed=SEED, verbose=false, loss_function=loss, thread_count=N_THREADS)
    mb_fn = is_reg ?
        () -> MichiBoostRegressor(; iterations=ITERS, learning_rate=LR, depth=DEPTH,
                  loss_function=loss, random_seed=SEED, verbose=false) :
        () -> MichiBoostClassifier(; iterations=ITERS, learning_rate=LR, depth=DEPTH,
                  loss_function=loss, random_seed=SEED, verbose=false)

    t_cb = median(@benchmark(CatBoost.fit!($cb_fn(), $cb_pool), samples=3, evals=1)).time / 1e6
    t_mb = median(@benchmark(MichiBoost.fit!($mb_fn(), $mb_pool), samples=3, evals=1)).time / 1e6
    return t_cb, t_mb
end

function sweep_task(title, gen_fn, loss, is_reg)
    print_header("TRAINING: $title")
    for (n, p) in GRID
        X, y = gen_fn(; n=round(Int, n / 0.8), p=p)
        X_tr, y_tr, _, _ = train_test_split(X, y)
        t_cb, t_mb = bench_fit(X_tr, y_tr, loss, is_reg)
        print_grid_row(n, p, t_cb, t_mb)
    end
end

if SECTION in ("all", "regression")
    sweep_task("Regression (RMSE)", regression_data, "RMSE", true)
end

if SECTION in ("all", "binary")
    sweep_task("Binary classification (Logloss)", binary_data, "Logloss", false)
end

if SECTION in ("all", "multiclass")
    # 3-class target across the full n range
    sweep_task("Multiclass (3 classes)",
               (; n, p) -> multiclass_data(; n, p, k=3),
               "MultiClass", false)
end

if SECTION in ("all", "categorical")
    print_header("TRAINING: Categorical (n=5000, 5 cat + 5 num, cardinality sweep)")
    for k in CAT_CARDINALITIES
        X_num, X_cat, y = categorical_data(n=round(Int, 5000 / 0.8); k=k)
        rng = MersenneTwister(SEED)
        idx = randperm(rng, length(y))
        n_tr = round(Int, 0.8 * length(y))
        tr = idx[1:n_tr]

        _, jl_df_tr, cb_pool_tr, jl_pool_tr, _ =
            make_cat_frames(X_num[tr, :], X_cat[tr, :], y[tr])

        cb_fn = () -> CatBoost.CatBoostClassifier(iterations=ITERS, learning_rate=LR,
                         depth=DEPTH, random_seed=SEED, verbose=false,
                         loss_function="Logloss", thread_count=N_THREADS)
        mb_fn = () -> MichiBoostClassifier(; iterations=ITERS, learning_rate=LR, depth=DEPTH,
                         loss_function="Logloss", random_seed=SEED, verbose=false)

        t_cb = median(@benchmark(CatBoost.fit!($cb_fn(), $cb_pool_tr), samples=3, evals=1)).time / 1e6
        t_mb = median(@benchmark(MichiBoost.fit!($mb_fn(), $jl_pool_tr), samples=3, evals=1)).time / 1e6

        su = t_cb / t_mb
        tag = su >= 1.0 ? "MB $(round(su; digits=2))x" : "CB $(round(1/su; digits=2))x"
        println("  cardinality=$(lpad(k, 4))  CatBoost $(rpad(round(t_cb; digits=1), 8))ms  MichiBoost $(rpad(round(t_mb; digits=1), 8))ms  $tag")
    end
end

if SECTION in ("all", "inference")
    print_header("INFERENCE: batch size sweep (regression, n_train=10000, p=20)")
    X, y = regression_data(n=round(Int, 10_000 / 0.8), p=20)
    X_tr, y_tr, _, _ = train_test_split(X, y)

    cb = cb_train(X_tr, y_tr; loss="RMSE")
    mb = mb_train(X_tr, y_tr; loss="RMSE")

    rng = MersenneTwister(SEED)
    for b in BATCH_SIZES
        # Sampled rows (with replacement) rather than replicated — exercises
        # the full tree path distribution instead of a single branch.
        idx = rand(rng, 1:size(X_tr, 1), b)
        X_b = X_tr[idx, :]
        X_b_np = np.array(X_b)

        t_cb = median(@benchmark(CatBoost.predict($cb, $X_b_np))).time / 1e6
        t_mb = median(@benchmark(MichiBoost.predict($mb, $X_b))).time / 1e6
        su = t_cb / t_mb
        tag = su >= 1.0 ? "MB $(round(su; digits=2))x" : "CB $(round(1/su; digits=2))x"
        println("  batch=$(lpad(b, 6))  CatBoost $(rpad(round(t_cb; digits=3), 9))ms  MichiBoost $(rpad(round(t_mb; digits=3), 9))ms  $tag")
    end
end

println()
