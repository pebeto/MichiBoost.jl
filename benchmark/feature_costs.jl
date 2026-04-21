# Timing of each advertised feature: cross-validation, early stopping,
# SHAP, RSM feature subsampling, sample weights, and model serialization.
# Compared against CatBoost where CatBoost has a direct equivalent; purely
# MichiBoost features (RSM toggle, sample-weight overhead, save/load) are
# measured against a no-op baseline so the cost is visible on its own.
#
#   julia --project=benchmark -t 4 benchmark/feature_costs.jl

include("common.jl")

using BenchmarkTools

section(title) = (println("\n", "="^72); println(title); println("="^72))

function cmp_row(name, t_mb, t_cb=nothing)
    if t_cb === nothing
        println("  $(rpad(name, 22))  MichiBoost $(round(t_mb; digits=1))ms")
    else
        su = t_cb / t_mb
        tag = su >= 1.0 ? "MB $(round(su; digits=2))x" : "CB $(round(1/su; digits=2))x"
        println("  $(rpad(name, 22))  CatBoost $(rpad(round(t_cb; digits=1), 8))ms  MichiBoost $(rpad(round(t_mb; digits=1), 8))ms  $tag")
    end
end

# Fixtures (single moderate-size dataset keeps comparisons apples-to-apples)
X_reg_tr, y_reg_tr, X_reg_te, y_reg_te = train_test_split(regression_data(n=2000, p=20)...)
X_bin_tr, y_bin_tr, X_bin_te, y_bin_te = train_test_split(binary_data(n=2000, p=20)...)

section("CROSS-VALIDATION  (5-fold, $(ITERS) iters)")
let
    cb_pool = CatBoost.Pool(data=np.array(X_reg_tr), label=np.array(y_reg_tr))
    cb_params = PyDict(Dict("iterations" => ITERS, "depth" => DEPTH,
                            "loss_function" => "RMSE", "logging_level" => "Silent"))
    t_cb = median(@benchmark(CatBoost.cv($cb_pool, params=$cb_params, fold_count=5),
                             samples=3, evals=1)).time / 1e6

    jl_pool = MichiBoost.Pool(X_reg_tr; label=y_reg_tr)
    jl_params = Dict("iterations" => ITERS, "depth" => DEPTH, "loss_function" => "RMSE")
    t_mb = median(@benchmark(MichiBoost.cv($jl_pool, params=$jl_params, fold_count=5),
                             samples=3, evals=1)).time / 1e6

    cmp_row("Regression (RMSE)", t_mb, t_cb)
end

let
    cb_pool = CatBoost.Pool(data=np.array(X_bin_tr), label=np.array(y_bin_tr))
    cb_params = PyDict(Dict("iterations" => ITERS, "depth" => DEPTH,
                            "loss_function" => "Logloss", "logging_level" => "Silent"))
    t_cb = median(@benchmark(CatBoost.cv($cb_pool, params=$cb_params, fold_count=5),
                             samples=3, evals=1)).time / 1e6

    jl_pool = MichiBoost.Pool(X_bin_tr; label=y_bin_tr)
    jl_params = Dict("iterations" => ITERS, "depth" => DEPTH, "loss_function" => "Logloss")
    t_mb = median(@benchmark(MichiBoost.cv($jl_pool, params=$jl_params, fold_count=5),
                             samples=3, evals=1)).time / 1e6

    cmp_row("Binary (Logloss)", t_mb, t_cb)
end

section("EARLY STOPPING  (max 200 iters, patience 20)")
let
    ES_ITERS, ES_STOP = 200, 20

    cb_pool_tr = CatBoost.Pool(data=np.array(X_reg_tr), label=np.array(y_reg_tr))
    cb_pool_te = CatBoost.Pool(data=np.array(X_reg_te), label=np.array(y_reg_te))
    cb_fn = () -> CatBoost.CatBoostRegressor(iterations=ES_ITERS, learning_rate=LR, depth=DEPTH,
                     random_seed=SEED, verbose=false, loss_function="RMSE", thread_count=N_THREADS)
    t_cb = median(@benchmark(CatBoost.fit!($cb_fn(), $cb_pool_tr, eval_set=$cb_pool_te,
                                           early_stopping_rounds=$ES_STOP),
                             samples=3, evals=1)).time / 1e6

    jl_pool_tr = MichiBoost.Pool(X_reg_tr; label=y_reg_tr)
    jl_pool_te = MichiBoost.Pool(X_reg_te; label=y_reg_te)
    mb_fn = () -> MichiBoostRegressor(; iterations=ES_ITERS, learning_rate=LR, depth=DEPTH,
                     loss_function="RMSE", random_seed=SEED, verbose=false)
    t_mb = median(@benchmark(MichiBoost.fit!($mb_fn(), $jl_pool_tr, eval_pool=$jl_pool_te,
                                             early_stopping_rounds=$ES_STOP),
                             samples=3, evals=1)).time / 1e6

    cmp_row("Regression (RMSE)", t_mb, t_cb)
end

section("SHAP VALUES  (per-sample attribution on test set)")
let
    cb = cb_train(X_reg_tr, y_reg_tr; loss="RMSE")
    mb = mb_train(X_reg_tr, y_reg_tr; loss="RMSE")

    cb_pool_te = CatBoost.Pool(data=np.array(X_reg_te), label=np.array(y_reg_te))

    t_cb = median(@benchmark(
        $cb.get_feature_importance(type="ShapValues", data=$cb_pool_te),
        samples=3, evals=1)).time / 1e6
    t_mb = median(@benchmark(MichiBoost.shap_values($mb, $X_reg_te), samples=3, evals=1)).time / 1e6

    cmp_row("Regression (n_te=$(size(X_reg_te,1)))", t_mb, t_cb)
end

section("RSM  (feature subsampling, MichiBoost only)")
let
    jl_pool = MichiBoost.Pool(X_reg_tr; label=y_reg_tr)
    mk(rsm) = () -> MichiBoostRegressor(; iterations=ITERS, learning_rate=LR, depth=DEPTH,
                       loss_function="RMSE", random_seed=SEED, verbose=false, rsm=rsm)

    t_full = median(@benchmark(MichiBoost.fit!($(mk(1.0))(), $jl_pool), samples=3, evals=1)).time / 1e6
    t_half = median(@benchmark(MichiBoost.fit!($(mk(0.5))(), $jl_pool), samples=3, evals=1)).time / 1e6

    cmp_row("rsm=1.0 (baseline)", t_full)
    cmp_row("rsm=0.5", t_half)
    println("  Δ: $(round(100 * (t_half - t_full) / t_full; digits=1))% vs rsm=1.0")
end

section("SAMPLE WEIGHTS  (MichiBoost only, uniform weights)")
let
    w = ones(Float64, length(y_reg_tr))
    pool_uw = MichiBoost.Pool(X_reg_tr; label=y_reg_tr)
    pool_w  = MichiBoost.Pool(X_reg_tr; label=y_reg_tr, weight=w)
    mk = () -> MichiBoostRegressor(; iterations=ITERS, learning_rate=LR, depth=DEPTH,
                  loss_function="RMSE", random_seed=SEED, verbose=false)

    t_uw = median(@benchmark(MichiBoost.fit!($mk(), $pool_uw), samples=3, evals=1)).time / 1e6
    t_w  = median(@benchmark(MichiBoost.fit!($mk(), $pool_w),  samples=3, evals=1)).time / 1e6

    cmp_row("no weights (baseline)", t_uw)
    cmp_row("uniform weights",       t_w)
    println("  Overhead: $(round(100 * (t_w - t_uw) / t_uw; digits=1))% vs unweighted")
end

section("SAVE / LOAD  (serialization roundtrip)")
let
    mb = mb_train(X_reg_tr, y_reg_tr; loss="RMSE")
    path = tempname() * ".jls"

    t_save = median(@benchmark(MichiBoost.save_model($mb, $path))).time / 1e6
    t_load = median(@benchmark(MichiBoost.load_model($path))).time / 1e6

    cmp_row("save", t_save)
    cmp_row("load", t_load)
    rm(path; force=true)
end

println()
