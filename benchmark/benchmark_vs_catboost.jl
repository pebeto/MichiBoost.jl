# MichiBoost.jl vs CatBoost — Correctness & Performance
#
# Run with:
#   julia --project=benchmark -t 4 benchmark/benchmark_vs_catboost.jl

using BenchmarkTools
using CatBoost
using DataFrames
using MichiBoost
using PythonCall
using Random
using Statistics
using Test

const np = pyimport("numpy")
const pd = pyimport("pandas")

const ITERS = 100
const LR = 0.03
const DEPTH = 6
const SEED = 9
const N_THREADS = Threads.nthreads()

function regression_data(; n=200, p=10, seed=SEED)
    rng = MersenneTwister(seed)
    X = randn(rng, n, p)
    y = X * randn(rng, p) .+ 0.5 .* randn(rng, n)
    return X, y
end

function binary_data(; n=200, p=10, seed=SEED)
    rng = MersenneTwister(seed)
    X = randn(rng, n, p)
    y = Float64.(rand(rng, n) .< (1.0 ./ (1.0 .+ exp.(-(X * randn(rng, p))))))
    return X, y
end

function multiclass_data(; n=300, p=10, k=3, seed=SEED)
    rng = MersenneTwister(seed)
    X = randn(rng, n, p)
    y = Float64.([mod(round(Int, X[i, 1] + X[i, 2] + 10), k) for i in 1:n])
    return X, y
end

function categorical_data(; n=1000, p_num=5, p_cat=5, k=20, seed=SEED)
    rng = MersenneTwister(seed)
    X_num = randn(rng, n, p_num)
    X_cat = string.(rand(rng, 0:(k - 1), n, p_cat))
    y = Float64.(rand(rng, n) .< 0.5)
    return X_num, X_cat, y
end

function train_test_split(X::Matrix, y::Vector; ratio=0.8, seed=SEED)
    rng = MersenneTwister(seed)
    idx = randperm(rng, length(y))
    n_tr = round(Int, ratio * length(y))
    tr, te = idx[1:n_tr], idx[n_tr+1:end]
    return X[tr, :], y[tr], X[te, :], y[te]
end

function cb_train(X, y; loss="RMSE", cb_pool=nothing)
    pool = cb_pool !== nothing ? cb_pool : CatBoost.Pool(data=np.array(X), label=np.array(y))
    is_reg = loss in ("RMSE", "MAE")
    m = is_reg ?
        CatBoost.CatBoostRegressor(
            iterations=ITERS, learning_rate=LR, depth=DEPTH,
            random_seed=SEED, verbose=false, loss_function=loss,
            thread_count=N_THREADS) :
        CatBoost.CatBoostClassifier(
            iterations=ITERS, learning_rate=LR, depth=DEPTH,
            random_seed=SEED, verbose=false, loss_function=loss,
            thread_count=N_THREADS)
    CatBoost.fit!(m, pool)
    return m
end

function mb_train(data, y; loss="RMSE")
    m = loss in ("RMSE", "MAE") ?
        MichiBoostRegressor(; iterations=ITERS, learning_rate=LR, depth=DEPTH,
            loss_function=loss, random_seed=SEED, verbose=false) :
        MichiBoostClassifier(; iterations=ITERS, learning_rate=LR, depth=DEPTH,
            loss_function=loss, random_seed=SEED, verbose=false)
    MichiBoost.fit!(m, data, y)
    return m
end

function make_cat_frames(X_num, X_cat, y)
    cat_names = ["cat$i" for i in 1:size(X_cat, 2)]
    num_names = ["num$i" for i in 1:size(X_num, 2)]
    py_df = pd.DataFrame(pydict(merge(
        Dict(cat_names[i] => X_cat[:, i] for i in eachindex(cat_names)),
        Dict(num_names[i] => X_num[:, i] for i in eachindex(num_names)),
    )))
    jl_df = hcat(
        DataFrame(X_cat, cat_names),
        DataFrame(X_num, num_names),
    )
    cb_pool = CatBoost.Pool(data=py_df, label=np.array(y), cat_features=cat_names)
    jl_pool = MichiBoost.Pool(jl_df; label=y)
    return py_df, jl_df, cb_pool, jl_pool
end

@testset "CatBoost vs MichiBoost — Correctness" begin
    @testset "Regression" begin
        X, y = regression_data(n=400, p=10)
        X_tr, y_tr, X_te, y_te = train_test_split(X, y)
        cb = cb_train(X_tr, y_tr)
        mb = mb_train(X_tr, y_tr)

        cb_pred = pyconvert(Vector{Float64}, CatBoost.predict(cb, np.array(X_te)))
        jl_pred = MichiBoost.predict(mb, X_te)

        r = cor(cb_pred, jl_pred)
        println("Regression correlation (test): r = $(round(r; digits=4))")
        println("CatBoost RMSE: $(round(sqrt(mean((cb_pred .- y_te).^2)); digits=4)),  ",
                "MichiBoost RMSE: $(round(sqrt(mean((jl_pred .- y_te).^2)); digits=4))")
        @test r > 0.8
    end

    @testset "Binary classification" begin
        X, y = binary_data(n=400, p=10)
        X_tr, y_tr, X_te, y_te = train_test_split(X, y)
        cb = cb_train(X_tr, y_tr; loss="Logloss")
        mb = mb_train(X_tr, y_tr; loss="Logloss")

        cb_prob = pyconvert(Matrix{Float64}, CatBoost.predict_proba(cb, np.array(X_te)))[:, 2]
        jl_prob = MichiBoost.predict(mb, X_te; prediction_type="Probability")

        r = cor(cb_prob, jl_prob)
        agreement = mean((cb_prob .>= 0.5) .== (jl_prob .>= 0.5))
        println("Binary prob correlation (test): r = $(round(r; digits=4))")
        println("Class agreement: $(round(100*agreement; digits=1))%")
        println("CatBoost accuracy: $(round(100*mean((cb_prob .>= 0.5) .== (y_te .== 1.0)); digits=1))%,  ",
                "MichiBoost accuracy: $(round(100*mean((jl_prob .>= 0.5) .== (y_te .== 1.0)); digits=1))%")
        @test r > 0.7
        @test agreement > 0.7
    end

    @testset "Multiclass classification" begin
        X, y = multiclass_data(n=600, p=10, k=3)
        X_tr, y_tr, X_te, y_te = train_test_split(X, y)
        cb = cb_train(X_tr, y_tr; loss="MultiClass")
        mb = mb_train(X_tr, y_tr; loss="MultiClass")

        cb_prob = pyconvert(Matrix{Float64}, CatBoost.predict_proba(cb, np.array(X_te)))
        jl_prob = MichiBoost.predict(mb, X_te; prediction_type="Probability")

        cb_cls = [argmax(cb_prob[i, :]) for i in axes(cb_prob, 1)]
        jl_cls = [argmax(jl_prob[i, :]) for i in axes(jl_prob, 1)]
        agreement = mean(cb_cls .== jl_cls)
        println("Multiclass agreement (test): $(round(100*agreement; digits=1))%")
        println("CatBoost accuracy: $(round(100*mean(cb_cls .== (Int.(y_te) .+ 1)); digits=1))%,  ",
                "MichiBoost accuracy: $(round(100*mean(jl_cls .== (Int.(y_te) .+ 1)); digits=1))%")
        @test agreement > 0.3
    end

    @testset "Categorical features" begin
        X_num, X_cat, y = categorical_data(n=1000)
        rng = MersenneTwister(SEED)
        idx = randperm(rng, length(y))
        n_tr = round(Int, 0.8 * length(y))
        tr, te = idx[1:n_tr], idx[n_tr+1:end]

        _, jl_df_tr, cb_pool_tr, _ = make_cat_frames(X_num[tr, :], X_cat[tr, :], y[tr])
        py_df_te, jl_df_te, _, _   = make_cat_frames(X_num[te, :], X_cat[te, :], y[te])

        cb = cb_train(nothing, y[tr]; loss="Logloss", cb_pool=cb_pool_tr)
        mb = mb_train(jl_df_tr, y[tr]; loss="Logloss")

        cb_prob = pyconvert(Matrix{Float64}, CatBoost.predict_proba(cb, py_df_te))[:, 2]
        jl_prob = MichiBoost.predict(mb, jl_df_te; prediction_type="Probability")

        agreement = mean((cb_prob .>= 0.5) .== (jl_prob .>= 0.5))
        println("Categorical class agreement (test): $(round(100*agreement; digits=1))%")
        println("CatBoost accuracy: $(round(100*mean((cb_prob .>= 0.5) .== (y[te] .== 1.0)); digits=1))%,  ",
                "MichiBoost accuracy: $(round(100*mean((jl_prob .>= 0.5) .== (y[te] .== 1.0)); digits=1))%")
        @test agreement > 0.6
    end
end

println("\n", "="^60)
println("BENCHMARK: CatBoost.jl vs MichiBoost.jl")
println("="^60)

for (label, X, y, loss) in [
    ("Small regression (200×10)",                     regression_data(n=200,  p=10)...,  "RMSE"),
    ("MAE regression (200×10)",                       regression_data(n=200,  p=10)...,  "MAE"),
    ("Medium regression (2000×20)",                   regression_data(n=2000, p=20)...,  "RMSE"),
    ("Binary classification (1000×15)",               binary_data(n=1000,     p=15)...,  "Logloss"),
    ("Multiclass classification (500×10, 3 classes)", multiclass_data(n=500,  p=10)...,  "MultiClass"),
]
    println("\n─── $label ───")
    X_tr, y_tr, X_te, y_te = train_test_split(X, y)

    is_reg = loss in ("RMSE", "MAE")

    # Pre-build pools outside benchmarks
    cb_pool_tr = CatBoost.Pool(data=np.array(X_tr), label=np.array(y_tr))
    jl_pool_tr = MichiBoost.Pool(X_tr; label=y_tr)

    # Build constructors outside so no ternary runs inside @benchmark
    cb_model_fn = is_reg ?
        () -> CatBoost.CatBoostRegressor(iterations=ITERS, learning_rate=LR, depth=DEPTH,
                  random_seed=SEED, verbose=false, loss_function=loss, thread_count=N_THREADS) :
        () -> CatBoost.CatBoostClassifier(iterations=ITERS, learning_rate=LR, depth=DEPTH,
                  random_seed=SEED, verbose=false, loss_function=loss, thread_count=N_THREADS)
    mb_model_fn = is_reg ?
        () -> MichiBoostRegressor(; iterations=ITERS, learning_rate=LR, depth=DEPTH,
                  loss_function=loss, random_seed=SEED, verbose=false) :
        () -> MichiBoostClassifier(; iterations=ITERS, learning_rate=LR, depth=DEPTH,
                  loss_function=loss, random_seed=SEED, verbose=false)

    t_cb = median(@benchmark(CatBoost.fit!($cb_model_fn(), $cb_pool_tr))).time / 1e6
    t_mb = median(@benchmark(MichiBoost.fit!($mb_model_fn(), $jl_pool_tr))).time / 1e6
    println("  Training:  CatBoost $(round(t_cb; digits=1)) ms  |  MichiBoost $(round(t_mb; digits=1)) ms")

    # Train final models for inference timing
    cb_model = cb_train(X_tr, y_tr; loss)
    mb_model = mb_train(X_tr, y_tr; loss)

    # Pre-convert test data outside inference benchmarks
    X_te_np = np.array(X_te)

    t_cb_pred = median(@benchmark(CatBoost.predict($cb_model, $X_te_np))).time / 1e6
    t_mb_pred = median(@benchmark(MichiBoost.predict($mb_model, $X_te))).time / 1e6
    println("  Inference: CatBoost $(round(t_cb_pred; digits=3)) ms  |  MichiBoost $(round(t_mb_pred; digits=3)) ms")
end

let
    println("\n─── Categorical features (1000×10, 5 cat + 5 num) ───")
    X_num, X_cat, y = categorical_data()
    rng = MersenneTwister(SEED)
    idx = randperm(rng, length(y))
    n_tr = round(Int, 0.8 * length(y))
    tr, te = idx[1:n_tr], idx[n_tr+1:end]

    py_df_tr, jl_df_tr, cb_pool_tr, jl_pool_tr =
        make_cat_frames(X_num[tr, :], X_cat[tr, :], y[tr])
    py_df_te, jl_df_te, _, _ =
        make_cat_frames(X_num[te, :], X_cat[te, :], y[te])

    cb_model_fn = () -> CatBoost.CatBoostClassifier(
        iterations=ITERS, learning_rate=LR, depth=DEPTH,
        random_seed=SEED, verbose=false, loss_function="Logloss",
        thread_count=N_THREADS)
    mb_model_fn = () -> MichiBoostClassifier(;
        iterations=ITERS, learning_rate=LR, depth=DEPTH,
        loss_function="Logloss", random_seed=SEED, verbose=false)

    t_cb = median(@benchmark(CatBoost.fit!($cb_model_fn(), $cb_pool_tr))).time / 1e6
    t_mb = median(@benchmark(MichiBoost.fit!($mb_model_fn(), $jl_pool_tr))).time / 1e6
    println("  Training:  CatBoost $(round(t_cb; digits=1)) ms  |  MichiBoost $(round(t_mb; digits=1)) ms")

    cb_model = cb_train(nothing, y[tr]; loss="Logloss", cb_pool=cb_pool_tr)
    mb_model = mb_train(jl_df_tr, y[tr]; loss="Logloss")

    # Pre-build inference pools outside benchmarks
    cat_names = ["cat$i" for i in 1:size(X_cat, 2)]
    cb_pool_te = CatBoost.Pool(data=py_df_te, cat_features=cat_names)

    t_cb_pred = median(@benchmark(CatBoost.predict($cb_model, $cb_pool_te))).time / 1e6
    t_mb_pred = median(@benchmark(MichiBoost.predict($mb_model, $jl_df_te))).time / 1e6
    println("  Inference: CatBoost $(round(t_cb_pred; digits=3)) ms  |  MichiBoost $(round(t_mb_pred; digits=3)) ms")
end

println("\n", "="^60, "\n")