# MichiBoost.jl vs CatBoost (Python/C++) — Correctness & Performance
#
# Run with:
#   julia --project=test/benchmark_project test/benchmark_vs_catboost.jl

using BenchmarkTools
using MichiBoost
using PythonCall
using Random
using Statistics
using Test

const catboost = pyimport("catboost")
const np       = pyimport("numpy")

const ITERS = 100
const LR    = 0.03
const DEPTH = 6
const SEED  = 9

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
    X_cat = string.(rand(rng, 0:(k - 1), n, p_cat))  # string-coded for CatBoost compatibility
    cat_indices = collect(0:(p_cat - 1))  # 0-based for CatBoost
    y = Float64.(rand(rng, n) .< 0.5)
    return X_num, X_cat, y, cat_indices
end

function cb_regressor(X, y)
    m = catboost.CatBoostRegressor(;
        iterations=ITERS, learning_rate=LR, depth=DEPTH,
        random_seed=SEED, verbose=false)
    m.fit(catboost.Pool(np.array(X), label=np.array(y)))
    return m
end

function cb_classifier(X, y; loss="Logloss")
    m = catboost.CatBoostClassifier(;
        iterations=ITERS, learning_rate=LR, depth=DEPTH,
        random_seed=SEED, verbose=false, loss_function=loss)
    m.fit(catboost.Pool(np.array(X), label=np.array(y)))
    return m
end

function mb_train(X, y; loss="RMSE")
    MichiBoost.train(Pool(X; label=y);
        iterations=ITERS, learning_rate=LR, depth=DEPTH,
        loss_function=loss, random_seed=SEED, verbose=false)
end

@testset "CatBoost vs MichiBoost — Correctness" begin
    @testset "Regression" begin
        X, y = regression_data()
        cb = cb_regressor(X, y)
        mb = mb_train(X, y)

        py_pred = pyconvert(Vector{Float64}, cb.predict(np.array(X)))
        jl_pred = MichiBoost.predict(mb, Pool(X))

        r = cor(py_pred, jl_pred)
        println("Regression correlation: r = $(round(r; digits=4))")
        println("CatBoost RMSE: $(round(sqrt(mean((py_pred .- y).^2)); digits=4)),  ",
                "MichiBoost RMSE: $(round(sqrt(mean((jl_pred .- y).^2)); digits=4))")
        @test r > 0.8
    end

    @testset "Binary classification" begin
        X, y = binary_data()
        cb = cb_classifier(X, y)
        mb = mb_train(X, y; loss="Logloss")

        py_prob = pyconvert(Matrix{Float64}, cb.predict_proba(np.array(X)))[:, 2]
        jl_prob = MichiBoost.predict(mb, Pool(X))

        r = cor(py_prob, jl_prob)
        agreement = mean((py_prob .>= 0.5) .== (jl_prob .>= 0.5))
        println("Binary prob correlation: r = $(round(r; digits=4))")
        println("Class agreement: $(round(100*agreement; digits=1))%")
        println("CatBoost accuracy: $(round(100*mean((py_prob .>= 0.5) .== (y .== 1.0)); digits=1))%,  ",
                "MichiBoost accuracy: $(round(100*mean((jl_prob .>= 0.5) .== (y .== 1.0)); digits=1))%")
        @test r > 0.7
        @test agreement > 0.7
    end

    @testset "Multiclass classification" begin
        X, y = multiclass_data()
        cb = cb_classifier(X, y; loss="MultiClass")
        mb = mb_train(X, y; loss="MultiClass")

        py_prob = pyconvert(Matrix{Float64}, cb.predict_proba(np.array(X)))
        jl_prob = MichiBoost.predict(mb, Pool(X))

        py_cls = [argmax(py_prob[i, :]) for i in axes(py_prob, 1)]
        jl_cls = [argmax(jl_prob[i, :]) for i in axes(jl_prob, 1)]
        agreement = mean(py_cls .== jl_cls)
        println("Multiclass agreement: $(round(100*agreement; digits=1))%")
        println("CatBoost accuracy: $(round(100*mean(py_cls .== (Int.(y) .+ 1)); digits=1))%,  ",
                "MichiBoost accuracy: $(round(100*mean(jl_cls .== (Int.(y) .+ 1)); digits=1))%")
        @test agreement > 0.3
    end
end

println("\n", "="^60)
println("BENCHMARK: CatBoost (Python/C++) vs MichiBoost.jl")
println("="^60)

for (label, task, X, y, loss) in [
    ("Small regression (200×10)",       "Regression",  regression_data(n=200,  p=10)..., "RMSE"),
    ("Medium regression (2000×20)",     "Regression",  regression_data(n=2000, p=20)..., "RMSE"),
    ("Binary classification (1000×15)", "Binary",      binary_data(n=1000,    p=15)..., "Logloss"),
    ("Multiclass classification (500×10, 3 classes)", "Multiclass", multiclass_data(n=500, p=10)..., "MultiClass"),
]
    println("\n─── $label ───")
    py_pool = catboost.Pool(np.array(X), label=np.array(y))
    jl_pool = Pool(X; label=y)

    cb_fn = task == "Regression" ? catboost.CatBoostRegressor : catboost.CatBoostClassifier
    cb_kwargs = (iterations=ITERS, learning_rate=LR, depth=DEPTH, random_seed=SEED, verbose=false)
    task in ("Binary", "Multiclass") && (cb_kwargs = merge(cb_kwargs, (loss_function=loss,)))

    t_cb = median(@benchmark($cb_fn(; $cb_kwargs...).fit($py_pool))).time / 1e6
    t_mb = median(@benchmark(MichiBoost.train($jl_pool; iterations=$ITERS,
        learning_rate=$LR, depth=$DEPTH, loss_function=$loss,
        random_seed=$SEED, verbose=false))).time / 1e6

    println("  Training:  CatBoost $(round(t_cb; digits=1)) ms  |  MichiBoost $(round(t_mb; digits=1)) ms")
end

let
    println("\n─── Categorical features (1000×10, 5 cat + 5 num) ───")
    X_num, X_cat, y, cat_idx = categorical_data()

    # CatBoost: pandas DataFrame with string categorical columns
    pd = pyimport("pandas")
    cat_names = ["cat$i" for i in 1:size(X_cat, 2)]
    num_names = ["num$i" for i in 1:size(X_num, 2)]
    py_dict = pydict(
        merge(
            Dict(cat_names[i] => X_cat[:, i] for i in eachindex(cat_names)),
            Dict(num_names[i] => X_num[:, i] for i in eachindex(num_names)),
        )
    )
    py_df = pd.DataFrame(py_dict)
    py_pool = catboost.Pool(py_df, label=np.array(y), cat_features=cat_names)

    # MichiBoost: Pool auto-detects string columns as categorical
    using DataFrames
    df = hcat(
        DataFrame(X_cat, ["cat$i" for i in 1:size(X_cat, 2)]),
        DataFrame(X_num, ["num$i" for i in 1:size(X_num, 2)]),
    )
    jl_pool = Pool(df; label=y)

    cb_kwargs = (iterations=ITERS, learning_rate=LR, depth=DEPTH, random_seed=SEED,
                 verbose=false, loss_function="Logloss")
    t_cb = median(@benchmark(catboost.CatBoostClassifier(; $cb_kwargs...).fit($py_pool))).time / 1e6
    t_mb = median(@benchmark(MichiBoost.train($jl_pool; iterations=$ITERS,
        learning_rate=$LR, depth=$DEPTH, loss_function="Logloss",
        random_seed=$SEED, verbose=false))).time / 1e6

    println("  Training:  CatBoost $(round(t_cb; digits=1)) ms  |  MichiBoost $(round(t_mb; digits=1)) ms")
end

println("\n", "="^60, "\n")
