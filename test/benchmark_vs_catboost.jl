# MichiBoost.jl vs CatBoost.jl (Python wrapper) — Correctness & Benchmark
#
# Run standalone:  julia --project=. test/benchmark_vs_catboost.jl
#
# Requires extras: PythonCall, CondaPkg, BenchmarkTools
# (listed in [extras] of Project.toml — install with Pkg.add if needed)
#
# This file:
#   1. Trains both implementations on identical data with the same hyperparams.
#   2. Compares predictions to verify MichiBoost produces reasonable results
#      relative to the reference Python CatBoost (C++ engine).
#   3. Benchmarks training and prediction for both backends.

using MichiBoost
using BenchmarkTools
using PythonCall
using Statistics
using Test
using Random

const py_catboost = pyimport("catboost")
const py_numpy    = pyimport("numpy")

function python_train_regressor(X, y; iterations=100, learning_rate=0.03,
                                depth=6, random_seed=42, verbose=false)
    py_X = py_numpy.array(X)
    py_y = py_numpy.array(y)
    pool = py_catboost.Pool(py_X, label=py_y)
    model = py_catboost.CatBoostRegressor(;
        iterations, learning_rate, depth, random_seed, verbose)
    model.fit(pool)
    preds = pyconvert(Vector{Float64}, model.predict(py_X))
    return model, preds
end

function python_train_classifier(X, y; iterations=100, learning_rate=0.03,
                                 depth=6, random_seed=42, verbose=false,
                                 loss_function="Logloss")
    py_X = py_numpy.array(X)
    py_y = py_numpy.array(y)
    pool = py_catboost.Pool(py_X, label=py_y)
    model = py_catboost.CatBoostClassifier(;
        iterations, learning_rate, depth, random_seed, verbose, loss_function)
    model.fit(pool)
    probs = pyconvert(Array{Float64}, model.predict_proba(py_X))
    classes = pyconvert(Vector, model.predict(py_X))
    return model, probs, classes
end

function make_regression_data(; n=200, p=10, seed=9)
    rng = MersenneTwister(seed)
    X = randn(rng, n, p)
    β = randn(rng, p)
    y = X * β .+ 0.5 .* randn(rng, n)
    return X, y
end

function make_binary_data(; n=200, p=10, seed=9)
    rng = MersenneTwister(seed)
    X = randn(rng, n, p)
    β = randn(rng, p)
    logits = X * β
    probs = 1.0 ./ (1.0 .+ exp.(-logits))
    y = Float64.(rand(rng, n) .< probs)
    return X, y
end

function make_multiclass_data(; n=300, p=10, k=3, seed=9)
    rng = MersenneTwister(seed)
    X = randn(rng, n, p)
    y = [mod(round(Int, X[i, 1] + X[i, 2] + 10), k) for i in 1:n]
    return X, Float64.(y)
end

const ITERS = 100
const LR    = 0.03
const DEPTH = 6
const SEED  = 9

@testset "CatBoost.jl (Python) vs MichiBoost.jl — Correctness" begin
    @testset "Regression" begin
        X, y = make_regression_data()

        _, py_preds = python_train_regressor(X, y;
            iterations=ITERS, learning_rate=LR, depth=DEPTH, random_seed=SEED)

        pool = Pool(X; label=y)
        jl_model = MichiBoost.train(pool;
            iterations=ITERS, learning_rate=LR, depth=DEPTH,
            loss_function="RMSE", random_seed=SEED, verbose=false)
        jl_preds = MichiBoost.predict(jl_model, pool)

        @test length(py_preds) == length(jl_preds)

        r = cor(py_preds, jl_preds)
        println("  Regression correlation: r = $(round(r; digits=4))")
        @test r > 0.8

        rmse_py = sqrt(mean((py_preds .- y) .^ 2))
        rmse_jl = sqrt(mean((jl_preds .- y) .^ 2))
        println("  CatBoost.jl RMSE: $(round(rmse_py; digits=4)),  MichiBoost.jl RMSE: $(round(rmse_jl; digits=4))")
        @test rmse_jl < 3.0 * rmse_py + 1.0
    end

    @testset "Binary classification" begin
        X, y = make_binary_data()

        _, py_probs, _ = python_train_classifier(X, y;
            iterations=ITERS, learning_rate=LR, depth=DEPTH, random_seed=SEED)

        pool = Pool(X; label=y)
        jl_model = MichiBoost.train(pool;
            iterations=ITERS, learning_rate=LR, depth=DEPTH,
            loss_function="Logloss", random_seed=SEED, verbose=false)
        jl_probs = MichiBoost.predict(jl_model, pool)

        py_prob_pos = py_probs[:, 2]
        @test length(py_prob_pos) == length(jl_probs)

        r = cor(py_prob_pos, jl_probs)
        println("  Binary prob correlation: r = $(round(r; digits=4))")
        @test r > 0.7

        agreement = mean((py_prob_pos .>= 0.5) .== (jl_probs .>= 0.5))
        println("  Class agreement: $(round(100*agreement; digits=1))%")
        @test agreement > 0.7

        acc_py = mean((py_prob_pos .>= 0.5) .== (y .== 1.0))
        acc_jl = mean((jl_probs .>= 0.5) .== (y .== 1.0))
        println("  CatBoost.jl accuracy: $(round(100*acc_py; digits=1))%,  MichiBoost.jl accuracy: $(round(100*acc_jl; digits=1))%")
        @test acc_py > 0.6
        @test acc_jl > 0.6
    end

    @testset "Multi-class classification" begin
        X, y = make_multiclass_data()

        _, py_probs, _ = python_train_classifier(X, y;
            iterations=ITERS, learning_rate=LR, depth=DEPTH, random_seed=SEED,
            loss_function="MultiClass")

        pool = Pool(X; label=y)
        jl_model = MichiBoost.train(pool;
            iterations=ITERS, learning_rate=LR, depth=DEPTH,
            loss_function="MultiClass", random_seed=SEED, verbose=false)
        jl_probs = MichiBoost.predict(jl_model, pool)

        @test size(py_probs) == size(jl_probs)

        py_cls = [argmax(py_probs[i, :]) for i in axes(py_probs, 1)]
        jl_cls = [argmax(jl_probs[i, :]) for i in axes(jl_probs, 1)]
        agreement = mean(py_cls .== jl_cls)
        println("  Multiclass agreement: $(round(100*agreement; digits=1))%")
        @test agreement > 0.3

        acc_py = mean(py_cls .== (Int.(y) .+ 1))
        acc_jl = mean(jl_cls .== (Int.(y) .+ 1))
        println("  CatBoost.jl accuracy: $(round(100*acc_py; digits=1))%,  MichiBoost.jl accuracy: $(round(100*acc_jl; digits=1))%")
        @test acc_py > 0.4
        @test acc_jl > 0.33
    end
end

@testset "CatBoost.jl (Python) vs MichiBoost.jl — Benchmarks" begin

    println("\n" * "="^60)
    println("  BENCHMARK: CatBoost.jl (Python/C++) vs MichiBoost.jl")
    println("="^60)

    for (label, n, p) in [("Small (200×10)", 200, 10),
                           ("Medium (2000×20)", 2000, 20)]

        println("\n─── Regression: $label ───")
        X, y = make_regression_data(; n, p)

        py_X = py_numpy.array(X)
        py_y = py_numpy.array(y)
        py_pool = py_catboost.Pool(py_X, label=py_y)

        t_cb = @elapsed for _ in 1:3
            m = py_catboost.CatBoostRegressor(;
                iterations=ITERS, learning_rate=LR, depth=DEPTH,
                random_seed=SEED, verbose=false)
            m.fit(py_pool)
        end
        t_cb /= 3

        jl_pool = Pool(X; label=y)
        t_mb = @elapsed for _ in 1:3
            MichiBoost.train(jl_pool;
                iterations=ITERS, learning_rate=LR, depth=DEPTH,
                loss_function="RMSE", random_seed=SEED, verbose=false)
        end
        t_mb /= 3

        println("  Training:    CatBoost.jl $(round(t_cb*1000; digits=1)) ms  |  MichiBoost.jl $(round(t_mb*1000; digits=1)) ms")

        py_model = py_catboost.CatBoostRegressor(;
            iterations=ITERS, learning_rate=LR, depth=DEPTH,
            random_seed=SEED, verbose=false)
        py_model.fit(py_pool)

        jl_model = MichiBoost.train(jl_pool;
            iterations=ITERS, learning_rate=LR, depth=DEPTH,
            loss_function="RMSE", random_seed=SEED, verbose=false)

        t_cb_pred = @elapsed for _ in 1:50
            py_model.predict(py_X)
        end
        t_cb_pred /= 50

        t_mb_pred = @elapsed for _ in 1:50
            MichiBoost.predict(jl_model, jl_pool)
        end
        t_mb_pred /= 50

        println("  Prediction:  CatBoost.jl $(round(t_cb_pred*1000; digits=2)) ms  |  MichiBoost.jl $(round(t_mb_pred*1000; digits=2)) ms")

        @test true
    end

    println("\n─── Binary Classification (1000×15) ───")
    X, y = make_binary_data(; n=1000, p=15)

    py_X = py_numpy.array(X)
    py_y = py_numpy.array(y)
    py_pool = py_catboost.Pool(py_X, label=py_y)
    jl_pool = Pool(X; label=y)

    t_cb = @elapsed for _ in 1:3
        m = py_catboost.CatBoostClassifier(;
            iterations=ITERS, learning_rate=LR, depth=DEPTH,
            random_seed=SEED, verbose=false)
        m.fit(py_pool)
    end
    t_cb /= 3

    t_mb = @elapsed for _ in 1:3
        MichiBoost.train(jl_pool;
            iterations=ITERS, learning_rate=LR, depth=DEPTH,
            loss_function="Logloss", random_seed=SEED, verbose=false)
    end
    t_mb /= 3

    println("  Training:    CatBoost.jl $(round(t_cb*1000; digits=1)) ms  |  MichiBoost.jl $(round(t_mb*1000; digits=1)) ms")

    println("\n" * "="^60)
    println("  BENCHMARK COMPLETE")
    println("="^60 * "\n")

    @test true
end
