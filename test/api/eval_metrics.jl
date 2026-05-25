using MichiBoost
using Random
using Statistics
using Test

@testset "eval_metrics: regression keys, shape, last-iter consistency" begin
    Random.seed!(0)
    n = 100
    X = randn(n, 3)
    y = X[:, 1] .+ randn(n) .* 0.1
    iters = 20
    pool = Pool(X; label=y)

    model = MichiBoostRegressor(; iterations=iters, depth=2, verbose=false)
    fit!(model, pool)

    res = eval_metrics(model, pool; metrics=[Metrics.RMSE, "MAE", Metrics.R2])

    @test Set(keys(res)) == Set(["RMSE", "MAE", "R2"])
    @test all(length(v) == iters for v in values(res))

    # Last iteration must equal the score / direct metric on the same pool.
    ŷ = predict(model, X)
    @test isapprox(res["RMSE"][end], sqrt(mean((y .- ŷ) .^ 2)); atol=1e-9)
    @test isapprox(res["MAE"][end], mean(abs.(y .- ŷ)); atol=1e-9)
    @test isapprox(res["R2"][end], score(model, X, y); atol=1e-9)
end

@testset "eval_metrics: binary keys, shape, monotone-ish trajectory" begin
    Random.seed!(1)
    n = 200
    X = randn(n, 4)
    y = Float64.(X[:, 1] .+ X[:, 2] .> 0)
    iters = 30
    pool = Pool(X; label=y)

    clf = MichiBoostClassifier(; iterations=iters, depth=3, verbose=false)
    fit!(clf, pool)

    res = eval_metrics(clf, pool; metrics=[Metrics.Logloss, Metrics.AUC, Metrics.Accuracy])
    @test Set(keys(res)) == Set(["Logloss", "AUC", "Accuracy"])
    @test all(length(v) == iters for v in values(res))

    # Symbol-form entries resolve through the same dispatcher.
    sym_res = eval_metrics(clf, pool; metrics=[:Logloss, :AUC, :Accuracy])
    @test sym_res["Logloss"] == res["Logloss"]
    @test sym_res["AUC"] == res["AUC"]
    @test sym_res["Accuracy"] == res["Accuracy"]

    # Bounds.
    @test all(0.0 .<= res["AUC"] .<= 1.0)
    @test all(0.0 .<= res["Accuracy"] .<= 1.0)
    @test all(res["Logloss"] .>= 0.0)
    # Last iteration: training accuracy ≥ first iteration's.
    @test res["Accuracy"][end] >= res["Accuracy"][1]
end

@testset "eval_metrics: multiclass" begin
    Random.seed!(2)
    n = 150
    X = randn(n, 4)
    y = Float64.(mod.(1:n, 3))
    iters = 25
    pool = Pool(X; label=y)

    clf = MichiBoostClassifier(;
        iterations=iters, depth=2, loss_function=Losses.MultiClass, verbose=false
    )
    fit!(clf, pool)

    res = eval_metrics(clf, pool; metrics=[Metrics.MultiLogloss, Metrics.Accuracy])
    @test Set(keys(res)) == Set(["MultiLogloss", "Accuracy"])
    @test length(res["MultiLogloss"]) == iters
    @test length(res["Accuracy"]) == iters
    @test all(0.0 .<= res["Accuracy"] .<= 1.0)
    @test all(res["MultiLogloss"] .>= 0.0)
end

@testset "eval_metrics: task-mismatch errors propagate" begin
    Random.seed!(3)
    X = randn(40, 2)
    y = randn(40)
    pool = Pool(X; label=y)
    model = MichiBoostRegressor(; iterations=5, depth=2, verbose=false)
    fit!(model, pool)

    @test_throws ErrorException eval_metrics(model, pool; metrics=[Metrics.AUC])
end

@testset "eval_metrics: fails on unlabelled pool / untrained model" begin
    pool = Pool(randn(5, 2))   # no label
    model = MichiBoostRegressor(; iterations=5)
    @test_throws ErrorException eval_metrics(model, pool; metrics=[Metrics.RMSE])

    Random.seed!(0)
    X = randn(20, 2)
    y = randn(20)
    fit!(model, X, y)
    @test_throws ErrorException eval_metrics(
        model,
        Pool(X);
        metrics=[Metrics.RMSE],   # this pool has no label
    )
end

@testset "eval_metrics: bad metric type errors" begin
    Random.seed!(0)
    X = randn(20, 2)
    y = randn(20)
    pool = Pool(X; label=y)
    model = MichiBoostRegressor(; iterations=5, depth=2, verbose=false)
    fit!(model, pool)

    @test_throws ErrorException eval_metrics(model, pool; metrics=[42])
end
