using MichiBoost: MichiBoost, Pool, MichiBoostRegressor
using Random
using Test

@testset "RSM feature subsampling" begin
    Random.seed!(42)
    X = randn(100, 6)
    y = X[:, 1] .+ X[:, 2] .+ randn(100) .* 0.1

    model_full = MichiBoostRegressor(; iterations=20, depth=3, rsm=1.0, random_seed=1, verbose=false)
    MichiBoost.fit!(model_full, X, y)

    model_rsm = MichiBoostRegressor(; iterations=20, depth=3, rsm=0.5, random_seed=1, verbose=false)
    MichiBoost.fit!(model_rsm, X, y)

    preds_full = MichiBoost.predict(model_full, X)
    preds_rsm = MichiBoost.predict(model_rsm, X)
    @test length(preds_rsm) == 100
    @test all(isfinite, preds_rsm)
    @test preds_full != preds_rsm
end

@testset "Early stopping" begin
    Random.seed!(42)
    n = 200
    X = randn(n, 5)
    y = X[:, 1] .+ randn(n) .* 0.1

    train_pool = Pool(X[1:160, :]; label=y[1:160])
    val_pool = Pool(X[161:end, :]; label=y[161:end])

    model = MichiBoostRegressor(; iterations=500, depth=3, early_stopping_rounds=10, verbose=false)
    MichiBoost.fit!(model, train_pool; eval_set=val_pool)

    @test length(model.model.trees) < 500
    preds = MichiBoost.predict(model, val_pool)
    @test all(isfinite, preds)
end
