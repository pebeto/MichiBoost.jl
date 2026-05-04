using MichiBoost
using Random
using Test

@testset "shrink! — truncates trees in place" begin
    Random.seed!(0)
    X = randn(80, 3)
    y = X[:, 1] .+ randn(80) .* 0.1

    model = MichiBoostRegressor(; iterations=20, depth=2, verbose=false)
    fit!(model, X, y)

    @test length(model.model.trees) == 20

    returned = shrink!(model, 5)
    @test returned === model              # returns wrapper for chaining
    @test length(model.model.trees) == 5
end

@testset "shrink! — predictions match staged snapshot" begin
    Random.seed!(1)
    X = randn(60, 3)
    y = X[:, 1] .+ randn(60) .* 0.1
    iters = 15
    n_keep = 7

    model = MichiBoostRegressor(; iterations=iters, depth=2, verbose=false)
    fit!(model, X, y)

    # Capture the staged prediction at iteration n_keep before shrinking.
    staged = staged_predict(model, X)
    expected = staged[:, n_keep]

    shrink!(model, n_keep)
    @test predict(model, X) ≈ expected
end

@testset "shrink! — n=0 leaves only the initial prediction" begin
    Random.seed!(2)
    X = randn(40, 2)
    y = X[:, 1] .+ randn(40) .* 0.1

    model = MichiBoostRegressor(; iterations=10, depth=2, verbose=false)
    fit!(model, X, y)
    shrink!(model, 0)

    @test length(model.model.trees) == 0
    # All rows predict the initial baseline.
    preds = predict(model, X)
    @test all(preds .≈ model.model.initial_pred)
end

@testset "shrink! — n equals current length is a no-op" begin
    Random.seed!(3)
    X = randn(40, 2)
    y = randn(40)

    model = MichiBoostRegressor(; iterations=10, depth=2, verbose=false)
    fit!(model, X, y)
    before = predict(model, X)
    shrink!(model, length(model.model.trees))
    @test predict(model, X) ≈ before
end

@testset "shrink! — out-of-range raises" begin
    Random.seed!(0)
    X = randn(20, 2)
    y = randn(20)

    model = MichiBoostRegressor(; iterations=5, depth=2, verbose=false)
    fit!(model, X, y)

    @test_throws ErrorException shrink!(model, -1)
    @test_throws ErrorException shrink!(model, length(model.model.trees) + 1)
end

@testset "shrink! — fails on untrained wrapper" begin
    model = MichiBoostRegressor(; iterations=5)
    @test_throws ErrorException shrink!(model, 1)
end

@testset "shrink! — pairs with get_best_iteration" begin
    Random.seed!(4)
    X = randn(300, 4)
    y = Float64.(X[:, 1] .+ X[:, 2] .> 0)
    train_pool = Pool(X[1:200, :]; label=y[1:200])
    val_pool = Pool(X[201:end, :]; label=y[201:end])

    clf = MichiBoostClassifier(;
        iterations=200, depth=2, early_stopping_rounds=5, verbose=false
    )
    fit!(clf, train_pool; eval_set=val_pool)

    bi = get_best_iteration(clf)
    @test length(clf.model.trees) == bi   # ES already truncated to best_iteration

    # Shrinking further should leave the model with that many trees.
    shrink!(clf, max(1, bi - 2))
    @test length(clf.model.trees) == max(1, bi - 2)
end
