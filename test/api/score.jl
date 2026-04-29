using MichiBoost
using Random, Statistics
using Test

@testset "score — regressor returns R²" begin
    Random.seed!(42)
    n = 1000
    X = randn(n, 5)
    y = 2.0 .* X[:, 1] .- 1.5 .* X[:, 2] .+ 0.5 .* X[:, 3] .+ randn(n) .* 0.1

    X_train, y_train = X[1:800, :], y[1:800]
    X_test, y_test = X[801:end, :], y[801:end]

    model = MichiBoostRegressor(;
        iterations=200, learning_rate=0.05, depth=4, verbose=false
    )
    fit!(model, X_train, y_train)

    s = score(model, X_test, y_test)
    ŷ = predict(model, X_test)
    ȳ = mean(y_test)
    expected = 1.0 - sum((y_test .- ŷ) .^ 2) / sum((y_test .- ȳ) .^ 2)

    @test s isa Float64
    @test isapprox(s, expected; atol=1e-9)
    @test s > 0.95   # the synthetic signal is easy
end

@testset "score — regressor edge cases" begin
    Random.seed!(0)
    X = randn(10, 3)
    y = fill(5.0, 10)

    model = MichiBoostRegressor(; iterations=10, depth=2, verbose=false)
    fit!(model, X, y)

    # Constant y, perfect prediction → defined as 1.0.
    @test score(model, X, y) == 1.0
end

@testset "score — classifier returns accuracy" begin
    Random.seed!(42)
    n = 600
    X = randn(n, 4)
    y = Float64.(X[:, 1] .+ X[:, 2] .> 0)

    X_train, y_train = X[1:500, :], y[1:500]
    X_test, y_test = X[501:end, :], y[501:end]

    clf = MichiBoostClassifier(; iterations=200, learning_rate=0.05, depth=4, verbose=false)
    fit!(clf, X_train, y_train)

    s = score(clf, X_test, y_test)
    expected = mean(predict(clf, X_test) .== y_test)

    @test s isa Float64
    @test 0.0 <= s <= 1.0
    @test isapprox(s, expected; atol=1e-12)
end

@testset "score — classifier with string labels" begin
    Random.seed!(42)
    y = ["cat", "dog", "cat", "bird", "dog", "bird", "cat", "dog"]
    X = randn(8, 3)

    clf = MichiBoostClassifier(; iterations=20, depth=2, verbose=false)
    fit!(clf, X, y)

    s = score(clf, X, y)
    @test s isa Float64
    @test 0.0 <= s <= 1.0
end

@testset "score — multiclass classifier" begin
    Random.seed!(42)
    n = 600
    X = randn(n, 5)
    y = mod.(1:n, 3)

    X_train, y_train = X[1:500, :], y[1:500]
    X_test, y_test = X[501:end, :], y[501:end]

    clf = MichiBoostClassifier(;
        iterations=100, depth=3, loss_function="MultiClass", verbose=false
    )
    fit!(clf, X_train, y_train)

    s = score(clf, X_test, y_test)
    expected = mean(predict(clf, X_test) .== y_test)

    @test isapprox(s, expected; atol=1e-12)
end

@testset "score — accepts Pool" begin
    Random.seed!(42)
    X = randn(100, 3)
    y = Float64.(X[:, 1] .> 0)
    pool = Pool(X; label=y)

    clf = MichiBoostClassifier(; iterations=20, depth=2, verbose=false)
    fit!(clf, pool)

    s_pool = score(clf, pool, y)
    s_mat = score(clf, X, y)
    @test s_pool == s_mat
end

@testset "score — fails on untrained model" begin
    model = MichiBoostRegressor(; iterations=10)
    @test_throws ErrorException score(model, randn(5, 2), [1.0, 2.0, 3.0, 4.0, 5.0])
end

@testset "score — mismatched lengths" begin
    Random.seed!(0)
    X = randn(20, 2)
    y = randn(20)
    model = MichiBoostRegressor(; iterations=5, depth=2, verbose=false)
    fit!(model, X, y)

    @test_throws DimensionMismatch score(model, X, randn(15))
end
