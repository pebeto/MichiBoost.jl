using MichiBoost
using Random
using Test

@testset "is_fitted — false before fit!, true after" begin
    Random.seed!(0)
    X = randn(40, 3)
    y = X[:, 1] .+ randn(40) .* 0.1

    model = MichiBoostRegressor(; iterations=10, depth=2, verbose=false)
    @test is_fitted(model) == false

    fit!(model, X, y)
    @test is_fitted(model) == true
end

@testset "is_fitted — works for classifier" begin
    Random.seed!(1)
    X = randn(60, 3)
    y = Float64.(X[:, 1] .> 0)

    clf = MichiBoostClassifier(; iterations=10, depth=2, verbose=false)
    @test is_fitted(clf) == false

    fit!(clf, X, y)
    @test is_fitted(clf) == true
end

@testset "is_fitted — survives shrink" begin
    Random.seed!(2)
    X = randn(40, 2)
    y = X[:, 1] .+ randn(40) .* 0.1

    model = MichiBoostRegressor(; iterations=8, depth=2, verbose=false)
    fit!(model, X, y)
    shrink(model, 0)
    @test is_fitted(model) == true   # underlying MichiBoostModel still present
end
