using MichiBoost
using Random
using Test

@testset "get_params: reflects constructor kwargs" begin
    model = MichiBoostRegressor(; iterations=123, depth=4)
    p = get_params(model)
    @test p isa Dict{Symbol,Any}
    @test p[:iterations] == 123
    @test p[:depth] == 4
    @test p[:loss_function] == "RMSE"   # default applied at construction
end

@testset "get_params: returns an independent copy" begin
    model = MichiBoostRegressor(; iterations=10)
    p = get_params(model)
    p[:iterations] = 999
    @test get_params(model)[:iterations] == 10  # wrapper untouched
end

@testset "set_params!: overrides and adds keys, returns wrapper" begin
    model = MichiBoostRegressor(; iterations=10)
    returned = set_params!(model; iterations=50, depth=3)
    @test returned === model
    p = get_params(model)
    @test p[:iterations] == 50
    @test p[:depth] == 3
end

@testset "set_params!: affects subsequent fit!" begin
    Random.seed!(0)
    X = randn(60, 3)
    y = X[:, 1] .+ randn(60) .* 0.1

    model = MichiBoostRegressor(; iterations=5, depth=2, verbose=false)
    set_params!(model; iterations=20)
    fit!(model, X, y)
    @test length(model.model.trees) == 20
end

@testset "set_params!: works after fit! (no retraining)" begin
    Random.seed!(1)
    X = randn(40, 2)
    y = X[:, 1] .+ randn(40) .* 0.1

    model = MichiBoostRegressor(; iterations=8, depth=2, verbose=false)
    fit!(model, X, y)
    n_before = length(model.model.trees)
    set_params!(model; iterations=200)   # stored, but not retrained
    @test length(model.model.trees) == n_before
    @test get_params(model)[:iterations] == 200
end

@testset "get_params/set_params!: works for classifier" begin
    clf = MichiBoostClassifier(; iterations=10)
    @test get_params(clf)[:loss_function] == "Logloss"
    set_params!(clf; depth=5)
    @test get_params(clf)[:depth] == 5
end
