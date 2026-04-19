using MichiBoost: MichiBoost, Pool, MichiBoostRegressor, MichiBoostClassifier, shap_values
using Random
using Test

@testset "SHAP values — regression" begin
    Random.seed!(42)
    X = randn(50, 4)
    y = 2.0 .* X[:, 1] .- X[:, 2] .+ randn(50) .* 0.1

    model = MichiBoostRegressor(; iterations=30, depth=4, random_seed=1, verbose=false)
    MichiBoost.fit!(model, X, y)

    shap = shap_values(model, X)
    @test size(shap) == (50, 4)
    @test all(isfinite, shap)

    # Feature 1 should dominate (coefficient 2.0), feature 2 next (coefficient -1.0)
    mean_abs = vec(mean(abs.(shap); dims=1))
    @test argmax(mean_abs) == 1
end

@testset "SHAP values — binary classification" begin
    Random.seed!(1)
    X = randn(60, 3)
    y = Float64.(X[:, 1] .+ X[:, 2] .> 0)

    model = MichiBoostClassifier(; iterations=20, depth=3, random_seed=1, verbose=false)
    MichiBoost.fit!(model, X, y)

    shap = shap_values(model, X)
    @test size(shap) == (60, 3)
    @test all(isfinite, shap)
end

@testset "SHAP values — multiclass" begin
    Random.seed!(7)
    X = randn(90, 4)
    y = Float64.(repeat([1, 2, 3], 30))

    model = MichiBoostClassifier(; iterations=20, depth=3, random_seed=1, verbose=false)
    MichiBoost.fit!(model, X, y)

    shap = shap_values(model, X)
    @test size(shap) == (90, 4, 3)
    @test all(isfinite, shap)
end

@testset "SHAP values — accepts raw matrix" begin
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
    y = [1.0, 2.0, 3.0, 4.0]

    model = MichiBoostRegressor(; iterations=10, depth=2, verbose=false)
    MichiBoost.fit!(model, X, y)

    shap = shap_values(model, X)
    @test size(shap) == (4, 2)
end
