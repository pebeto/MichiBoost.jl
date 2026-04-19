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

@testset "SHAP values — rows sum to prediction minus baseline (regression)" begin
    Random.seed!(42)
    X = randn(40, 4)
    y = 2.0 .* X[:, 1] .- X[:, 2] .+ randn(40) .* 0.1

    model = MichiBoostRegressor(; iterations=50, depth=4, random_seed=1, verbose=false)
    MichiBoost.fit!(model, X, y)

    shap = shap_values(model, X)
    raw = predict(model, X; prediction_type="RawFormulaVal")
    # Baseline is E[f(x)] under uniform leaf weighting: initial_pred + sum_t lr*mean(leaves_t)
    m = model.model
    expected_raw = m.initial_pred + sum(m.learning_rate * mean(t.leaf_values) for t in m.trees)

    @test all(isapprox.(vec(sum(shap; dims=2)), raw .- expected_raw; atol=1e-10))
end

@testset "SHAP values — rows sum to prediction minus baseline (multiclass)" begin
    Random.seed!(7)
    X = randn(90, 4)
    y = Float64.(repeat([1, 2, 3], 30))

    model = MichiBoostClassifier(; iterations=30, depth=3, random_seed=1, verbose=false)
    MichiBoost.fit!(model, X, y)

    shap = shap_values(model, X)
    raw = predict(model, X; prediction_type="RawFormulaVal")
    m = model.model
    n_classes = m.n_classes
    for c in 1:n_classes
        expected_raw_c = m.initial_pred[c] + sum(m.learning_rate * mean(t.leaf_values[:, c]) for t in m.trees)
        @test all(isapprox.(vec(sum(shap[:, :, c]; dims=2)), raw[:, c] .- expected_raw_c; atol=1e-10))
    end
end

@testset "SHAP values — accepts raw matrix" begin
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
    y = [1.0, 2.0, 3.0, 4.0]

    model = MichiBoostRegressor(; iterations=10, depth=2, verbose=false)
    MichiBoost.fit!(model, X, y)

    shap = shap_values(model, X)
    @test size(shap) == (4, 2)
end
