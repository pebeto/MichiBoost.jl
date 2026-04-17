using MichiBoost: MichiBoost, Pool, MichiBoostClassifier, predict_proba
using Random
using Test

@testset "Binary classification" begin
    X = [0.0 3.0; 4.0 1.0; 8.0 1.0; 9.0 1.0]
    y = [0.0, 0.0, 1.0, 1.0]

    model = MichiBoostClassifier(; iterations=10, learning_rate=0.5, depth=2, verbose=false)
    MichiBoost.fit!(model, X, y)

    preds = MichiBoost.predict(model, X)
    @test length(preds) == 4

    probs = predict_proba(model, X)
    @test length(probs) == 4
    @test all(0.0 .<= probs .<= 1.0)
end

@testset "Multiclass training and prediction" begin
    Random.seed!(1)
    X = randn(60, 4)
    y = Float64.(repeat([1, 2, 3], 20))

    model = MichiBoostClassifier(; iterations=20, depth=3, verbose=false)
    MichiBoost.fit!(model, X, y)

    probs = predict_proba(model, X)
    @test size(probs) == (60, 3)
    @test all(x -> isapprox(x, 1.0; atol=1e-6), sum(probs; dims=2))

    classes = MichiBoost.predict(model, X)
    @test length(classes) == 60
    @test all(c -> c in [1.0, 2.0, 3.0], classes)
end

@testset "String class labels" begin
    X = [0.0 1.0; 1.0 0.0; 2.0 0.0; 3.0 1.0]
    y = ["cat", "dog", "cat", "dog"]

    model = MichiBoostClassifier(; iterations=10, depth=2, verbose=false)
    MichiBoost.fit!(model, X, y)

    preds = MichiBoost.predict(model, X)
    @test all(p -> p in ("cat", "dog"), preds)
end
