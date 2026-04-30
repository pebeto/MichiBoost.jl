using MichiBoost
using MichiBoost: _apply_class_weights
using Random
using Statistics
using Test

@testset "class_weights — multiplies into Pool.weight" begin
    Random.seed!(0)
    n = 200
    X = randn(n, 3)
    y = vcat(zeros(180), ones(20))   # 90/10 imbalance
    pool = Pool(X; label=y)

    weighted = _apply_class_weights(pool, Dict(0.0 => 1.0, 1.0 => 5.0))
    @test weighted !== pool                    # original is not mutated
    @test weighted.weight !== nothing
    @test pool.weight === nothing
    @test all(weighted.weight[y .== 0.0] .== 1.0)
    @test all(weighted.weight[y .== 1.0] .== 5.0)
end

@testset "class_weights — composes with existing pool weights" begin
    X = randn(6, 2)
    y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    w = [1.0, 2.0, 1.0, 1.0, 1.0, 3.0]
    pool = Pool(X; label=y, weight=w)

    weighted = _apply_class_weights(pool, Dict(0.0 => 2.0, 1.0 => 4.0))
    @test weighted.weight == [2.0, 4.0, 2.0, 4.0, 4.0, 12.0]
end

@testset "class_weights — accepts Int keys against Float labels" begin
    X = randn(4, 2)
    y = [0.0, 1.0, 0.0, 1.0]
    pool = Pool(X; label=y)

    weighted = _apply_class_weights(pool, Dict(0 => 1.0, 1 => 7.0))
    @test weighted.weight == [1.0, 7.0, 1.0, 7.0]
end

@testset "class_weights — string labels via label_classes" begin
    X = randn(6, 2)
    y = ["cat", "dog", "cat", "bird", "dog", "bird"]
    pool = Pool(X; label=y)

    weighted = _apply_class_weights(pool, Dict("cat" => 1.0, "dog" => 2.0, "bird" => 4.0))
    # Pool stores label_classes sorted; the per-row weight depends on the
    # sample's label, not on storage order.
    expected = [d == "cat" ? 1.0 : (d == "dog" ? 2.0 : 4.0) for d in y]
    @test weighted.weight == expected
end

@testset "class_weights — missing class errors" begin
    X = randn(4, 2)
    y = [0.0, 1.0, 0.0, 1.0]
    pool = Pool(X; label=y)

    @test_throws ErrorException _apply_class_weights(pool, Dict(0.0 => 1.0))
end

@testset "class_weights — labelled Pool required" begin
    X = randn(3, 2)
    pool = Pool(X)
    @test_throws ErrorException _apply_class_weights(pool, Dict(0.0 => 1.0))
end

@testset "class_weights — end-to-end on imbalanced binary" begin
    Random.seed!(42)
    n = 400
    X = randn(n, 3)
    # Strong 90/10 imbalance; the unweighted classifier predicts the majority
    # class for many samples, the heavily-weighted one shifts the decision.
    y = Float64.(vcat(zeros(360), ones(40)))

    plain = MichiBoostClassifier(; iterations=50, depth=3, random_seed=1, verbose=false)
    fit!(plain, X, y)

    weighted = MichiBoostClassifier(;
        iterations=50,
        depth=3,
        random_seed=1,
        verbose=false,
        class_weights=Dict(0.0 => 1.0, 1.0 => 20.0),
    )
    fit!(weighted, X, y)

    p_plain = predict_proba(plain, X)
    p_weighted = predict_proba(weighted, X)
    # Heavily upweighting the positive class should raise its predicted
    # probability on average.
    @test mean(p_weighted) > mean(p_plain)
end

@testset "class_weights — rejected on regressor" begin
    X = randn(20, 2)
    y = randn(20)
    reg = MichiBoostRegressor(;
        iterations=10, depth=2, verbose=false, class_weights=Dict(0.0 => 1.0)
    )
    @test_throws ErrorException fit!(reg, X, y)
end
