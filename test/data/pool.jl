using MichiBoost: MichiBoost, Pool
using Test

@testset "Pool construction from table" begin
    tbl = (floats=collect(0.5:0.5:3.0), ints=collect(1:6))
    pool = Pool(tbl)
    @test pool.n_samples == 6
    @test pool.n_features == 2
    @test MichiBoost.n_numerical(pool) == 2
    @test MichiBoost.n_categorical(pool) == 0
end

@testset "Pool construction with categorical features" begin
    tbl = (cat=["a", "b", "a", "c"], num=[1.0, 2.0, 3.0, 4.0])
    pool = Pool(tbl; label=[1.0, 0.0, 1.0, 0.0])
    @test pool.n_samples == 4
    @test MichiBoost.n_categorical(pool) == 1
    @test MichiBoost.n_numerical(pool) == 1
    @test pool.label == [1.0, 0.0, 1.0, 0.0]
end

@testset "Pool construction from matrix" begin
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    pool = Pool(X; label=[10.0, 20.0, 30.0])
    @test pool.n_samples == 3
    @test MichiBoost.n_numerical(pool) == 2
end

@testset "Pool with feature_names and weight" begin
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    pool = Pool(X; label=[1.0, 2.0, 3.0],
                feature_names=[:alpha, :beta],
                weight=[1.0, 2.0, 1.0])
    @test pool.feature_names == [:alpha, :beta]
    @test pool.weight == [1.0, 2.0, 1.0]
end

@testset "Sample weights are honored during training" begin
    using Random
    Random.seed!(42)
    # Two classes with very different weights — the weighted model should
    # predict differently from the unweighted one.
    X = [0.0 1.0; 1.0 0.0; 2.0 0.0; 3.0 1.0; 4.0 0.0; 5.0 1.0]
    y = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    w = [10.0, 10.0, 1.0, 1.0, 10.0, 1.0]  # heavily upweight class-0 samples

    pool_unweighted = Pool(X; label=y)
    pool_weighted   = Pool(X; label=y, weight=w)

    model_uw = MichiBoostClassifier(; iterations=20, depth=3, random_seed=1, verbose=false)
    model_w  = MichiBoostClassifier(; iterations=20, depth=3, random_seed=1, verbose=false)
    MichiBoost.fit!(model_uw, pool_unweighted)
    MichiBoost.fit!(model_w,  pool_weighted)

    preds_uw = MichiBoost.predict(model_uw, X; prediction_type="Probability")
    preds_w  = MichiBoost.predict(model_w,  X; prediction_type="Probability")
    @test preds_uw != preds_w
end

@testset "Pool slicing" begin
    tbl = (a=[1.0, 2.0, 3.0, 4.0], b=[5.0, 6.0, 7.0, 8.0])
    pool = Pool(tbl; label=[10.0, 20.0, 30.0, 40.0])
    sliced = MichiBoost.slice(pool, [1, 3])
    @test sliced.n_samples == 2
    @test sliced.label == [10.0, 30.0]
end

@testset "fit! with unlabeled Pool preserves categorical columns" begin
    using DataFrames
    using MichiBoost: MichiBoostClassifier
    df = DataFrame(
        cat1=["a", "b", "a", "b", "c", "c", "a", "b"],
        num1=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    labels = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

    unlabeled = Pool(df)
    @test unlabeled.label === nothing
    @test MichiBoost.n_categorical(unlabeled) == 1

    model = MichiBoostClassifier(; iterations=10, depth=2, verbose=false)
    MichiBoost.fit!(model, unlabeled, labels)

    preds = MichiBoost.predict(model, df)
    @test length(preds) == 8
    @test all(isfinite, preds)
end
