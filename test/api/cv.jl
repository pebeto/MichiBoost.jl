using MichiBoost: MichiBoost, Pool, cv
using Random
using Test

@testset "Regression CV" begin
    data = (x1=randn(20), x2=randn(20))
    labels = randn(20)
    pool = Pool(data; label=labels)

    scores = cv(
        pool;
        fold_count=2,
        verbose=false,
        params=Dict("iterations" => 10, "depth" => 2, "loss_function" => "RMSE"),
    )
    @test haskey(scores, :mean_test_loss)
    @test length(scores.test_loss) == 2
    @test scores.mean_train_loss > 0.0
    @test scores.mean_test_loss > 0.0
end

@testset "cv() forwards all params" begin
    Random.seed!(7)
    X = randn(120, 6)
    y = X[:, 1] .+ X[:, 2] .+ randn(120) .* 0.1
    pool = Pool(X; label=y)

    s1 = cv(
        pool;
        fold_count=2,
        random_seed=1,
        params=Dict(
            "iterations" => 20, "depth" => 3, "rsm" => 1.0, "loss_function" => "RMSE"
        ),
    )
    s2 = cv(
        pool;
        fold_count=2,
        random_seed=1,
        params=Dict(
            "iterations" => 20, "depth" => 3, "rsm" => 0.3, "loss_function" => "RMSE"
        ),
    )
    @test s1.train_loss != s2.train_loss
end

@testset "Binary classification CV" begin
    Random.seed!(42)
    n = 100
    X = randn(n, 3)
    y = Float64.([X[i, 1] + X[i, 2] > 0 for i in 1:n])
    pool = Pool(X; label=y)

    scores = cv(
        pool;
        fold_count=3,
        verbose=false,
        params=Dict("iterations" => 20, "depth" => 3, "loss_function" => "Logloss"),
    )
    @test haskey(scores, :mean_test_loss)
    @test length(scores.test_loss) == 3
    @test scores.mean_train_loss > 0.0
    @test scores.mean_test_loss > 0.0
    @test all(x -> x > 0.0, scores.train_loss)
    @test all(x -> x > 0.0, scores.test_loss)
end

@testset "Multiclass classification CV" begin
    Random.seed!(123)
    n = 150
    X = randn(n, 4)
    y = Float64.([
        if (X[i, 1] > 0.5)
            1.0
        elseif (X[i, 2] > 0.5)
            2.0
        else
            3.0
        end for i in 1:n
    ])
    pool = Pool(X; label=y)

    scores = cv(
        pool;
        fold_count=3,
        verbose=false,
        params=Dict("iterations" => 20, "depth" => 3, "loss_function" => "MultiClass"),
    )
    @test haskey(scores, :mean_test_loss)
    @test length(scores.test_loss) == 3
    @test scores.mean_train_loss > 0.0
    @test scores.mean_test_loss > 0.0
    @test all(x -> x > 0.0, scores.train_loss)
    @test all(x -> x > 0.0, scores.test_loss)
end

@testset "Multiclass CV class-imbalanced folds" begin
    # Class 3 appears only twice — with 3 folds it will be absent from
    # at least one training fold. Unseen test samples should contribute
    # uniform-prior loss (-log(1/n_classes)), not zero.
    X = Float64.(reshape(1:30, 10, 3))
    y = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0]
    pool = Pool(X; label=y)

    scores = cv(
        pool;
        fold_count=3,
        shuffle=false,
        random_seed=0,
        params=Dict("iterations" => 5, "depth" => 2, "loss_function" => "MultiClass"),
    )
    @test all(x -> x > 0.0, scores.test_loss)
end

@testset "Stratified CV preserves class proportions" begin
    # 90 / 10 imbalance; with stratified=true every fold should have the
    # same ~90 / 10 mix instead of one fold accidentally getting 0 minority.
    Random.seed!(0)
    n_pos, n_neg, k = 90, 10, 5
    X = randn(n_pos + n_neg, 3)
    y = vcat(zeros(n_pos), ones(n_neg))
    pool = Pool(X; label=y)

    folds = MichiBoost._stratified_folds(y, k, true, MersenneTwister(42))
    @test length(folds) == k
    @test sum(length, folds) == n_pos + n_neg
    # Every fold should contain at least one minority sample.
    @test all(f -> any(y[f] .== 1.0), folds)
    # Class proportions per fold should be near the global ratio (within ±1).
    for f in folds
        n_minority = sum(y[f] .== 1.0)
        @test abs(n_minority - n_neg / k) <= 1
    end
end

@testset "Stratified CV runs end-to-end on imbalanced binary data" begin
    Random.seed!(1)
    X = randn(120, 4)
    y = vcat(zeros(108), ones(12))   # ~10% minority
    pool = Pool(X; label=y)

    params = Dict("iterations" => 10, "depth" => 2, "loss_function" => "Logloss")
    s_plain = cv(pool; fold_count=4, stratified=false, random_seed=1, params=params)
    s_stratified = cv(pool; fold_count=4, stratified=true, random_seed=1, params=params)

    @test length(s_stratified.test_loss) == 4
    @test all(x -> x > 0.0, s_stratified.test_loss)
    # Both NamedTuples have the same shape.
    @test keys(s_plain) == keys(s_stratified)
end

@testset "Stratified CV errors on under-populated class" begin
    Random.seed!(0)
    X = randn(20, 3)
    y = vcat(zeros(18), ones(2))   # minority has 2 samples
    pool = Pool(X; label=y)

    @test_throws ErrorException cv(
        pool;
        fold_count=5,
        stratified=true,
        params=Dict("iterations" => 5, "loss_function" => "Logloss"),
    )
end
