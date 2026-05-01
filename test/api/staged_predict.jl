using MichiBoost
using Random
using Test

@testset "staged_predict — regressor shape and last-iter consistency" begin
    Random.seed!(0)
    n, p = 80, 4
    X = randn(n, p)
    y = X[:, 1] .+ randn(n) .* 0.1
    iters = 30
    model = MichiBoostRegressor(; iterations=iters, depth=2, verbose=false)
    fit!(model, X, y)

    s = staged_predict(model, X)
    @test s isa Matrix{Float64}
    @test size(s) == (n, iters)
    # Last column == final predict.
    @test s[:, end] ≈ predict(model, X)
end

@testset "staged_predict — binary classifier with all prediction types" begin
    Random.seed!(1)
    n = 80
    X = randn(n, 3)
    y = Float64.(X[:, 1] .> 0)
    iters = 25
    clf = MichiBoostClassifier(; iterations=iters, depth=2, verbose=false)
    fit!(clf, X, y)

    s_class = staged_predict(clf, X)                                       # default Class
    s_prob = staged_predict(clf, X; prediction_type=PredictionTypes.Probability)
    s_raw = staged_predict(clf, X; prediction_type=PredictionTypes.RawFormulaVal)

    @test size(s_class) == (n, iters)
    @test size(s_prob) == (n, iters)
    @test size(s_raw) == (n, iters)

    # Last column matches the corresponding non-staged predict.
    @test s_class[:, end] == predict(clf, X)
    @test s_prob[:, end] ≈ predict(clf, X; prediction_type=PredictionTypes.Probability)
    @test s_raw[:, end] ≈ predict(clf, X; prediction_type=PredictionTypes.RawFormulaVal)

    # Class labels live in the model's class_labels set.
    @test all(in(clf.model.class_labels), s_class)
end

@testset "staged_predict — multiclass: shapes and final consistency" begin
    Random.seed!(2)
    n = 90
    X = randn(n, 4)
    y = mod.(1:n, 3)
    iters = 20
    clf = MichiBoostClassifier(;
        iterations=iters, depth=2, loss_function=Losses.MultiClass, verbose=false
    )
    fit!(clf, X, y)

    s_class = staged_predict(clf, X)
    s_prob = staged_predict(clf, X; prediction_type=PredictionTypes.Probability)
    s_raw = staged_predict(clf, X; prediction_type=PredictionTypes.RawFormulaVal)

    @test size(s_class) == (n, iters)
    @test size(s_prob) == (n, 3, iters)
    @test size(s_raw) == (n, 3, iters)

    # Last slice matches non-staged.
    @test s_class[:, end] == predict(clf, X)
    @test s_prob[:, :, end] ≈ predict(clf, X; prediction_type=PredictionTypes.Probability)
    @test s_raw[:, :, end] ≈ predict(clf, X; prediction_type=PredictionTypes.RawFormulaVal)

    # Probability slices each sum to ~1 per sample.
    sums = sum(s_prob; dims=2)
    @test all(isapprox.(sums, 1.0; atol=1e-9))
end

@testset "staged_predict_proba — convenience for classifiers" begin
    Random.seed!(3)
    n = 60
    X = randn(n, 3)
    y = Float64.(X[:, 1] .> 0)
    iters = 15
    clf = MichiBoostClassifier(; iterations=iters, depth=2, verbose=false)
    fit!(clf, X, y)

    @test staged_predict_proba(clf, X) ==
        staged_predict(clf, X; prediction_type=PredictionTypes.Probability)
end

@testset "staged_predict — fails on untrained model" begin
    model = MichiBoostRegressor(; iterations=10)
    @test_throws ErrorException staged_predict(model, randn(5, 2))
end

@testset "staged_predict — early stopping truncates iterations" begin
    Random.seed!(4)
    X = randn(300, 4)
    y = Float64.(X[:, 1] .+ X[:, 2] .> 0)
    train_pool = Pool(X[1:200, :]; label=y[1:200])
    val_pool = Pool(X[201:end, :]; label=y[201:end])
    clf = MichiBoostClassifier(;
        iterations=200, depth=2, early_stopping_rounds=5, verbose=false
    )
    fit!(clf, train_pool; eval_set=val_pool)

    s = staged_predict(clf, X)
    # Number of staged iterations equals the number of trees actually retained.
    @test size(s, 2) == length(clf.model.trees)
    @test size(s, 2) == get_best_iteration(clf)
end
