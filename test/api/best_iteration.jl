using MichiBoost
using Random
using Test

@testset "get_best_iteration / get_best_score: without early stopping" begin
    Random.seed!(0)
    X = randn(100, 3)
    y = X[:, 1] .+ randn(100) .* 0.1

    model = MichiBoostRegressor(; iterations=20, depth=2, verbose=false)
    fit!(model, X, y)

    # No ES → best_iteration is the number of trees actually built;
    # best_score is `nothing` because no eval signal was tracked.
    @test get_best_iteration(model) == length(model.model.trees) == 20
    @test get_best_score(model) === nothing
end

@testset "get_best_iteration / get_best_score: with early stopping" begin
    Random.seed!(42)
    X = randn(400, 4)
    y = Float64.(X[:, 1] .+ X[:, 2] .> 0)

    train_pool = Pool(X[1:300, :]; label=y[1:300])
    val_pool = Pool(X[301:end, :]; label=y[301:end])

    clf = MichiBoostClassifier(;
        iterations=200, depth=3, early_stopping_rounds=10, verbose=false
    )
    fit!(clf, train_pool; eval_set=val_pool)

    bi = get_best_iteration(clf)
    bs = get_best_score(clf)

    @test bi isa Int
    @test bi >= 1
    @test bi <= 200
    # Trees were truncated to best_iteration when ES triggered.
    @test length(clf.model.trees) == bi

    @test bs isa Float64
    @test isfinite(bs)
    @test bs >= 0.0   # logloss is non-negative
end

@testset "get_best_iteration / get_best_score: with eval_metric=AUC" begin
    Random.seed!(7)
    X = randn(400, 4)
    y = Float64.(X[:, 1] .+ X[:, 2] .> 0)

    train_pool = Pool(X[1:300, :]; label=y[1:300])
    val_pool = Pool(X[301:end, :]; label=y[301:end])

    clf = MichiBoostClassifier(;
        iterations=200,
        depth=3,
        early_stopping_rounds=10,
        eval_metric=Metrics.AUC,
        verbose=false,
    )
    fit!(clf, train_pool; eval_set=val_pool)

    bs = get_best_score(clf)
    @test bs !== nothing
    # AUC is bounded in [0, 1].
    @test 0.0 <= bs <= 1.0
end

@testset "get_best_iteration / get_best_score: eval_metric=:AUC mirrors tag form" begin
    Random.seed!(7)
    X = randn(400, 4)
    y = Float64.(X[:, 1] .+ X[:, 2] .> 0)

    train_pool = Pool(X[1:300, :]; label=y[1:300])
    val_pool = Pool(X[301:end, :]; label=y[301:end])

    # The Symbol form must produce the same trajectory and stop at the same
    # iteration as the tag-type form.
    function fit_with(metric)
        Random.seed!(7)
        clf = MichiBoostClassifier(;
            iterations=200,
            depth=3,
            early_stopping_rounds=10,
            eval_metric=metric,
            verbose=false,
        )
        fit!(clf, train_pool; eval_set=val_pool)
        return clf
    end

    tag_clf = fit_with(Metrics.AUC)
    sym_clf = fit_with(:AUC)

    @test get_best_iteration(sym_clf) == get_best_iteration(tag_clf)
    @test get_best_score(sym_clf) == get_best_score(tag_clf)
end

@testset "get_best_*: fail on untrained model" begin
    model = MichiBoostRegressor(; iterations=5)
    @test_throws ErrorException get_best_iteration(model)
    @test_throws ErrorException get_best_score(model)
end
