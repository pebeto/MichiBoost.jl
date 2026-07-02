using MichiBoost
using Random
using Statistics
using Test

@testset "MultiClassOneVsAll — fits, normalized probabilities" begin
    Random.seed!(0)
    n = 600
    X = randn(n, 5)
    raw = [(X[i, 1] + 0.4randn(), X[i, 2] + 0.4randn(), 0.5randn()) for i in 1:n]
    y = Float64.([argmax(r) - 1 for r in raw])

    clf = MichiBoostClassifier(;
        iterations=80,
        depth=3,
        learning_rate=0.1,
        loss_function=MultiClassOneVsAll(),
        verbose=false,
        random_seed=1,
    )
    fit!(clf, X, y)

    P = predict_proba(clf, X)
    @test size(P) == (n, 3)
    @test all(P .>= 0)
    @test all(isapprox.(sum(P; dims=2), 1.0; atol=1e-9))
    @test mean(predict(clf, X) .== y) > 0.7
end

@testset "MultiClassOneVsAll — comparable to softmax MultiClass" begin
    Random.seed!(1)
    n = 800
    X = randn(n, 6)
    raw = [(X[i, 1] + 0.5randn(), X[i, 2] + 0.5randn(), X[i, 3] + 0.5randn()) for i in 1:n]
    y = Float64.([argmax(r) - 1 for r in raw])

    ova = MichiBoostClassifier(;
        iterations=80,
        depth=3,
        loss_function=MultiClassOneVsAll(),
        verbose=false,
        random_seed=2,
    )
    soft = MichiBoostClassifier(;
        iterations=80, depth=3, loss_function="MultiClass", verbose=false, random_seed=2
    )
    fit!(ova, X, y)
    fit!(soft, X, y)
    acc_ova = mean(predict(ova, X) .== y)
    acc_soft = mean(predict(soft, X) .== y)
    @test acc_ova > acc_soft - 0.1
end

@testset "MultiClassOneVsAll — four classes" begin
    Random.seed!(2)
    n = 800
    X = randn(n, 5)
    raw = [(X[i, 1], X[i, 2], X[i, 3], X[i, 4]) .+ 0.5 .* randn(4) for i in 1:n]
    y = Float64.([argmax(r) - 1 for r in raw])

    clf = MichiBoostClassifier(;
        iterations=60, depth=3, loss_function=MultiClassOneVsAll(), verbose=false
    )
    fit!(clf, X, y)
    @test clf.model.n_classes == 4
    @test size(predict_proba(clf, X)) == (n, 4)
end

@testset "MultiClassOneVsAll — rejected by MichiBoostRegressor" begin
    Random.seed!(3)
    X = randn(60, 3)
    y = Float64.(rand(0:2, 60))
    reg = MichiBoostRegressor(; iterations=5, verbose=false)
    @test_throws ErrorException fit!(reg, X, y; loss_function=MultiClassOneVsAll())
end

@testset "MultiClassOneVsAll — save / load round-trip" begin
    Random.seed!(4)
    n = 300
    X = randn(n, 4)
    y = Float64.([argmax((X[i, 1], X[i, 2], X[i, 3]) .+ 0.4 .* randn(3)) - 1 for i in 1:n])
    clf = MichiBoostClassifier(;
        iterations=40, depth=2, loss_function=MultiClassOneVsAll(), verbose=false
    )
    fit!(clf, X, y)
    before = predict_proba(clf, X)
    path = tempname() * ".jld2"
    try
        save_model(clf, path)
        loaded = load_model(path)
        # `predict` on a raw multiclass model returns the probability matrix.
        @test predict(loaded, Pool(X)) ≈ before
    finally
        rm(path; force=true)
    end
end

# --- RMSEWithUncertainty ----------------------------------------------------

# Heteroscedastic fixture: noise scale grows with |X[:,1]|.
function _uncertainty_fixture(; n=600, seed=0)
    Random.seed!(seed)
    X = randn(n, 4)
    y = 2.0 .* X[:, 1] .+ (0.3 .+ abs.(X[:, 1])) .* randn(n)
    return X, y
end

@testset "RMSEWithUncertainty — outputs mean and positive std" begin
    X, y = _uncertainty_fixture()
    m = MichiBoostRegressor(;
        iterations=120,
        depth=3,
        learning_rate=0.1,
        loss_function=RMSEWithUncertainty(),
        verbose=false,
        random_seed=1,
    )
    fit!(m, X, y)
    out = predict(m, X)
    @test size(out) == (length(y), 2)
    @test all(out[:, 2] .> 0)              # std positive via exp link
    @test cor(out[:, 1], y) > 0.8          # mean column tracks the target
end

@testset "RMSEWithUncertainty — captures heteroscedasticity" begin
    X, y = _uncertainty_fixture(; n=1000, seed=5)
    m = MichiBoostRegressor(;
        iterations=200,
        depth=3,
        learning_rate=0.1,
        loss_function=RMSEWithUncertainty(),
        verbose=false,
        random_seed=2,
    )
    fit!(m, X, y)
    σ = predict(m, X)[:, 2]
    hi = abs.(X[:, 1]) .> 1.0
    @test mean(σ[hi]) > mean(σ[.!hi])      # wider intervals where noise is larger
end

@testset "RMSEWithUncertainty — early stopping with default metric" begin
    X, y = _uncertainty_fixture(; n=800, seed=6)
    tr = Pool(X[1:600, :]; label=y[1:600])
    va = Pool(X[601:end, :]; label=y[601:end])
    m = MichiBoostRegressor(;
        iterations=500,
        depth=3,
        early_stopping_rounds=10,
        loss_function=RMSEWithUncertainty(),
        verbose=false,
        random_seed=3,
    )
    fit!(m, tr; eval_set=va)
    @test length(m.model.trees) <= 500
    @test size(predict(m, Pool(X)), 2) == 2
end

@testset "RMSEWithUncertainty — save / load round-trip" begin
    X, y = _uncertainty_fixture(; n=300, seed=7)
    m = MichiBoostRegressor(;
        iterations=60, depth=2, loss_function=RMSEWithUncertainty(), verbose=false
    )
    fit!(m, X, y)
    before = predict(m, X)
    path = tempname() * ".jld2"
    try
        save_model(m, path)
        loaded = load_model(path)
        @test predict(loaded, Pool(X)) ≈ before
    finally
        rm(path; force=true)
    end
end

@testset "RMSEWithUncertainty — rejected by MichiBoostClassifier" begin
    X, y = _uncertainty_fixture(; n=80, seed=8)
    clf = MichiBoostClassifier(; iterations=5, verbose=false)
    @test_throws ErrorException fit!(clf, X, y; loss_function=RMSEWithUncertainty())
end

@testset "RMSEWithUncertainty — staged_predict and eval_metric error" begin
    X, y = _uncertainty_fixture(; n=200, seed=9)
    m = MichiBoostRegressor(;
        iterations=20, depth=2, loss_function=RMSEWithUncertainty(), verbose=false
    )
    fit!(m, X, y)
    @test_throws ErrorException staged_predict(m, X)

    tr = Pool(X[1:150, :]; label=y[1:150])
    va = Pool(X[151:end, :]; label=y[151:end])
    m2 = MichiBoostRegressor(;
        iterations=50,
        depth=2,
        early_stopping_rounds=5,
        loss_function=RMSEWithUncertainty(),
        eval_metric="RMSE",
        verbose=false,
    )
    @test_throws ErrorException fit!(m2, tr; eval_set=va)
end
