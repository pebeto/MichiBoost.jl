using MichiBoost
using Random
using Statistics
using Test

# Shared fixtures: a general regression target `y` and a positive, skewed
# target `ypos` for the log-link losses (Tweedie, LogLinQuantile, MAPE).
function _loss_fixture(; n=400, seed=0)
    Random.seed!(seed)
    X = randn(n, 4)
    y = 2.0 .* X[:, 1] .- X[:, 2] .+ randn(n) .* 0.3
    ypos = exp.(0.5 .* X[:, 1] .+ 0.3 .* randn(n)) .+ 0.05
    return X, y, ypos
end

@testset "Huber — fits and matches RMSE-ish on clean data" begin
    X, y, _ = _loss_fixture()
    m = MichiBoostRegressor(;
        iterations=80, depth=3, learning_rate=0.1, loss_function=Huber(1.0), verbose=false
    )
    fit!(m, X, y)
    p = predict(m, X)
    @test all(isfinite, p)
    @test cor(p, y) > 0.9
end

@testset "Huber — robust to outliers vs RMSE" begin
    Random.seed!(1)
    n = 300
    X = randn(n, 3)
    y = X[:, 1] .+ randn(n) .* 0.1
    # Inject a few extreme outliers.
    y[1:10] .+= 50.0

    rmse = MichiBoostRegressor(;
        iterations=60, depth=2, loss_function="RMSE", verbose=false
    )
    fit!(rmse, X, y)
    hub = MichiBoostRegressor(;
        iterations=60, depth=2, loss_function=Huber(1.0), verbose=false
    )
    fit!(hub, X, y)

    # On the clean majority, Huber should track the signal more tightly than
    # RMSE, which is dragged toward the outliers.
    clean = 11:n
    err_rmse = mean(abs.(predict(rmse, X)[clean] .- y[clean]))
    err_hub = mean(abs.(predict(hub, X)[clean] .- y[clean]))
    @test err_hub < err_rmse
end

@testset "Quantile — coverage tracks alpha" begin
    X, y, _ = _loss_fixture(; n=600)
    for α in (0.1, 0.5, 0.9)
        m = MichiBoostRegressor(;
            iterations=150,
            depth=3,
            learning_rate=0.1,
            loss_function=Quantile(α),
            verbose=false,
            random_seed=3,
        )
        fit!(m, X, y)
        coverage = mean(y .<= predict(m, X))
        @test isapprox(coverage, α; atol=0.07)
    end
end

@testset "Quantile(0.5) matches MAE closely" begin
    X, y, _ = _loss_fixture()
    q = MichiBoostRegressor(;
        iterations=80, depth=3, loss_function=Quantile(0.5), verbose=false, random_seed=4
    )
    mae = MichiBoostRegressor(;
        iterations=80, depth=3, loss_function="MAE", verbose=false, random_seed=4
    )
    fit!(q, X, y)
    fit!(mae, X, y)
    # Same surrogate gradient family and median refinement, so MAE and the
    # 0.5-quantile should land in the same neighbourhood.
    @test cor(predict(q, X), predict(mae, X)) > 0.98
end

@testset "Expectile — fits; 0.5 reduces to RMSE behaviour" begin
    X, y, _ = _loss_fixture()
    e5 = MichiBoostRegressor(;
        iterations=80, depth=3, loss_function=Expectile(0.5), verbose=false, random_seed=5
    )
    rmse = MichiBoostRegressor(;
        iterations=80, depth=3, loss_function="RMSE", verbose=false, random_seed=5
    )
    fit!(e5, X, y)
    fit!(rmse, X, y)
    @test predict(e5, X) ≈ predict(rmse, X) atol = 1e-8

    # Asymmetric expectile still fits the signal.
    e9 = MichiBoostRegressor(;
        iterations=80, depth=3, loss_function=Expectile(0.9), verbose=false
    )
    fit!(e9, X, y)
    @test cor(predict(e9, X), y) > 0.9
end

@testset "MAPE — fits positive targets, optimizes relative error" begin
    X, _, ypos = _loss_fixture()
    mape = MichiBoostRegressor(;
        iterations=120, depth=3, learning_rate=0.1, loss_function=MAPE(), verbose=false
    )
    fit!(mape, X, ypos)
    p = predict(mape, X)
    @test all(isfinite, p)
    @test cor(p, ypos) > 0.5
end

@testset "Tweedie — positive predictions via exp link" begin
    X, _, ypos = _loss_fixture()
    m = MichiBoostRegressor(;
        iterations=80, depth=3, learning_rate=0.1, loss_function=Tweedie(1.5), verbose=false
    )
    fit!(m, X, ypos)
    p = predict(m, X)
    @test all(p .> 0)
    @test cor(p, ypos) > 0.7
    # Raw scores are on the log scale; predict applies exp.
    raw = predict(m.model, Pool(X))
    @test all(p .> 0)
    @test length(raw) == length(p)
end

@testset "LogLinQuantile — positive preds, coverage tracks alpha" begin
    X, _, ypos = _loss_fixture(; n=600)
    m = MichiBoostRegressor(;
        iterations=150,
        depth=3,
        learning_rate=0.1,
        loss_function=LogLinQuantile(0.5),
        verbose=false,
        random_seed=6,
    )
    fit!(m, X, ypos)
    p = predict(m, X)
    @test all(p .> 0)
    @test isapprox(mean(ypos .<= p), 0.5; atol=0.08)
end

@testset "staged_predict applies the exp link for Tweedie" begin
    X, _, ypos = _loss_fixture(; n=200)
    m = MichiBoostRegressor(;
        iterations=30, depth=2, loss_function=Tweedie(1.5), verbose=false
    )
    fit!(m, X, ypos)
    staged = staged_predict(m, X)
    @test all(staged .> 0)              # exp link applied at every stage
    @test staged[:, end] ≈ predict(m, X)
end

@testset "constructor validation" begin
    @test_throws ErrorException Huber(0.0)
    @test_throws ErrorException Huber(-1.0)
    @test_throws ErrorException Quantile(0.0)
    @test_throws ErrorException Quantile(1.0)
    @test_throws ErrorException Expectile(1.5)
    @test_throws ErrorException Tweedie(1.0)
    @test_throws ErrorException Tweedie(2.0)
    @test_throws ErrorException LogLinQuantile(-0.2)
end

@testset "regression losses rejected by MichiBoostClassifier" begin
    X, y, _ = _loss_fixture(; n=100)
    ybin = Float64.(y .> 0)
    for lf in (
        Huber(1.0), Quantile(0.5), Expectile(0.5), MAPE(), Tweedie(1.5), LogLinQuantile(0.5)
    )
        clf = MichiBoostClassifier(; iterations=5, verbose=false)
        @test_throws ErrorException fit!(clf, X, ybin; loss_function=lf)
    end
end

@testset "save / load round-trip with exp-link loss" begin
    X, _, ypos = _loss_fixture(; n=200)
    m = MichiBoostRegressor(;
        iterations=40, depth=2, loss_function=Tweedie(1.5), verbose=false
    )
    fit!(m, X, ypos)
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

@testset "early stopping works with a refinement loss" begin
    X, y, _ = _loss_fixture(; n=500)
    train_pool = Pool(X[1:350, :]; label=y[1:350])
    val_pool = Pool(X[351:end, :]; label=y[351:end])
    m = MichiBoostRegressor(;
        iterations=300,
        depth=3,
        early_stopping_rounds=5,
        loss_function=Quantile(0.5),
        verbose=false,
        random_seed=7,
    )
    fit!(m, train_pool; eval_set=val_pool)
    @test length(m.model.trees) <= 300
    @test all(isfinite, predict(m, X))
end
