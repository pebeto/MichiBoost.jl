using MichiBoost
using Random
using Test

@testset "init_model — regression continuation matches single fit" begin
    Random.seed!(0)
    X = randn(200, 5)
    y = X[:, 1] .+ 0.5 .* X[:, 2] .+ randn(200) .* 0.1

    base = MichiBoostRegressor(;
        iterations=20, depth=3, learning_rate=0.1, random_seed=42, verbose=false
    )
    fit!(base, X, y)

    cont = MichiBoostRegressor(;
        iterations=30, depth=3, learning_rate=0.1, random_seed=42, verbose=false
    )
    fit!(cont, X, y; init_model=base)

    @test length(cont.model.trees) == length(base.model.trees) + 30
    @test cont.model.initial_pred == base.model.initial_pred
    # Continuation must keep base's predictions stable as a prefix; the tail
    # adds the new trees' contributions.
    full = MichiBoostRegressor(;
        iterations=20 + 30, depth=3, learning_rate=0.1, random_seed=42, verbose=false
    )
    fit!(full, X, y)
    @test cont.model.trees[1:20] == base.model.trees
    @test predict(cont, X) ≈ predict(full, X) atol = 1e-10
end

@testset "init_model — predictions match init_model when no new trees added" begin
    Random.seed!(1)
    X = randn(150, 4)
    y = X[:, 1] .+ randn(150) .* 0.2

    base = MichiBoostRegressor(;
        iterations=15, depth=2, learning_rate=0.05, random_seed=7, verbose=false
    )
    fit!(base, X, y)

    cont = MichiBoostRegressor(;
        iterations=0, depth=2, learning_rate=0.05, random_seed=7, verbose=false
    )
    fit!(cont, X, y; init_model=base)

    @test length(cont.model.trees) == length(base.model.trees)
    @test predict(cont, X) ≈ predict(base, X)
end

@testset "init_model — binary classification continuation" begin
    Random.seed!(2)
    X = randn(300, 6)
    y = Float64.(X[:, 1] .+ X[:, 2] .> 0)

    base = MichiBoostClassifier(;
        iterations=10, depth=3, learning_rate=0.1, random_seed=3, verbose=false
    )
    fit!(base, X, y)

    cont = MichiBoostClassifier(;
        iterations=20, depth=3, learning_rate=0.1, random_seed=3, verbose=false
    )
    fit!(cont, X, y; init_model=base)

    @test length(cont.model.trees) == length(base.model.trees) + 20
    @test cont.model.n_classes == 2
    p_cont = predict_proba(cont, X)
    @test all(0 .<= p_cont .<= 1)
end

@testset "init_model — multiclass continuation" begin
    Random.seed!(3)
    n, k = 200, 3
    X = randn(n, 5)
    raw = [(X[i, 1] + 0.5 * randn(), X[i, 2] + 0.5 * randn(), randn()) for i in 1:n]
    y = Float64.([argmax(r) - 1 for r in raw])

    base = MichiBoostClassifier(;
        iterations=10, depth=2, learning_rate=0.1, random_seed=9, verbose=false
    )
    fit!(base, X, y)

    cont = MichiBoostClassifier(;
        iterations=15, depth=2, learning_rate=0.1, random_seed=9, verbose=false
    )
    fit!(cont, X, y; init_model=base)

    @test cont.model.is_multiclass
    @test cont.model.n_classes == k
    @test length(cont.model.trees) == length(base.model.trees) + 15
    @test size(predict_proba(cont, X)) == (n, k)
end

@testset "init_model — accepts wrapper, not just MichiBoostModel" begin
    Random.seed!(4)
    X = randn(80, 3)
    y = X[:, 1] .+ randn(80) .* 0.1

    base = MichiBoostRegressor(; iterations=5, depth=2, verbose=false)
    fit!(base, X, y)

    cont = MichiBoostRegressor(; iterations=5, depth=2, verbose=false)
    fit!(cont, X, y; init_model=base)
    @test length(cont.model.trees) == length(base.model.trees) + 5
end

@testset "init_model — errors on task mismatch" begin
    Random.seed!(5)
    X = randn(100, 4)
    y_reg = X[:, 1] .+ randn(100) .* 0.1
    y_cls = Float64.(X[:, 1] .> 0)

    reg = MichiBoostRegressor(; iterations=5, depth=2, verbose=false)
    fit!(reg, X, y_reg)

    clf = MichiBoostClassifier(; iterations=5, depth=2, verbose=false)
    @test_throws ErrorException fit!(clf, X, y_cls; init_model=reg)
end

@testset "init_model — errors on feature-shape mismatch" begin
    Random.seed!(6)
    X5 = randn(80, 5)
    X3 = randn(80, 3)
    y = randn(80)

    base = MichiBoostRegressor(; iterations=5, depth=2, verbose=false)
    fit!(base, X5, y)

    cont = MichiBoostRegressor(; iterations=5, depth=2, verbose=false)
    @test_throws ErrorException fit!(cont, X3, y; init_model=base)
end

@testset "init_model — errors on bad type" begin
    Random.seed!(7)
    X = randn(40, 2)
    y = randn(40)

    cont = MichiBoostRegressor(; iterations=5, depth=2, verbose=false)
    @test_throws ErrorException fit!(cont, X, y; init_model="not a model")
end

@testset "init_model — categorical features reuse encoder" begin
    Random.seed!(8)
    n = 200
    cats = rand(["a", "b", "c"], n)
    nums = randn(n, 2)
    y = Float64.([c == "a" ? 0 : c == "b" ? 1 : 0 for c in cats]) .+ 0.1 .* randn(n)

    pool = Pool(hcat(cats, nums); cat_features=[1], label=y)

    base = MichiBoostRegressor(; iterations=10, depth=2, verbose=false)
    fit!(base, pool)

    pool2 = Pool(hcat(cats, nums); cat_features=[1], label=y)
    cont = MichiBoostRegressor(; iterations=10, depth=2, verbose=false)
    fit!(cont, pool2; init_model=base)

    @test length(cont.model.trees) == length(base.model.trees) + 10
    @test cont.model.encoder === base.model.encoder
end

@testset "init_model — early stopping seeded from inherited predictions" begin
    Random.seed!(9)
    X = randn(400, 5)
    y = X[:, 1] .+ X[:, 2] .+ randn(400) .* 0.2
    train_pool = Pool(X[1:300, :]; label=y[1:300])
    val_pool = Pool(X[301:end, :]; label=y[301:end])

    base = MichiBoostRegressor(; iterations=20, depth=3, verbose=false)
    fit!(base, train_pool)

    cont = MichiBoostRegressor(;
        iterations=100, depth=3, early_stopping_rounds=5, verbose=false
    )
    fit!(cont, train_pool; init_model=base, eval_set=val_pool)
    @test length(cont.model.trees) <= length(base.model.trees) + 100
    @test cont.model.best_iteration >= length(base.model.trees)
end
