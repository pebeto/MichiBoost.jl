using MichiBoost
using Random
using Test

@testset "snapshot — periodic save during training" begin
    Random.seed!(0)
    X = randn(120, 4)
    y = X[:, 1] .+ randn(120) .* 0.1
    snapfile = tempname() * ".jld2"

    model = MichiBoostRegressor(;
        iterations=25,
        depth=2,
        learning_rate=0.1,
        verbose=false,
        snapshot_path=snapfile,
        snapshot_interval=10,
    )
    fit!(model, X, y)
    try
        @test isfile(snapfile)
        snap = load_model(snapfile)
        # Final snapshot (after the loop) captures the full 25 trees.
        @test length(snap.trees) == 25
        @test snap.initial_pred == model.model.initial_pred
    finally
        rm(snapfile; force=true)
    end
end

@testset "snapshot — no-op without path" begin
    Random.seed!(1)
    X = randn(60, 3)
    y = randn(60)

    model = MichiBoostRegressor(; iterations=10, depth=2, verbose=false)
    fit!(model, X, y)
    @test length(model.model.trees) == 10
end

@testset "snapshot — resumes via init_model with matching predictions" begin
    Random.seed!(2)
    X = randn(150, 4)
    y = X[:, 1] .+ randn(150) .* 0.1
    snapfile = tempname() * ".jld2"

    # Train 20 iterations and save the intermediate state at iter 10. The
    # in-loop save is the snapshot we want to resume from. We then truncate
    # the resumed run to 10 new iters so the total matches a fresh 20-iter fit.
    base = MichiBoostRegressor(;
        iterations=10,
        depth=2,
        learning_rate=0.1,
        verbose=false,
        snapshot_path=snapfile,
        snapshot_interval=10,
    )
    fit!(base, X, y)

    full = MichiBoostRegressor(; iterations=20, depth=2, learning_rate=0.1, verbose=false)
    fit!(full, X, y)

    try
        snap = load_model(snapfile)
        @test length(snap.trees) == 10

        resumed = MichiBoostRegressor(;
            iterations=10, depth=2, learning_rate=0.1, verbose=false
        )
        fit!(resumed, X, y; init_model=snap)
        @test length(resumed.model.trees) == 20
        @test predict(resumed, X) ≈ predict(full, X) atol = 1e-10
    finally
        rm(snapfile; force=true)
    end
end

@testset "snapshot — captures ES-truncated state on early break" begin
    Random.seed!(3)
    X = randn(300, 5)
    y = X[:, 1] .+ X[:, 2] .+ randn(300) .* 0.05
    train_pool = Pool(X[1:200, :]; label=y[1:200])
    val_pool = Pool(X[201:end, :]; label=y[201:end])
    snapfile = tempname() * ".jld2"

    model = MichiBoostRegressor(;
        iterations=500,
        depth=3,
        early_stopping_rounds=3,
        verbose=false,
        snapshot_path=snapfile,
        snapshot_interval=20,
        random_seed=1,
    )
    fit!(model, train_pool; eval_set=val_pool)
    try
        @test isfile(snapfile)
        snap = load_model(snapfile)
        # Final on-disk snapshot reflects post-ES truncation, not the last
        # interval before the break. Tree count must match the live model.
        @test length(snap.trees) == length(model.model.trees)
        @test snap.best_iteration == model.model.best_iteration
    finally
        rm(snapfile; force=true)
    end
end

@testset "snapshot — interval=1 saves every iteration" begin
    Random.seed!(4)
    X = randn(40, 2)
    y = randn(40)
    snapfile = tempname() * ".jld2"

    model = MichiBoostRegressor(;
        iterations=5, depth=2, verbose=false, snapshot_path=snapfile, snapshot_interval=1
    )
    fit!(model, X, y)
    try
        snap = load_model(snapfile)
        @test length(snap.trees) == 5
    finally
        rm(snapfile; force=true)
    end
end

@testset "snapshot — non-multiple total writes final state" begin
    # iterations=13, interval=5: in-loop saves fire at iter 5 and 10. The
    # final iter (13) is not a multiple, so the after-loop save catches it.
    Random.seed!(5)
    X = randn(80, 3)
    y = randn(80)
    snapfile = tempname() * ".jld2"

    model = MichiBoostRegressor(;
        iterations=13, depth=2, verbose=false, snapshot_path=snapfile, snapshot_interval=5
    )
    fit!(model, X, y)
    try
        snap = load_model(snapfile)
        @test length(snap.trees) == 13
    finally
        rm(snapfile; force=true)
    end
end

@testset "snapshot — binary classification round-trip" begin
    Random.seed!(6)
    X = randn(150, 4)
    y = Float64.(X[:, 1] .> 0)
    snapfile = tempname() * ".jld2"

    clf = MichiBoostClassifier(;
        iterations=12, depth=3, verbose=false, snapshot_path=snapfile, snapshot_interval=4
    )
    fit!(clf, X, y)
    try
        snap = load_model(snapfile)
        @test snap.n_classes == 2
        @test !snap.is_multiclass
        @test length(snap.trees) == 12
    finally
        rm(snapfile; force=true)
    end
end

@testset "snapshot — multiclass round-trip" begin
    Random.seed!(7)
    n = 180
    X = randn(n, 4)
    raw = [(X[i, 1] + 0.3 * randn(), X[i, 2] + 0.3 * randn(), randn()) for i in 1:n]
    y = Float64.([argmax(r) - 1 for r in raw])
    snapfile = tempname() * ".jld2"

    clf = MichiBoostClassifier(;
        iterations=10, depth=2, verbose=false, snapshot_path=snapfile, snapshot_interval=5
    )
    fit!(clf, X, y)
    try
        snap = load_model(snapfile)
        @test snap.is_multiclass
        @test snap.n_classes == 3
        @test length(snap.trees) == 10
    finally
        rm(snapfile; force=true)
    end
end

@testset "snapshot — interval=0 errors" begin
    X = randn(20, 2)
    y = randn(20)
    snapfile = tempname() * ".jld2"

    model = MichiBoostRegressor(;
        iterations=5, depth=2, verbose=false, snapshot_path=snapfile, snapshot_interval=0
    )
    @test_throws ErrorException fit!(model, X, y)
end

@testset "snapshot — negative interval errors" begin
    X = randn(20, 2)
    y = randn(20)
    snapfile = tempname() * ".jld2"

    model = MichiBoostRegressor(;
        iterations=5, depth=2, verbose=false, snapshot_path=snapfile, snapshot_interval=-5
    )
    @test_throws ErrorException fit!(model, X, y)
end
