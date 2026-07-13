using MichiBoost
using Random
using Statistics
using Test

# Count monotonicity violations: sweep `col` across a grid for many random base
# rows and check the prediction never moves against `sign`.
function _mono_violations(predict_fn, col, sign; nfeat=4, ntrials=200, seed=100)
    Random.seed!(seed)
    grid = collect(-3.0:0.25:3.0)
    v = 0
    for _ in 1:ntrials
        base = randn(nfeat)
        prev = nothing
        for g in grid
            row = copy(base)
            row[col] = g
            p = predict_fn(reshape(row, 1, nfeat))[1]
            if prev !== nothing
                d = p - prev
                (sign > 0 && d < -1e-9) && (v += 1)
                (sign < 0 && d > 1e-9) && (v += 1)
            end
            prev = p
        end
    end
    return v
end

@testset "monotone — regression, increasing and decreasing" begin
    Random.seed!(0)
    n = 800
    X = randn(n, 4)
    y = 1.5 .* X[:, 1] .- 1.2 .* X[:, 2] .+ 0.5 .* X[:, 3] .+ randn(n) .* 0.3

    m = MichiBoostRegressor(;
        iterations=200,
        depth=4,
        learning_rate=0.1,
        verbose=false,
        monotone_constraints=Dict(1 => 1, 2 => -1),
        random_seed=1,
    )
    fit!(m, X, y)
    @test _mono_violations(x -> predict(m, x), 1, 1) == 0
    @test _mono_violations(x -> predict(m, x), 2, -1) == 0
end

@testset "monotone — unconstrained model can violate (sanity)" begin
    Random.seed!(0)
    n = 800
    X = randn(n, 4)
    y = 1.5 .* X[:, 1] .- 1.2 .* X[:, 2] .+ randn(n) .* 0.3
    m = MichiBoostRegressor(;
        iterations=200, depth=4, learning_rate=0.1, verbose=false, random_seed=1
    )
    fit!(m, X, y)
    # Feature 3 is pure noise in y, so an unconstrained fit wiggles in it.
    @test _mono_violations(x -> predict(m, x), 3, 1) > 0
end

@testset "monotone — keeps a good fit when data agrees" begin
    Random.seed!(1)
    n = 1000
    X = randn(n, 3)
    y = 2.0 .* X[:, 1] .+ 0.5 .* X[:, 2] .+ randn(n) .* 0.2

    con = MichiBoostRegressor(;
        iterations=200,
        depth=4,
        learning_rate=0.1,
        verbose=false,
        monotone_constraints=Dict(1 => 1),
        random_seed=2,
    )
    unc = MichiBoostRegressor(;
        iterations=200, depth=4, learning_rate=0.1, verbose=false, random_seed=2
    )
    fit!(con, X, y)
    fit!(unc, X, y)
    rmse(m) = sqrt(mean((predict(m, X) .- y) .^ 2))
    # The constraint matches the truth, so the fit stays strong. Guaranteed
    # monotonicity still costs some accuracy (subtrees are separated at
    # midpoints), so allow modest slack over the unconstrained fit.
    @test cor(predict(con, X), y) > 0.95
    @test rmse(con) < rmse(unc) * 1.3
end

@testset "monotone — binary classification probabilities are monotone" begin
    Random.seed!(2)
    n = 800
    X = randn(n, 3)
    y = Float64.(1.5 .* X[:, 1] .+ randn(n) .> 0)

    clf = MichiBoostClassifier(;
        iterations=150,
        depth=3,
        learning_rate=0.1,
        verbose=false,
        monotone_constraints=Dict(1 => 1),
        random_seed=3,
    )
    fit!(clf, X, y)
    prob(x) = predict_proba(clf, x)
    @test _mono_violations(prob, 1, 1; nfeat=3) == 0
end

@testset "monotone — accepts feature names and vector form" begin
    Random.seed!(3)
    n = 400
    X = randn(n, 3)
    y = X[:, 1] .- X[:, 2] .+ randn(n) .* 0.2
    pool = Pool(X; label=y, feature_names=[:a, :b, :c])

    by_name = MichiBoostRegressor(;
        iterations=120,
        depth=3,
        verbose=false,
        monotone_constraints=Dict(:a => 1, :b => -1),
        random_seed=4,
    )
    fit!(by_name, pool)
    @test _mono_violations(x -> predict(by_name, x), 1, 1; nfeat=3) == 0
    @test _mono_violations(x -> predict(by_name, x), 2, -1; nfeat=3) == 0

    by_vec = MichiBoostRegressor(;
        iterations=120,
        depth=3,
        verbose=false,
        monotone_constraints=[1, -1, 0],
        random_seed=4,
    )
    fit!(by_vec, X, y)
    @test _mono_violations(x -> predict(by_vec, x), 1, 1; nfeat=3) == 0
end

@testset "monotone — errors" begin
    Random.seed!(4)
    n = 200
    X = randn(n, 3)
    y = randn(n)

    # Bad sign.
    m1 = MichiBoostRegressor(;
        iterations=5, verbose=false, monotone_constraints=Dict(1 => 2)
    )
    @test_throws ErrorException fit!(m1, X, y)

    # Unknown feature name.
    m2 = MichiBoostRegressor(;
        iterations=5, verbose=false, monotone_constraints=Dict(:nope => 1)
    )
    @test_throws ErrorException fit!(m2, X, y)

    # Wrong vector length.
    m3 = MichiBoostRegressor(; iterations=5, verbose=false, monotone_constraints=[1, 0])
    @test_throws ErrorException fit!(m3, X, y)

    # Constraint on a categorical feature.
    cats = rand(["a", "b", "c"], n)
    pool = Pool(hcat(cats, X); cat_features=[1], label=y)
    m4 = MichiBoostRegressor(;
        iterations=5, verbose=false, monotone_constraints=Dict(1 => 1)
    )
    @test_throws ErrorException fit!(m4, pool)
end

@testset "monotone — rejected by multiclass and refinement losses" begin
    Random.seed!(5)
    n = 300
    X = randn(n, 3)
    ymc = Float64.(rand(0:2, n))
    mc = MichiBoostClassifier(;
        iterations=5, verbose=false, monotone_constraints=Dict(1 => 1)
    )
    @test_throws ErrorException fit!(mc, X, ymc)

    yreg = X[:, 1] .+ randn(n) .* 0.2
    mae = MichiBoostRegressor(;
        iterations=5, verbose=false, loss_function="MAE", monotone_constraints=Dict(1 => 1)
    )
    @test_throws ErrorException fit!(mae, X, yreg)
end

@testset "monotone — survives save / load" begin
    Random.seed!(6)
    n = 500
    X = randn(n, 3)
    y = X[:, 1] .+ randn(n) .* 0.3
    m = MichiBoostRegressor(;
        iterations=150, depth=3, verbose=false, monotone_constraints=Dict(1 => 1)
    )
    fit!(m, X, y)
    path = tempname() * ".jld2"
    try
        save_model(m, path)
        loaded = load_model(path)
        @test _mono_violations(x -> predict(loaded, Pool(x)), 1, 1; nfeat=3) == 0
    finally
        rm(path; force=true)
    end
end
