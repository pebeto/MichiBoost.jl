using MichiBoost
using MichiBoost: LossFunction
using Random
using Statistics
using Test

import MichiBoost: gradient_hessian!, initial_prediction, loss, task_type, link_inverse

# A user-defined loss that should match the built-in RMSE numerically when the
# engine substitutes it. Used as a sanity check that the public API hooks
# match the internal one.
struct UserRMSE <: LossFunction end

function gradient_hessian!(
    g::AbstractVector,
    h::AbstractVector,
    ::UserRMSE,
    y::AbstractVector,
    pred::AbstractVector,
    _scratch,
)
    g .= y .- pred
    fill!(h, 1.0)
    return nothing
end

initial_prediction(::UserRMSE, y::AbstractVector) = mean(y)
loss(::UserRMSE, y::AbstractVector, pred::AbstractVector) = sqrt(mean((y .- pred) .^ 2))

# Huber loss: smooth-L2 inside |residual| < δ, linear outside.
struct HuberLoss <: LossFunction
    delta::Float64
end

function gradient_hessian!(
    g::AbstractVector,
    h::AbstractVector,
    lf::HuberLoss,
    y::AbstractVector,
    pred::AbstractVector,
    _scratch,
)
    δ = lf.delta
    @inbounds for i in eachindex(y)
        r = y[i] - pred[i]
        if abs(r) <= δ
            g[i] = r
            h[i] = 1.0
        else
            g[i] = δ * sign(r)
            h[i] = 1e-3   # tiny non-zero so leaf-value Newton step is finite
        end
    end
    return nothing
end

initial_prediction(::HuberLoss, y::AbstractVector) = median(y)

function loss(lf::HuberLoss, y::AbstractVector, pred::AbstractVector)
    δ = lf.delta
    s = 0.0
    @inbounds for i in eachindex(y)
        r = y[i] - pred[i]
        s += abs(r) <= δ ? 0.5 * r^2 : δ * (abs(r) - 0.5 * δ)
    end
    return s / length(y)
end

@testset "Custom LossFunction: UserRMSE matches built-in RMSE" begin
    Random.seed!(0)
    X = randn(80, 3)
    y = X[:, 1] .+ 0.5 .* X[:, 2] .+ randn(80) .* 0.1

    builtin = MichiBoostRegressor(;
        iterations=15, depth=3, learning_rate=0.1, random_seed=42, verbose=false
    )
    fit!(builtin, X, y)
    pred_builtin = predict(builtin, X)

    custom = MichiBoostRegressor(;
        iterations=15,
        depth=3,
        learning_rate=0.1,
        random_seed=42,
        verbose=false,
        loss_function=UserRMSE(),
    )
    fit!(custom, X, y)
    pred_custom = predict(custom, X)

    @test pred_custom ≈ pred_builtin
end

@testset "Custom LossFunction: Huber fits and predicts sensibly" begin
    Random.seed!(1)
    n = 120
    X = randn(n, 2)
    y = X[:, 1] .+ randn(n) .* 0.1

    # Inject a few large outliers; Huber should be more robust than RMSE.
    outlier_idx = [10, 30, 60, 90]
    y[outlier_idx] .+= 50.0

    huber = MichiBoostRegressor(;
        iterations=40,
        depth=3,
        learning_rate=0.1,
        random_seed=0,
        verbose=false,
        loss_function=HuberLoss(1.0),
    )
    fit!(huber, X, y)
    preds = predict(huber, X)

    # On non-outlier rows, MAE should beat the trivial mean-prediction baseline.
    inliers = setdiff(1:n, outlier_idx)
    mae_huber = mean(abs.(preds[inliers] .- y[inliers]))
    mae_baseline = mean(abs.(mean(y) .- y[inliers]))
    @test mae_huber < mae_baseline
end

@testset "Custom LossFunction: regression loss rejected by MichiBoostClassifier" begin
    Random.seed!(2)
    X = randn(40, 2)
    y = Float64.(X[:, 1] .> 0)

    clf = MichiBoostClassifier(; iterations=5, verbose=false, loss_function=UserRMSE())
    @test_throws ErrorException fit!(clf, X, y)
end

# Custom binary classification loss
struct UserBinaryLogloss <: LossFunction end
task_type(::UserBinaryLogloss) = :binary

@inline _user_sigmoid(x::Real) = 1.0 / (1.0 + exp(-x))
link_inverse(::UserBinaryLogloss, raw::AbstractVector) = _user_sigmoid.(raw)

function gradient_hessian!(
    g::AbstractVector,
    h::AbstractVector,
    ::UserBinaryLogloss,
    y::AbstractVector,
    pred::AbstractVector,
    scratch::AbstractVector,
)
    scratch .= _user_sigmoid.(pred)
    g .= y .- scratch
    h .= scratch .* (1.0 .- scratch)
    return nothing
end

function initial_prediction(::UserBinaryLogloss, y::AbstractVector)
    p = clamp(mean(y), 1e-7, 1.0 - 1e-7)
    return log(p / (1.0 - p))
end

function loss(::UserBinaryLogloss, y::AbstractVector, pred::AbstractVector)
    p = clamp.(_user_sigmoid.(pred), 1e-15, 1.0 - 1e-15)
    return -mean(y .* log.(p) .+ (1.0 .- y) .* log.(1.0 .- p))
end

@testset "Custom LossFunction: UserBinaryLogloss matches built-in Logloss" begin
    Random.seed!(10)
    X = randn(120, 3)
    y = Float64.(X[:, 1] .+ 0.5 .* X[:, 2] .> 0)

    builtin = MichiBoostClassifier(;
        iterations=15, depth=3, learning_rate=0.1, random_seed=7, verbose=false
    )
    fit!(builtin, X, y)
    proba_builtin = predict_proba(builtin, X)
    classes_builtin = predict(builtin, X)

    custom = MichiBoostClassifier(;
        iterations=15,
        depth=3,
        learning_rate=0.1,
        random_seed=7,
        verbose=false,
        loss_function=UserBinaryLogloss(),
    )
    fit!(custom, X, y)
    proba_custom = predict_proba(custom, X)
    classes_custom = predict(custom, X)

    @test proba_custom ≈ proba_builtin
    @test classes_custom == classes_builtin
end

@testset "Custom LossFunction: binary loss rejected by MichiBoostRegressor" begin
    Random.seed!(11)
    X = randn(40, 2)
    y = Float64.(X[:, 1] .> 0)
    reg = MichiBoostRegressor(;
        iterations=5, verbose=false, loss_function=UserBinaryLogloss()
    )
    @test_throws ErrorException fit!(reg, X, y)
end

# Custom multiclass loss
struct UserMultiClass <: LossFunction end
task_type(::UserMultiClass) = :multiclass

function _row_softmax!(out::AbstractMatrix, raw::AbstractMatrix)
    for i in axes(raw, 1)
        m = maximum(view(raw, i, :))
        s = 0.0
        for j in axes(raw, 2)
            e = exp(raw[i, j] - m)
            out[i, j] = e
            s += e
        end
        for j in axes(raw, 2)
            out[i, j] /= s
        end
    end
    return out
end

link_inverse(::UserMultiClass, raw::AbstractMatrix) = _row_softmax!(similar(raw), raw)

function gradient_hessian!(
    g::AbstractMatrix,
    h::AbstractMatrix,
    ::UserMultiClass,
    y_onehot::AbstractMatrix,
    pred::AbstractMatrix,
    scratch::AbstractMatrix,
)
    _row_softmax!(scratch, pred)
    g .= y_onehot .- scratch
    h .= scratch .* (1.0 .- scratch)
    return nothing
end

function initial_prediction(::UserMultiClass, y_onehot::AbstractMatrix)
    class_probs = clamp.(vec(mean(y_onehot; dims=1)), 1e-7, 1.0 - 1e-7)
    return log.(class_probs)
end

function loss(::UserMultiClass, y_onehot::AbstractMatrix, pred::AbstractMatrix)
    probs = clamp.(_row_softmax!(similar(pred), pred), 1e-15, 1.0)
    return -mean(sum(y_onehot .* log.(probs); dims=2))
end

@testset "Custom LossFunction: UserMultiClass matches built-in MultiClass" begin
    Random.seed!(20)
    X = randn(150, 3)
    # 3-class label by row sign-pattern
    raw = X[:, 1] .+ 0.5 .* X[:, 2]
    y = [r < -0.3 ? 0.0 : (r < 0.3 ? 1.0 : 2.0) for r in raw]

    builtin = MichiBoostClassifier(;
        iterations=15, depth=3, learning_rate=0.1, random_seed=11, verbose=false
    )
    fit!(builtin, X, y)
    proba_builtin = predict_proba(builtin, X)

    custom = MichiBoostClassifier(;
        iterations=15,
        depth=3,
        learning_rate=0.1,
        random_seed=11,
        verbose=false,
        loss_function=UserMultiClass(),
    )
    fit!(custom, X, y)
    proba_custom = predict_proba(custom, X)

    @test size(proba_custom) == size(proba_builtin)
    @test proba_custom ≈ proba_builtin
end

@testset "Custom LossFunction: binary loss errors when data has >2 classes" begin
    Random.seed!(21)
    X = randn(60, 2)
    y = [Float64(i % 3) for i in 1:60]   # three classes
    clf = MichiBoostClassifier(;
        iterations=5, verbose=false, loss_function=UserBinaryLogloss()
    )
    @test_throws ErrorException fit!(clf, X, y)
end

@testset "Custom LossFunction: round-trip through save_model / load_model (binary)" begin
    Random.seed!(22)
    X = randn(80, 2)
    y = Float64.(X[:, 1] .> 0)

    clf = MichiBoostClassifier(;
        iterations=10,
        depth=2,
        learning_rate=0.1,
        random_seed=0,
        verbose=false,
        loss_function=UserBinaryLogloss(),
    )
    fit!(clf, X, y)
    proba_before = predict_proba(clf, X)

    tmp = tempname() * ".jld2"
    save_model(clf, tmp)
    loaded = load_model(tmp)
    proba_after = predict(loaded, Pool(X))   # raw model: predict applies link
    @test proba_after ≈ proba_before
    rm(tmp; force=true)
end

@testset "Custom LossFunction: works with cv()" begin
    Random.seed!(3)
    X = randn(60, 2)
    y = X[:, 1] .+ randn(60) .* 0.1
    pool = Pool(X; label=y)

    result = cv(
        pool;
        params=Dict(:iterations => 10, :depth => 2, :verbose => false),
        loss_function=UserRMSE(),
        fold_count=3,
        random_seed=0,
    )
    @test length(result.train_loss) == 3
    @test all(isfinite, result.test_loss)
end
