# Head-to-head model quality on held-out data.  For each task we report the
# actual task metric (RMSE / MAE / AUC / log-loss / accuracy) for both
# libraries and assert that MichiBoost stays within a small tolerance of
# CatBoost.  Correlation/agreement gates are intentionally avoided — a model
# can agree with CatBoost 85% of the time and still be bad.
#
#   julia --project=benchmark -t 4 benchmark/correctness.jl

include("common.jl")

using Test

function fmt(x; digits=4)
    x === nothing && return "—"
    isnan(x) && return "NaN"
    return string(round(x; digits=digits))
end

function row(name, mb, cb; digits=4)
    Δ = mb - cb
    sign_ = Δ >= 0 ? "+" : ""
    println("  $(rpad(name, 10))  MichiBoost $(fmt(mb; digits))   CatBoost $(fmt(cb; digits))   Δ $(sign_)$(fmt(Δ; digits))")
end

@testset "Correctness" begin

@testset "Regression (RMSE loss)" begin
    X, y = regression_data(n=2000, p=20)
    X_tr, y_tr, X_te, y_te = train_test_split(X, y)

    cb = cb_train(X_tr, y_tr; loss="RMSE")
    mb = mb_train(X_tr, y_tr; loss="RMSE")

    cb_pred = pyconvert(Vector{Float64}, CatBoost.predict(cb, np.array(X_te)))
    mb_pred = MichiBoost.predict(mb, X_te)

    println("\nRegression (RMSE loss)  n=$(size(X_tr,1))×$(size(X_tr,2))")
    row("RMSE", rmse(mb_pred, y_te), rmse(cb_pred, y_te))
    row("MAE",  mae(mb_pred, y_te),  mae(cb_pred, y_te))
    row("R²",   r2(mb_pred, y_te),   r2(cb_pred, y_te))

    # Allow up to 15% worse RMSE than CatBoost — same-order-of-magnitude quality.
    @test rmse(mb_pred, y_te) <= 1.15 * rmse(cb_pred, y_te)
end

@testset "Regression (MAE loss)" begin
    X, y = regression_data(n=2000, p=20)
    X_tr, y_tr, X_te, y_te = train_test_split(X, y)

    cb = cb_train(X_tr, y_tr; loss="MAE")
    mb = mb_train(X_tr, y_tr; loss="MAE")

    cb_pred = pyconvert(Vector{Float64}, CatBoost.predict(cb, np.array(X_te)))
    mb_pred = MichiBoost.predict(mb, X_te)

    println("\nRegression (MAE loss)  n=$(size(X_tr,1))×$(size(X_tr,2))")
    row("MAE",  mae(mb_pred, y_te),  mae(cb_pred, y_te))
    row("RMSE", rmse(mb_pred, y_te), rmse(cb_pred, y_te))

    @test mae(mb_pred, y_te) <= 1.15 * mae(cb_pred, y_te)
end

@testset "Binary classification" begin
    X, y = binary_data(n=4000, p=15)
    X_tr, y_tr, X_te, y_te = train_test_split(X, y)

    cb = cb_train(X_tr, y_tr; loss="Logloss")
    mb = mb_train(X_tr, y_tr; loss="Logloss")

    cb_prob = pyconvert(Matrix{Float64}, CatBoost.predict_proba(cb, np.array(X_te)))[:, 2]
    mb_prob = MichiBoost.predict(mb, X_te; prediction_type="Probability")

    cb_acc = mean((cb_prob .>= 0.5) .== (y_te .== 1.0))
    mb_acc = mean((mb_prob .>= 0.5) .== (y_te .== 1.0))

    println("\nBinary classification  n=$(size(X_tr,1))×$(size(X_tr,2))")
    row("LogLoss",  binary_logloss(mb_prob, y_te), binary_logloss(cb_prob, y_te))
    row("AUC",      auc(mb_prob, y_te),            auc(cb_prob, y_te))
    row("Accuracy", mb_acc,                        cb_acc)

    # Quality gate: AUC within 0.05, log-loss no more than 15% worse.
    @test auc(mb_prob, y_te) >= auc(cb_prob, y_te) - 0.05
    @test binary_logloss(mb_prob, y_te) <= 1.15 * binary_logloss(cb_prob, y_te)
end

@testset "Multiclass classification (k=3)" begin
    X, y = multiclass_data(n=3000, p=10, k=3)
    X_tr, y_tr, X_te, y_te = train_test_split(X, y)

    cb = cb_train(X_tr, y_tr; loss="MultiClass")
    mb = mb_train(X_tr, y_tr; loss="MultiClass")

    cb_prob = pyconvert(Matrix{Float64}, CatBoost.predict_proba(cb, np.array(X_te)))
    mb_prob = MichiBoost.predict(mb, X_te; prediction_type="Probability")

    y_int = Int.(y_te) .+ 1
    cb_cls = [argmax(cb_prob[i, :]) for i in axes(cb_prob, 1)]
    mb_cls = [argmax(mb_prob[i, :]) for i in axes(mb_prob, 1)]

    println("\nMulticlass (k=3)  n=$(size(X_tr,1))×$(size(X_tr,2))")
    row("LogLoss",  multiclass_logloss(mb_prob, y_int), multiclass_logloss(cb_prob, y_int))
    row("Accuracy", mean(mb_cls .== y_int),             mean(cb_cls .== y_int))

    @test mean(mb_cls .== y_int) >= mean(cb_cls .== y_int) - 0.05
end

@testset "Multiclass classification (k=10)" begin
    X, y = multiclass_data(n=5000, p=15, k=10)
    X_tr, y_tr, X_te, y_te = train_test_split(X, y)

    cb = cb_train(X_tr, y_tr; loss="MultiClass")
    mb = mb_train(X_tr, y_tr; loss="MultiClass")

    cb_prob = pyconvert(Matrix{Float64}, CatBoost.predict_proba(cb, np.array(X_te)))
    mb_prob = MichiBoost.predict(mb, X_te; prediction_type="Probability")

    y_int = Int.(y_te) .+ 1
    cb_cls = [argmax(cb_prob[i, :]) for i in axes(cb_prob, 1)]
    mb_cls = [argmax(mb_prob[i, :]) for i in axes(mb_prob, 1)]

    println("\nMulticlass (k=10)  n=$(size(X_tr,1))×$(size(X_tr,2))")
    row("LogLoss",  multiclass_logloss(mb_prob, y_int), multiclass_logloss(cb_prob, y_int))
    row("Accuracy", mean(mb_cls .== y_int),             mean(cb_cls .== y_int))

    @test mean(mb_cls .== y_int) >= mean(cb_cls .== y_int) - 0.05
end

@testset "Categorical features" begin
    X_num, X_cat, y = categorical_data(n=4000, p_num=5, p_cat=5, k=20)
    rng = MersenneTwister(SEED)
    idx = randperm(rng, length(y))
    n_tr = round(Int, 0.8 * length(y))
    tr, te = idx[1:n_tr], idx[n_tr+1:end]

    _, jl_df_tr, cb_pool_tr, _ = make_cat_frames(X_num[tr, :], X_cat[tr, :], y[tr])
    py_df_te, jl_df_te, _, _   = make_cat_frames(X_num[te, :], X_cat[te, :], y[te])

    cb = cb_train(nothing, y[tr]; loss="Logloss", cb_pool=cb_pool_tr)
    mb = mb_train(jl_df_tr, y[tr]; loss="Logloss")

    cb_prob = pyconvert(Matrix{Float64}, CatBoost.predict_proba(cb, py_df_te))[:, 2]
    mb_prob = MichiBoost.predict(mb, jl_df_te; prediction_type="Probability")

    cb_acc = mean((cb_prob .>= 0.5) .== (y[te] .== 1.0))
    mb_acc = mean((mb_prob .>= 0.5) .== (y[te] .== 1.0))

    println("\nCategorical (cardinality=20)  n_tr=$n_tr, 5 cat + 5 num")
    row("LogLoss",  binary_logloss(mb_prob, y[te]), binary_logloss(cb_prob, y[te]))
    row("AUC",      auc(mb_prob, y[te]),            auc(cb_prob, y[te]))
    row("Accuracy", mb_acc,                         cb_acc)

    # Categorical targets here are random, so we only require ballpark parity.
    @test auc(mb_prob, y[te]) >= auc(cb_prob, y[te]) - 0.1
end

end

println()
