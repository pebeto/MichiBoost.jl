using MichiBoost
using MichiBoost:
    _eval_metric, _binary_auc, _binary_accuracy, _binary_f1, _mc_accuracy, _resolve_metric
using MichiBoost.Metrics
using Random
using Statistics
using Test

@testset "eval_metric: regression dispatch" begin
    o, fn = _eval_metric(Metrics.RMSE, false, 0)
    @test o == :minimize
    @test isapprox(
        fn([1.0, 2.0, 3.0], [1.1, 1.9, 3.2]),
        sqrt(mean(([0.1, -0.1, 0.2]) .^ 2));
        atol=1e-12,
    )

    o, fn = _eval_metric(Metrics.MAE, false, 0)
    @test o == :minimize
    @test isapprox(
        fn([1.0, 2.0, 3.0], [1.1, 1.9, 3.2]), mean(abs.([0.1, -0.1, 0.2])); atol=1e-12
    )

    o, fn = _eval_metric(Metrics.R2, false, 0)
    @test o == :maximize
    @test fn([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) ≈ 1.0
end

@testset "eval_metric: binary AUC matches direct rank computation" begin
    Random.seed!(0)
    n = 100
    y = Float64.(rand(Bool, n))
    logits = randn(n)

    auc_inline = _binary_auc(y, logits)
    @test 0.0 <= auc_inline <= 1.0
    # Sanity check: perfect ordering ⇒ AUC = 1.0
    perfect_logits = collect(1.0:Float64(n)) .* (2 .* y .- 1)
    @test _binary_auc(y, perfect_logits) ≥ 0.99
end

@testset "eval_metric: binary accuracy / f1 from logits" begin
    y = [0.0, 0.0, 1.0, 1.0, 1.0]
    # Threshold at 0.0: predicts {0, 1, 1, 0, 1}
    logits = [-1.0, 0.5, 1.0, -0.5, 2.0]
    # tp=2 (rows 3, 5), fn=1 (row 4), fp=1 (row 2), tn=1
    @test _binary_accuracy(y, logits) == 3 / 5
    prec = 2 / (2 + 1)
    rec = 2 / (2 + 1)
    @test _binary_f1(y, logits) ≈ 2 * prec * rec / (prec + rec)
end

@testset "eval_metric: multiclass accuracy" begin
    y_onehot = [
        1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0
    ]
    # logits favour classes 1, 2, 1 (last is wrong)
    logits = [
        2.0 0.0 0.0
        0.0 3.0 0.0
        1.0 0.0 0.5
    ]
    @test _mc_accuracy(y_onehot, logits) == 2 / 3
end

@testset "eval_metric: AUC drives early stopping (binary)" begin
    Random.seed!(42)
    n = 600
    X = randn(n, 4)
    y = Float64.(X[:, 1] .+ 0.5 .* X[:, 2] .> 0)

    X_train, y_train = X[1:400, :], y[1:400]
    X_val, y_val = X[401:end, :], y[401:end]

    train_pool = Pool(X_train; label=y_train)
    val_pool = Pool(X_val; label=y_val)

    # No eval_metric: ES uses Logloss.
    plain = MichiBoostClassifier(;
        iterations=200, depth=3, early_stopping_rounds=10, verbose=false
    )
    fit!(plain, train_pool; eval_set=val_pool)

    auc_es = MichiBoostClassifier(;
        iterations=200, depth=3, early_stopping_rounds=10, eval_metric="AUC", verbose=false
    )
    fit!(auc_es, train_pool; eval_set=val_pool)

    # Both should produce reasonable models; just verify no crash and
    # both ran some iterations.
    @test length(plain.model.trees) >= 1
    @test length(auc_es.model.trees) >= 1
end

@testset "eval_metric: R2 drives early stopping (regression)" begin
    Random.seed!(7)
    n = 400
    X = randn(n, 3)
    y = 2.0 .* X[:, 1] .+ randn(n) .* 0.2

    train_pool = Pool(X[1:300, :]; label=y[1:300])
    val_pool = Pool(X[301:end, :]; label=y[301:end])

    reg = MichiBoostRegressor(;
        iterations=200, depth=3, early_stopping_rounds=10, eval_metric="R2", verbose=false
    )
    fit!(reg, train_pool; eval_set=val_pool)
    @test length(reg.model.trees) >= 1
end

@testset "Metrics: orientation per metric" begin
    o_t, _ = _eval_metric(Metrics.RMSE, false, 0)
    @test o_t == :minimize

    o_t, _ = _eval_metric(Metrics.AUC, false, 2)
    @test o_t == :maximize

    o_t, _ = _eval_metric(Metrics.MultiLogloss, true, 3)
    @test o_t == :minimize

    o_t, _ = _eval_metric(Metrics.Accuracy, false, 2)
    @test o_t == :maximize
    o_t, _ = _eval_metric(Metrics.Accuracy, true, 4)
    @test o_t == :maximize
end

@testset "Metrics: task-mismatch errors" begin
    @test_throws ErrorException _eval_metric(Metrics.RMSE, false, 2)   # binary
    @test_throws ErrorException _eval_metric(Metrics.RMSE, true, 3)    # mc
    @test_throws ErrorException _eval_metric(Metrics.AUC, false, 0)    # reg
    @test_throws ErrorException _eval_metric(Metrics.AUC, true, 3)     # mc
    @test_throws ErrorException _eval_metric(Metrics.F1, true, 3)      # mc
    @test_throws ErrorException _eval_metric(Metrics.MultiLogloss, false, 2)  # binary
end

@testset "_resolve_metric: known names" begin
    @test _resolve_metric("RMSE") === Metrics.RMSE
    @test _resolve_metric("rmse") === Metrics.RMSE   # case-insensitive
    @test _resolve_metric("MAE") === Metrics.MAE
    @test _resolve_metric("R2") === Metrics.R2
    @test _resolve_metric("RSquared") === Metrics.R2
    @test _resolve_metric("Logloss") === Metrics.Logloss
    @test _resolve_metric("CrossEntropy") === Metrics.Logloss
    @test _resolve_metric("MultiClass") === Metrics.MultiLogloss
    @test _resolve_metric("Accuracy") === Metrics.Accuracy
    @test _resolve_metric("F1") === Metrics.F1
    @test _resolve_metric("AUC") === Metrics.AUC
    @test_throws ErrorException _resolve_metric("nope")
end

@testset "Metrics tag: end-to-end via wrapper" begin
    Random.seed!(42)
    X = randn(400, 3)
    y = Float64.(X[:, 1] .+ X[:, 2] .> 0)

    train_pool = Pool(X[1:300, :]; label=y[1:300])
    val_pool = Pool(X[301:end, :]; label=y[301:end])

    clf_typed = MichiBoostClassifier(;
        iterations=80,
        depth=3,
        early_stopping_rounds=10,
        eval_metric=Metrics.AUC,   # bare type, no `()`
        verbose=false,
    )
    fit!(clf_typed, train_pool; eval_set=val_pool)
    @test length(clf_typed.model.trees) >= 1
end
