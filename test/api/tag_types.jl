using MichiBoost
using MichiBoost: _to_string
using MichiBoost.Losses: LossKind
using MichiBoost.BoostingTypes: BoostingType
using MichiBoost.AutoClassWeights: AutoClassWeightMode
using MichiBoost.PredictionTypes: PredictionType
using MichiBoost: _loss_name, _boosting_name, _auto_class_weight_name, _prediction_name
using Random
using Test

@testset "Losses tag: name resolution" begin
    @test _loss_name(Losses.RMSE) == "RMSE"
    @test _loss_name(Losses.MAE) == "MAE"
    @test _loss_name(Losses.Logloss) == "Logloss"
    @test _loss_name(Losses.CrossEntropy) == "CrossEntropy"
    @test _loss_name(Losses.MultiClass) == "MultiClass"
    @test _loss_name(Losses.MultiLogloss) == "MultiLogloss"
end

@testset "BoostingTypes tag: name resolution" begin
    @test _boosting_name(BoostingTypes.Ordered) == "Ordered"
    @test _boosting_name(BoostingTypes.Plain) == "Plain"
end

@testset "AutoClassWeights tag: name resolution" begin
    @test _auto_class_weight_name(AutoClassWeights.Balanced) == "Balanced"
    @test _auto_class_weight_name(AutoClassWeights.SqrtBalanced) == "SqrtBalanced"
end

@testset "PredictionTypes tag: name resolution" begin
    @test _prediction_name(PredictionTypes.Class) == "Class"
    @test _prediction_name(PredictionTypes.Probability) == "Probability"
    @test _prediction_name(PredictionTypes.RawFormulaVal) == "RawFormulaVal"
end

@testset "_to_string: accepts strings and tag types" begin
    @test _to_string("RMSE", LossKind, _loss_name, "loss_function") == "RMSE"
    @test _to_string(Losses.RMSE, LossKind, _loss_name, "loss_function") == "RMSE"
    @test _to_string(Losses.MAE, LossKind, _loss_name, "loss_function") == "MAE"
end

@testset "_to_string: accepts symbols" begin
    @test _to_string(:RMSE, LossKind, _loss_name, "loss_function") == "RMSE"
    @test _to_string(:Plain, BoostingType, _boosting_name, "boosting_type") == "Plain"
    @test _to_string(
        :Balanced, AutoClassWeightMode, _auto_class_weight_name, "auto_class_weights"
    ) == "Balanced"
    @test _to_string(:Probability, PredictionType, _prediction_name, "prediction_type") ==
        "Probability"
end

@testset "_to_string: rejects wrong types" begin
    @test_throws ErrorException _to_string(42, LossKind, _loss_name, "loss_function")
    # A Metric tag is the wrong abstract supertype for loss_function.
    @test_throws ErrorException _to_string(
        Metrics.AUC, LossKind, _loss_name, "loss_function"
    )
end

@testset "loss_function tag: end-to-end via wrapper" begin
    Random.seed!(0)
    X = randn(60, 3)
    y = X[:, 1] .+ randn(60) .* 0.1

    str_model = MichiBoostRegressor(;
        iterations=20, depth=2, loss_function="MAE", verbose=false
    )
    fit!(str_model, X, y)

    tag_model = MichiBoostRegressor(;
        iterations=20, depth=2, loss_function=Losses.MAE, verbose=false
    )
    fit!(tag_model, X, y)

    sym_model = MichiBoostRegressor(;
        iterations=20, depth=2, loss_function=:MAE, verbose=false
    )
    fit!(sym_model, X, y)

    # Same loss → same fit. Both run; comparing predictions for sanity.
    @test predict(str_model, X) ≈ predict(tag_model, X)
    @test predict(str_model, X) ≈ predict(sym_model, X)
end

@testset "boosting_type tag: end-to-end via wrapper" begin
    Random.seed!(0)
    X = randn(40, 2)
    y = randn(40)

    a = MichiBoostRegressor(;
        iterations=10, depth=2, verbose=false, boosting_type=BoostingTypes.Plain
    )
    fit!(a, X, y)

    b = MichiBoostRegressor(; iterations=10, depth=2, verbose=false, boosting_type="Plain")
    fit!(b, X, y)

    c = MichiBoostRegressor(; iterations=10, depth=2, verbose=false, boosting_type=:Plain)
    fit!(c, X, y)

    @test predict(a, X) ≈ predict(b, X)
    @test predict(a, X) ≈ predict(c, X)
end

@testset "auto_class_weights tag: end-to-end via wrapper" begin
    Random.seed!(0)
    X = randn(60, 2)
    y = vcat(zeros(50), ones(10))

    a = MichiBoostClassifier(;
        iterations=10, depth=2, verbose=false, auto_class_weights=AutoClassWeights.Balanced
    )
    fit!(a, X, y)

    b = MichiBoostClassifier(;
        iterations=10, depth=2, verbose=false, auto_class_weights="Balanced"
    )
    fit!(b, X, y)

    c = MichiBoostClassifier(;
        iterations=10, depth=2, verbose=false, auto_class_weights=:Balanced
    )
    fit!(c, X, y)

    @test predict_proba(a, X) ≈ predict_proba(b, X)
    @test predict_proba(a, X) ≈ predict_proba(c, X)
end

@testset "prediction_type tag: end-to-end via wrapper" begin
    Random.seed!(0)
    X = randn(40, 3)
    y = Float64.(X[:, 1] .> 0)
    clf = MichiBoostClassifier(; iterations=10, depth=2, verbose=false)
    fit!(clf, X, y)

    @test predict(clf, X; prediction_type=PredictionTypes.Probability) ==
        predict(clf, X; prediction_type="Probability")
    @test predict(clf, X; prediction_type=PredictionTypes.RawFormulaVal) ==
        predict(clf, X; prediction_type="RawFormulaVal")
    @test predict(clf, X; prediction_type=PredictionTypes.Class) ==
        predict(clf, X; prediction_type="Class")

    # Symbol form mirrors the string form.
    @test predict(clf, X; prediction_type=:Probability) ==
        predict(clf, X; prediction_type="Probability")
    @test predict(clf, X; prediction_type=:RawFormulaVal) ==
        predict(clf, X; prediction_type="RawFormulaVal")
    @test predict(clf, X; prediction_type=:Class) ==
        predict(clf, X; prediction_type="Class")
end
