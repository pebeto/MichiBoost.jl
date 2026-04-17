using MichiBoost: MichiBoost, MichiBoostClassifier
using Test

@testset "prediction_type RawFormulaVal" begin
    X = [0.0 3.0; 4.0 1.0; 8.0 1.0; 9.0 1.0]
    y = [0.0, 0.0, 1.0, 1.0]

    model = MichiBoostClassifier(; iterations=10, depth=2, verbose=false)
    MichiBoost.fit!(model, X, y)

    raw = MichiBoost.predict(model, X; prediction_type="RawFormulaVal")
    @test length(raw) == 4
    @test all(isfinite, raw)
end

@testset "prediction_type Probability" begin
    X = [0.0 3.0; 4.0 1.0; 8.0 1.0; 9.0 1.0]
    y = [0.0, 0.0, 1.0, 1.0]

    model = MichiBoostClassifier(; iterations=10, depth=2, verbose=false)
    MichiBoost.fit!(model, X, y)

    probs = MichiBoost.predict(model, X; prediction_type="Probability")
    @test all(0.0 .<= probs .<= 1.0)
end
