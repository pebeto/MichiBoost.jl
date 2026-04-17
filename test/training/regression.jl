using MichiBoost: MichiBoost, Pool, MichiBoostRegressor
using Test

@testset "Regression training and prediction" begin
    X = [1.0 4.0 5.0 6.0; 4.0 5.0 6.0 7.0; 30.0 40.0 50.0 60.0]
    y = [10.0, 20.0, 30.0]

    model = MichiBoostRegressor(; iterations=10, learning_rate=0.5, depth=2, verbose=false)
    MichiBoost.fit!(model, X, y)

    preds = MichiBoost.predict(model, X)
    @test length(preds) == 3
    @test all(isfinite, preds)
end

@testset "MAE loss function" begin
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
    y = [1.0, 2.0, 3.0, 4.0]

    model = MichiBoostRegressor(; iterations=10, depth=2, loss_function="MAE", verbose=false)
    MichiBoost.fit!(model, X, y)

    preds = MichiBoost.predict(model, X)
    @test length(preds) == 4
    @test all(isfinite, preds)
end
