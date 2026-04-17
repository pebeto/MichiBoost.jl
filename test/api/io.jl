using MichiBoost: MichiBoost, Pool, MichiBoostRegressor, save_model
using Test

@testset "Model save/load round-trip" begin
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
    y = [1.0, 2.0, 3.0, 4.0]

    model = MichiBoostRegressor(; iterations=5, learning_rate=0.5, depth=2, verbose=false)
    MichiBoost.fit!(model, X, y)

    preds_before = MichiBoost.predict(model, X)

    tmpfile = tempname()
    save_model(model, tmpfile)
    loaded = MichiBoost.load_model(tmpfile)

    preds_after = MichiBoost.predict(loaded, Pool(X))
    @test preds_before ≈ preds_after
    rm(tmpfile; force=true)
end
