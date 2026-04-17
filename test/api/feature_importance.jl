using MichiBoost: MichiBoost, MichiBoostRegressor, feature_importance
using Test

@testset "Feature importance returns real names" begin
    X = [1.0 4.0 5.0 6.0; 4.0 5.0 6.0 7.0; 30.0 40.0 50.0 60.0]
    y = [10.0, 20.0, 30.0]

    model = MichiBoostRegressor(; iterations=5, learning_rate=0.5, depth=2, verbose=false)
    MichiBoost.fit!(model, X, y)

    fi = feature_importance(model)
    @test length(fi) > 0
    @test all(p -> p isa Pair{Symbol,Float64}, fi)
    @test Set(first.(fi)) ⊆ Set([:x1, :x2, :x3, :x4])
end
