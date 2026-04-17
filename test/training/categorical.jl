using MichiBoost: MichiBoost, MichiBoostClassifier
using DataFrames
using Test

@testset "Categorical features — ordered encoding" begin
    df = DataFrame(cat1=["a", "b", "a", "b", "c", "c"],
                   num1=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                   num2=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    labels = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

    model = MichiBoostClassifier(; iterations=5, learning_rate=0.5, depth=2, verbose=false)
    MichiBoost.fit!(model, df, labels)

    preds = MichiBoost.predict(model, df)
    @test length(preds) == 6
end

@testset "Categorical features — plain encoding" begin
    df = DataFrame(cat1=["a", "b", "a", "b", "c", "c"],
                   num1=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    labels = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

    model = MichiBoostClassifier(; iterations=5, depth=2, boosting_type="Plain", verbose=false)
    MichiBoost.fit!(model, df, labels)

    preds = MichiBoost.predict(model, df)
    @test length(preds) == 6
end
