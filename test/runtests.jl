using Aqua
using MichiBoost
using DataFrames
using Statistics
using Test

@testset "Data" verbose = true begin
    include("data/pool.jl")
    include("data/encoding.jl")
end

@testset "Training" verbose = true begin
    include("training/regression.jl")
    include("training/classification.jl")
    include("training/categorical.jl")
    include("training/training_options.jl")
end

@testset "API" verbose = true begin
    include("api/prediction.jl")
    include("api/feature_importance.jl")
    include("api/io.jl")
    include("api/cv.jl")
    include("api/shap.jl")
end

Aqua.test_all(MichiBoost)
