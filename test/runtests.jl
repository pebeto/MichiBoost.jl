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
    include("training/class_weights.jl")
    include("training/eval_metric.jl")
    include("training/custom_loss.jl")
end

@testset "API" verbose = true begin
    include("api/prediction.jl")
    include("api/feature_importance.jl")
    include("api/io.jl")
    include("api/cv.jl")
    include("api/score.jl")
    include("api/shap.jl")
    include("api/tag_types.jl")
    include("api/best_iteration.jl")
    include("api/staged_predict.jl")
    include("api/eval_metrics.jl")
    include("api/shrink.jl")
    include("api/isfitted.jl")
    include("api/params.jl")
end

Aqua.test_all(MichiBoost)
