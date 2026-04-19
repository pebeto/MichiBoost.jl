module MichiBoost

using CategoricalArrays: CategoricalValue, unwrap
using Random: AbstractRNG, MersenneTwister, Random, randperm
using Serialization: Serialization
using Statistics: mean, median
using Tables: Tables

# Struct definitions
include("types.jl")

# Data handling
include("data/pool.jl")
include("data/quantization.jl")
include("data/encoding.jl")

# Loss functions
include("losses.jl")

# Symmetric tree inference and construction
include("trees/predict.jl")
include("trees/shap.jl")
include("trees/histograms.jl")
include("trees/build.jl")

# Core engine
include("train.jl")
include("predict.jl")
include("io.jl")

# User-facing API
include("api.jl")

export MichiBoostClassifier, MichiBoostRegressor
export Pool
export cv
export feature_importance
export fit!
export load_model, save_model
export predict, predict_classes, predict_proba
export shap_values
export slice

end # module
