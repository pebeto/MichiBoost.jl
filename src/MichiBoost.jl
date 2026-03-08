module MichiBoost

using CategoricalArrays: CategoricalValue, unwrap
using Random: Random, randperm, MersenneTwister
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
include("trees/build.jl")

# Core engine
include("train.jl")
include("predict.jl")
include("io.jl")

# User-facing API
include("api.jl")

export MichiBoostRegressor, MichiBoostClassifier
export fit!, predict, predict_proba, predict_classes
export Pool, slice
export feature_importance
export save_model, load_model
export cv

end # module
