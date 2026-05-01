module MichiBoost

using CategoricalArrays: CategoricalValue, unwrap
using Random: AbstractRNG, MersenneTwister, Random, randperm, shuffle!
using Serialization: Serialization
using StatisticalMeasures: StatisticalMeasures
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

# Eval-metric tag types and dispatcher
include("metrics.jl")

# String-replacement tag types for boosting_type / auto_class_weights /
# prediction_type
include("tags.jl")

# Symmetric tree inference and construction
include("trees/base/histograms.jl")
include("trees/base/predict.jl")
include("trees/base/shap.jl")
include("trees/base/build.jl")
include("trees/multiclass/histograms.jl")
include("trees/multiclass/predict.jl")
include("trees/multiclass/shap.jl")
include("trees/multiclass/build.jl")

# Core engine
include("train.jl")
include("predict.jl")
include("io.jl")

# SHAP dispatcher (depends on _prepare_features from predict.jl)
include("trees/shap.jl")

# User-facing API
include("api.jl")

export MichiBoostClassifier, MichiBoostRegressor
export AutoClassWeights, BoostingTypes, Losses, Metrics, PredictionTypes
export Pool
export cv
export feature_importance
export fit!
export get_best_iteration, get_best_score
export load_model, save_model
export predict, predict_classes, predict_proba
export score
export shap_values
export slice

end # module
