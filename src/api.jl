"""
    MichiBoostRegressor(; kwargs...) -> MichiBoostRegressor

Create a gradient-boosted regression model.

# Keyword Arguments
- `iterations::Int=1000` — number of boosting rounds (trees to build).
- `learning_rate::Float64=0.03` — step-size shrinkage applied to each tree.
- `depth::Int=6` — depth of each symmetric tree.
- `l2_leaf_reg::Float64=3.0` — L2 regularisation on leaf values.
- `loss_function::String="RMSE"` — `"RMSE"` or `"MAE"`.
- `border_count::Int=254` — max quantization borders per numerical feature.
- `min_data_in_leaf::Int=1` — minimum samples required in a leaf.
- `random_seed::Union{Int,Nothing}=nothing` — seed for reproducibility.
- `verbose::Bool=false` — print training progress.
- `boosting_type::String="Ordered"` — `"Ordered"` (default) or `"Plain"`.
- `early_stopping_rounds::Union{Int,Nothing}=nothing` — stop after this many
  rounds without improvement on `eval_set`.

# Example
```julia
model = MichiBoostRegressor(; iterations=200, depth=4)
fit!(model, X, y)
ŷ = predict(model, X_test)
```
"""
function MichiBoostRegressor(; kwargs...)
    params = Dict{Symbol,Any}(kwargs...)
    haskey(params, :loss_function) || (params[:loss_function] = "RMSE")
    return MichiBoostRegressor(params, nothing)
end

"""
    MichiBoostClassifier(; kwargs...) -> MichiBoostClassifier

Create a gradient-boosted classification model.  Supports binary (Logloss) and
multi-class (Softmax) targets.  Multi-class is auto-detected when the target has
more than two unique values.

Accepts the same keyword arguments as `MichiBoostRegressor`, plus:

- `loss_function::String="Logloss"` — `"Logloss"` for binary, `"MultiClass"`
  for multi-class (auto-detected if omitted).

# Example
```julia
model = MichiBoostClassifier(; iterations=200, depth=4)
fit!(model, X, y)
probs  = predict_proba(model, X_test)   # probabilities
labels = predict(model, X_test)          # class labels
```
"""
function MichiBoostClassifier(; kwargs...)
    params = Dict{Symbol,Any}(kwargs...)
    haskey(params, :loss_function) || (params[:loss_function] = "Logloss")
    return MichiBoostClassifier(params, nothing)
end

"""
    fit!(model, data, labels; cat_features=nothing, kwargs...) -> model
    fit!(model, pool::Pool; eval_set=nothing, kwargs...) -> model

Train `model` in-place on the given data.

# Arguments
- `model` — a `MichiBoostRegressor` or `MichiBoostClassifier`.
- `data` — a table, matrix, or `Pool`.
- `labels` — target vector (ignored when `data` is a `Pool` that already has a
  label).
- `cat_features` — categorical column indices (0-based) or names.
- `eval_set` — optional validation `Pool` for early stopping.
- `kwargs...` — any hyperparameter accepted by the model constructor; overrides
  the values stored in `model.params` for this call only.

Returns the mutated `model` (with `.model` populated).

# Example
```julia
model = MichiBoostRegressor(; iterations=100)
fit!(model, X_train, y_train)
```
"""
function fit!(m::MichiBoostWrapper, data, labels; cat_features=nothing, kwargs...)
    pool = data isa Pool ? data : Pool(data; label=labels, cat_features)
    if data isa Pool && data.label === nothing
        pool = Pool(
            data.features_numerical,
            data.features_categorical,
            data.cat_mapping,
            Float64.(labels),
            data.label_mapping,
            data.label_classes,
            data.feature_names,
            data.numerical_feature_indices,
            data.categorical_feature_indices,
            data.n_samples,
            data.n_features,
            data.weight,
            data.group_id,
        )
    end
    return fit!(m, pool; kwargs...)
end

function fit!(m::MichiBoostWrapper, pool::Pool; eval_set=nothing, kwargs...)
    p = merge(m.params, Dict{Symbol,Any}(kwargs...))
    default_loss = m isa MichiBoostRegressor ? "RMSE" : "Logloss"

    m.model = train(
        pool;
        iterations=Int(get(p, :iterations, 1000)),
        learning_rate=Float64(get(p, :learning_rate, 0.03)),
        depth=Int(get(p, :depth, 6)),
        l2_leaf_reg=Float64(get(p, :l2_leaf_reg, 3.0)),
        loss_function=String(get(p, :loss_function, default_loss)),
        border_count=Int(get(p, :border_count, 254)),
        min_data_in_leaf=Int(get(p, :min_data_in_leaf, 1)),
        random_seed=let v = get(p, :random_seed, nothing)
            v === nothing ? nothing : Int(v)
        end,
        verbose=Bool(get(p, :verbose, false)),
        rsm=Float64(get(p, :rsm, 1.0)),
        early_stopping_rounds=let v = get(p, :early_stopping_rounds, nothing)
            v === nothing ? nothing : Int(v)
        end,
        eval_pool=eval_set isa Pool ? eval_set : nothing,
        boosting_type=String(get(p, :boosting_type, "Ordered")),
    )
    return m
end

"""
    predict(model, data; prediction_type="Class", cat_features=nothing)

Generate predictions from a trained model.

# Arguments
- `model` — a trained `MichiBoostRegressor` or
  `MichiBoostClassifier`.
- `data` — a table, matrix, or `Pool`.
- `prediction_type` — one of:
  - `"Class"` (default) — regression values, or predicted class labels for
    classifiers.
  - `"Probability"` — predicted probabilities (classification only).
  - `"RawFormulaVal"` — raw logits / scores before any transformation.
- `cat_features` — categorical column indices or names (only needed when `data`
  is not a `Pool`).

# Returns
- **Regressor**: `Vector{Float64}` of predicted values.
- **Classifier** with `"Class"`: `Vector` of predicted class labels.
- **Classifier** with `"Probability"`: `Vector{Float64}` (binary) or
  `Matrix{Float64}` (multi-class, rows = samples, cols = classes).

# Example
```julia
ŷ = predict(model, X_test)
```
"""
function predict(
    m::MichiBoostWrapper,
    data;
    prediction_type::String="Class",
    cat_features=nothing,
)
    m.model === nothing && error("Model has not been trained. Call fit! first.")
    pool = data isa Pool ? data : Pool(data; cat_features)

    prediction_type == "RawFormulaVal" && return _predict_raw(m.model, pool)
    prediction_type == "Probability" && return MichiBoost.predict(m.model, pool)
    return m isa MichiBoostClassifier ? predict_classes(m.model, pool) :
           MichiBoost.predict(m.model, pool)
end

"""
    predict_proba(model::MichiBoostClassifier, data; cat_features=nothing)

Return predicted probabilities from a trained classifier.

- **Binary**: `Vector{Float64}` — probability of the positive class.
- **Multi-class**: `Matrix{Float64}` — one column per class, rows sum to 1.

# Example
```julia
probs = predict_proba(model, X_test)
```
"""
function predict_proba(m::MichiBoostClassifier, data; cat_features=nothing)
    m.model === nothing && error("Model has not been trained. Call fit! first.")
    pool = data isa Pool ? data : Pool(data; cat_features)
    return MichiBoost.predict(m.model, pool)
end

"""
    predict_classes(model, data)

Return predicted class labels from a trained classifier.  Equivalent to
`predict(model, data; prediction_type="Class")` for classifiers.
"""
function predict_classes end  # Actual method is on MichiBoostModel in predict.jl

function _predict_raw(model::MichiBoostModel, pool::Pool)
    n = pool.n_samples
    num_bins = if n_numerical(pool) > 0
        apply_borders(pool.features_numerical, model.borders)
    else
        Matrix{UInt16}(undef, n, 0)
    end
    cat_enc = if n_categorical(pool) > 0 && model.encoder !== nothing
        encode_categorical(model.encoder, pool.features_categorical)
    else
        Matrix{Float64}(undef, n, 0)
    end

    if model.is_multiclass
        preds = repeat(model.initial_pred', n, 1)
        for tree in model.trees
            preds .+= model.learning_rate .* predict_tree_multiclass(
                tree,
                num_bins,
                cat_enc,
            )
        end
        return preds
    else
        preds = fill(model.initial_pred::Float64, n)
        for tree in model.trees
            preds .+= model.learning_rate .* predict_tree(tree, num_bins, cat_enc)
        end
        return preds
    end
end

"""
    feature_importance(model) -> Vector{Pair{Symbol, Float64}}

Return feature importances as `feature_name => percentage` pairs, sorted by
importance (descending).  Importance is based on how often each feature was
chosen for a split across all trees.

# Example
```julia
fi = feature_importance(model)
# [:num_1 => 62.5, :num_3 => 25.0, :cat_1 => 12.5]
```
"""
function feature_importance(m::MichiBoostWrapper)
    m.model === nothing && error("Model has not been trained. Call fit! first.")
    return feature_importance(m.model)
end

# save / load

"""
    save_model(model, filepath::AbstractString)

Serialize a trained model to disk using Julia's `Serialization` module.

Works with both wrapper types (`MichiBoostRegressor`,
`MichiBoostClassifier`) and raw `MichiBoostModel` objects.

See also `load_model`.
"""
function save_model(m::MichiBoostWrapper, filepath::AbstractString)
    m.model === nothing && error("Model has not been trained. Call fit! first.")
    save_model(m.model, filepath)
end

"""
    load_model(filepath::AbstractString) -> MichiBoostModel

Load a model previously saved with `save_model`.

# Example
```julia
save_model(model, "my_model.jls")
loaded = load_model("my_model.jls")
predict(loaded, X_test)  # works with the raw MichiBoostModel
```
"""
function load_model end  # Actual method in io.jl

"""
    cv(pool::Pool; params=Dict(), fold_count=3, shuffle=true,
       random_seed=0, verbose=false, kwargs...) -> NamedTuple

Perform k-fold cross-validation on the given `Pool`.

# Arguments
- `pool` — a `Pool` with labels.
- `params` — `Dict` of training hyperparameters (string or symbol keys).
- `fold_count` — number of folds.
- `shuffle` — whether to shuffle indices before splitting.
- `random_seed` — seed for the shuffle.
- `verbose` — print per-fold results.
- `kwargs...` — additional hyperparameters (merged with `params`).

# Returns
A `NamedTuple` with fields:
- `train_loss::Vector{Float64}` — training loss per fold.
- `test_loss::Vector{Float64}` — test loss per fold.
- `mean_train_loss::Float64`
- `mean_test_loss::Float64`

# Example
```julia
pool = Pool(X; label=y)
result = cv(pool; fold_count=5,
            params=Dict("iterations" => 100, "depth" => 4))
println("Mean test loss: ", result.mean_test_loss)
```
"""
function cv(
    pool::Pool;
    params=Dict(),
    fold_count::Int=3,
    shuffle::Bool=true,
    random_seed::Int=0,
    verbose::Bool=false,
    kwargs...,
)
    all_params = Dict{Symbol,Any}()
    for (k, v) in params
        all_params[Symbol(k)] = v
    end
    for (k, v) in kwargs
        all_params[k] = v
    end

    label = get_label(pool)
    n = pool.n_samples
    rng = MersenneTwister(random_seed)
    indices = shuffle ? randperm(rng, n) : collect(1:n)
    fold_size = n ÷ fold_count
    loss_fn = get(all_params, :loss_function, "RMSE")

    train_losses, test_losses = Float64[], Float64[]

    for fold in 1:fold_count
        ts = (fold - 1) * fold_size + 1
        te = fold == fold_count ? n : fold * fold_size
        test_idx = indices[ts:te]
        train_idx = vcat(indices[1:(ts - 1)], indices[(te + 1):end])

        train_pool = slice(pool, train_idx)
        test_pool = slice(pool, test_idx)

        model = train(
            train_pool;
            iterations=Int(get(all_params, :iterations, 1000)),
            learning_rate=Float64(get(all_params, :learning_rate, 0.03)),
            depth=Int(get(all_params, :depth, 6)),
            l2_leaf_reg=Float64(get(all_params, :l2_leaf_reg, 3.0)),
            loss_function=String(loss_fn),
            verbose=Bool(verbose),
            random_seed=random_seed,
        )

        lf = make_loss(String(loss_fn))
        
        if model.is_multiclass
            # Get raw logits for multiclass
            train_logits = _predict_raw_logits(model, train_pool)
            test_logits = _predict_raw_logits(model, test_pool)
            
            # Convert labels to one-hot encoding
            train_y = get_label(train_pool)
            test_y = get_label(test_pool)
            class_labels = sort(unique(vcat(train_y, test_y)))
            n_classes = length(class_labels)
            label_map = Dict(class_labels[i] => i for i in eachindex(class_labels))
            
            train_y_onehot = zeros(Float64, length(train_y), n_classes)
            for i in eachindex(train_y)
                train_y_onehot[i, label_map[train_y[i]]] = 1.0
            end
            
            test_y_onehot = zeros(Float64, length(test_y), n_classes)
            for i in eachindex(test_y)
                test_y_onehot[i, label_map[test_y[i]]] = 1.0
            end
            
            push!(train_losses, loss(lf, train_y_onehot, train_logits))
            push!(test_losses, loss(lf, test_y_onehot, test_logits))
        elseif model.n_classes == 2
            # Get raw logits for binary classification
            train_logits = _predict_raw_logits(model, train_pool)
            test_logits = _predict_raw_logits(model, test_pool)
            
            push!(train_losses, loss(lf, get_label(train_pool), train_logits))
            push!(test_losses, loss(lf, get_label(test_pool), test_logits))
        else
            # Regression
            train_pred = MichiBoost.predict(model, train_pool)
            test_pred = MichiBoost.predict(model, test_pool)
            push!(train_losses, loss(lf, get_label(train_pool), train_pred))
            push!(test_losses, loss(lf, get_label(test_pool), test_pred))
        end
        if verbose
            train_str = last(train_losses)
            test_str = last(test_losses)
            println("Fold $fold/$fold_count: train=$train_str, test=$test_str")
        end
    end

    return (
        train_loss=train_losses,
        test_loss=test_losses,
        mean_train_loss=mean(train_losses),
        mean_test_loss=mean(test_losses),
    )
end

function _predict_raw_logits(model::MichiBoostModel, pool::Pool)
    num_bins, cat_encoded = _prepare_features(model, pool)
    n = pool.n_samples

    if model.is_multiclass
        preds = repeat(model.initial_pred', n, 1)
        for tree in model.trees
            predict_tree_mc!(preds, tree, num_bins, cat_encoded, model.learning_rate)
        end
        return preds
    else
        preds = fill(model.initial_pred::Float64, n)
        for tree in model.trees
            predict_tree!(preds, tree, num_bins, cat_encoded, model.learning_rate)
        end
        return preds
    end
end
