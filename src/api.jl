function MichiBoostRegressor(; kwargs...)
    params = Dict{Symbol,Any}(kwargs...)
    haskey(params, :loss_function) || (params[:loss_function] = "RMSE")
    return MichiBoostRegressor(params, nothing)
end

function MichiBoostClassifier(; kwargs...)
    params = Dict{Symbol,Any}(kwargs...)
    haskey(params, :loss_function) || (params[:loss_function] = "Logloss")
    return MichiBoostClassifier(params, nothing)
end

"""
    fit!(model, data, labels; cat_features=nothing, kwargs...) -> model
    fit!(model, pool::Pool; eval_set=nothing, kwargs...) -> model

Train `model` in-place on the given data. Returns the mutated `model` with its
`.model` field populated.

# Arguments
- `model`: a [`MichiBoostRegressor`](@ref) or [`MichiBoostClassifier`](@ref).
- `data`: a table, matrix, or [`Pool`](@ref).
- `labels`: target vector. Ignored when `data` is a [`Pool`](@ref) that
  already has a label.
- `cat_features`: 1-based categorical column indices or names.
- `eval_set`: optional validation [`Pool`](@ref) for early stopping.
- `kwargs...`: any hyperparameter accepted by the model constructor. Overrides
  the values stored in `model.params` for this call only.

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
            Float64.(labels),
            data.label_classes,
            data.feature_names,
            data.numerical_feature_indices,
            data.categorical_feature_indices,
            data.n_samples,
            data.n_features,
            data.weight,
        )
    end
    return fit!(m, pool; kwargs...)
end

function fit!(m::MichiBoostWrapper, pool::Pool; eval_set=nothing, kwargs...)
    p = merge(m.params, Dict{Symbol,Any}(kwargs...))
    default_loss = m isa MichiBoostRegressor ? "RMSE" : "Logloss"

    raw_loss = get(p, :loss_function, default_loss)
    resolved_loss = if raw_loss isa LossFunction
        tt = task_type(raw_loss)
        if tt === :regression
            m isa MichiBoostRegressor || error(
                "Custom regression `LossFunction` requires `MichiBoostRegressor`."
            )
        elseif tt === :binary || tt === :multiclass
            m isa MichiBoostClassifier ||
                error("Custom $tt `LossFunction` requires `MichiBoostClassifier`.")
        else
            error(
                "task_type(::$(typeof(raw_loss))) returned `$tt`; expected " *
                "`:regression`, `:binary`, or `:multiclass`.",
            )
        end
        raw_loss
    else
        _to_string(raw_loss, LossKind, _loss_name, "loss_function")
    end

    cw = get(p, :class_weights, nothing)
    acw = get(p, :auto_class_weights, nothing)
    if cw !== nothing || acw !== nothing
        m isa MichiBoostClassifier || error(
            "`class_weights` / `auto_class_weights` are only valid for MichiBoostClassifier",
        )
        cw === nothing ||
            acw === nothing ||
            error("`class_weights` and `auto_class_weights` are mutually exclusive")
        if acw !== nothing
            cw = _auto_class_weights(
                pool,
                _to_string(
                    acw, AutoClassWeightMode, _auto_class_weight_name, "auto_class_weights"
                ),
            )
        end
        pool = _apply_class_weights(pool, cw)
    end

    m.model = train(
        pool;
        iterations=Int(get(p, :iterations, 1000)),
        learning_rate=Float64(get(p, :learning_rate, 0.03)),
        depth=Int(get(p, :depth, 6)),
        l2_leaf_reg=Float64(get(p, :l2_leaf_reg, 3.0)),
        loss_function=resolved_loss,
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
        eval_metric=let v = get(p, :eval_metric, nothing)
            if v === nothing
                nothing
            elseif v isa Type && v <: Metric
                v
            elseif v isa AbstractString || v isa Symbol
                _resolve_metric(v)
            else
                error(
                    "`eval_metric` must be a `Metrics.*` type, a String, a Symbol, or `nothing`",
                )
            end
        end,
        boosting_type=_to_string(
            get(p, :boosting_type, "Ordered"), BoostingType, _boosting_name, "boosting_type"
        ),
    )
    return m
end

# Resolve a tag-or-string kwarg to its canonical String representation.
# `tag_super` is the abstract tag base type (e.g., LossKind);
# `name_fn` maps each concrete subtype to its canonical name.
function _to_string(v, tag_super::Type, name_fn::Function, kwarg::AbstractString)
    v isa AbstractString && return String(v)
    v isa Symbol && return String(v)
    v isa Type && v <: tag_super && return name_fn(v)
    return error("`$kwarg` must be a tag type, a String, or a Symbol, got `$(typeof(v))`")
end

# Return a new Pool whose `weight` has been multiplied by the per-class weight
# from `class_weights[label]`. Numeric keys are matched against `pool.label`
# (Float64) by value, non-numeric keys are matched against `pool.label_classes`.
function _apply_class_weights(pool::Pool, class_weights::AbstractDict)
    pool.label === nothing && error("`class_weights` requires a labelled Pool")
    n = pool.n_samples
    label_classes = pool.label_classes
    label_floats = pool.label

    # Resolve class_weights into a Float64-keyed map on the float-encoded label
    # space. Pool stores labels as Float64; if labels were non-numeric, the
    # originals live in `label_classes` with mapping
    # `label_classes[i] -> Float64(i - 1)`.
    cw_float = Dict{Float64,Float64}()
    if label_classes !== nothing
        for (i, original) in enumerate(label_classes)
            cw_float[Float64(i - 1)] = _lookup_class_weight(class_weights, original)
        end
    else
        for fl in unique(label_floats)
            cw_float[fl] = _lookup_class_weight(class_weights, fl)
        end
    end

    new_weights = pool.weight !== nothing ? copy(pool.weight) : ones(Float64, n)
    @inbounds for i in 1:n
        new_weights[i] *= cw_float[label_floats[i]]
    end

    return Pool(
        pool.features_numerical,
        pool.features_categorical,
        pool.label,
        pool.label_classes,
        pool.feature_names,
        pool.numerical_feature_indices,
        pool.categorical_feature_indices,
        pool.n_samples,
        pool.n_features,
        new_weights,
    )
end

# Compute a class_weights dict from label frequency. Matches CatBoost's
# `auto_class_weights="Balanced"` (`n / (n_classes * count)`) and
# `"SqrtBalanced"` (`sqrt(n / count)`).
function _auto_class_weights(pool::Pool, mode::String)
    pool.label === nothing && error("`auto_class_weights` requires a labelled Pool")
    label = pool.label
    counts = Dict{Float64,Int}()
    @inbounds for v in label
        counts[v] = get(counts, v, 0) + 1
    end
    n = length(label)
    n_classes = length(counts)

    weight = if mode == "Balanced"
        c -> n / (n_classes * counts[c])
    elseif mode == "SqrtBalanced"
        c -> sqrt(n / counts[c])
    else
        error("`auto_class_weights` must be \"Balanced\" or \"SqrtBalanced\", got `$mode`")
    end

    cw = Dict{Any,Float64}()
    if pool.label_classes !== nothing
        for (i, original) in enumerate(pool.label_classes)
            cw[original] = weight(Float64(i - 1))
        end
    else
        for fl in keys(counts)
            cw[fl] = weight(fl)
        end
    end
    return cw
end

function _lookup_class_weight(class_weights::AbstractDict, key)
    haskey(class_weights, key) && return Float64(class_weights[key])
    # Fall back to value-equality so that Int(0) keys match Float64(0.0) labels
    # and vice versa; common when users write `Dict(0 => 1.0, 1 => 5.0)`.
    for (k, v) in class_weights
        k == key && return Float64(v)
    end
    return error("`class_weights` is missing an entry for class `$key`")
end

"""
    predict(model, data; prediction_type=PredictionTypes.Class, cat_features=nothing)

Generate predictions from a trained model.

# Arguments
- `model`: a trained [`MichiBoostRegressor`](@ref) or
  [`MichiBoostClassifier`](@ref).
- `data`: a table, matrix, or [`Pool`](@ref).
- `prediction_type`: selects the output format. Accepts a `PredictionTypes`
  tag, its string name, or the matching Symbol:
  - `PredictionTypes.Class` (default): regression values, or predicted class
    labels for classifiers.
  - `PredictionTypes.Probability`: predicted probabilities (classification
    only).
  - `PredictionTypes.RawFormulaVal`: raw logits or scores before any
    transformation.
- `cat_features`: categorical column indices or names. Only needed when `data`
  is not a [`Pool`](@ref).

# Returns
- **Regressor**: `Vector{Float64}` of predicted values.
- **Classifier** with `PredictionTypes.Class`: `Vector` of predicted class
  labels.
- **Classifier** with `PredictionTypes.Probability`: `Vector{Float64}` for
  binary, `Matrix{Float64}` for multi-class (rows index samples, columns
  index classes).
- **Classifier** with `PredictionTypes.RawFormulaVal`: `Vector{Float64}` for
  binary logits, `Matrix{Float64}` for multi-class logits.

# Example
```julia
ŷ = predict(model, X_test)
```
"""
function predict(
    m::MichiBoostWrapper, data; prediction_type=PredictionTypes.Class, cat_features=nothing
)
    m.model === nothing && error("Model has not been trained. Call fit! first.")
    pool = data isa Pool ? data : Pool(data; cat_features)

    pt = _to_string(prediction_type, PredictionType, _prediction_name, "prediction_type")
    pt == "RawFormulaVal" && return _predict_raw(m.model, pool)
    pt == "Probability" && return MichiBoost.predict(m.model, pool)
    return if m isa MichiBoostClassifier
        predict_classes(m.model, pool)
    else
        MichiBoost.predict(m.model, pool)
    end
end

"""
    predict_proba(model::MichiBoostClassifier, data; cat_features=nothing)

Return predicted probabilities from a trained classifier.

- **Binary**: `Vector{Float64}`, the probability of the positive class.
- **Multi-class**: `Matrix{Float64}` with one column per class; rows sum to 1.

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

Return predicted class labels from a trained classifier. Equivalent to
`predict(model, data; prediction_type=PredictionTypes.Class)` for classifiers.

For binary classification, the threshold is 0.5 on the positive-class
probability. For multi-class, returns the class with the highest predicted
probability.
"""
function predict_classes end  # Actual method is on MichiBoostModel in predict.jl

# Build a per-iteration raw-score array by running each tree in sequence and
# snapshotting the running prediction after every addition.
#
# Returns:
# - regression / binary : Matrix{Float64} of shape (n_samples, n_iterations).
# - multi-class         : Array{Float64,3} of shape (n_samples, n_classes, n_iter).
function _staged_predict_raw(model::MichiBoostModel, pool::Pool)
    n = pool.n_samples
    n_iter = length(model.trees)
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
    leaf_buf = Vector{Int}(undef, n)

    if model.is_multiclass
        n_classes = model.n_classes
        out = Array{Float64,3}(undef, n, n_classes, n_iter)
        running = repeat((model.initial_pred::Vector{Float64})', n, 1)
        @inbounds for (i, tree) in enumerate(model.trees)
            predict_tree!(running, tree, num_bins, cat_enc, model.learning_rate, leaf_buf)
            @views out[:, :, i] .= running
        end
        return out
    else
        out = Matrix{Float64}(undef, n, n_iter)
        running = fill(model.initial_pred::Float64, n)
        @inbounds for (i, tree) in enumerate(model.trees)
            predict_tree!(running, tree, num_bins, cat_enc, model.learning_rate, leaf_buf)
            @views out[:, i] .= running
        end
        return out
    end
end

"""
    staged_predict(model, data; prediction_type=PredictionTypes.Class, cat_features=nothing)

Return predictions at every boosting iteration as an array. Use it to plot
convergence curves or to pick a truncation point post-hoc.

# Returns

| Task                         | `prediction_type`            | Shape                                      |
| ---------------------------- | ---------------------------- | ------------------------------------------ |
| Regression                   | any                          | `Matrix{Float64}` `(n_samples, n_iter)`    |
| Binary classification        | `Class`                      | `Matrix` of class labels                   |
| Binary classification        | `Probability`, `RawFormulaVal` | `Matrix{Float64}`                        |
| Multi-class classification   | `Class`                      | `Matrix` of class labels                   |
| Multi-class classification   | `Probability`, `RawFormulaVal` | `Array{Float64,3}` `(n_samples, n_classes, n_iter)` |

The result is densely materialised; memory cost is `n_samples × n_iterations`,
multiplied by `n_classes` in the multi-class case.
"""
function staged_predict(
    m::MichiBoostWrapper, data; prediction_type=PredictionTypes.Class, cat_features=nothing
)
    m.model === nothing && error("Model has not been trained. Call fit! first.")
    pool = data isa Pool ? data : Pool(data; cat_features)
    pt = _to_string(prediction_type, PredictionType, _prediction_name, "prediction_type")
    raw = _staged_predict_raw(m.model, pool)

    pt == "RawFormulaVal" && return raw
    m isa MichiBoostClassifier || return raw   # regression: raw == values

    model = m.model
    if model.is_multiclass
        n, k, niter = size(raw)
        if pt == "Probability"
            out = similar(raw)
            @inbounds for it in 1:niter
                _softmax_matrix!(view(out,:,:,it), view(raw,:,:,it))
            end
            return out
        end
        # Class labels: argmax across the class axis at each iteration.
        out = Matrix{eltype(model.class_labels)}(undef, n, niter)
        @inbounds for it in 1:niter, i in 1:n
            best_c = argmax(view(raw, i, :, it))
            out[i, it] = model.class_labels[best_c]
        end
        return out
    else
        # Binary classifier.
        n, niter = size(raw)
        if pt == "Probability"
            return _sigmoid.(raw)
        end
        out = Matrix{eltype(model.class_labels)}(undef, n, niter)
        @inbounds for it in 1:niter, i in 1:n
            out[i, it] = raw[i, it] >= 0.0 ? model.class_labels[2] : model.class_labels[1]
        end
        return out
    end
end

"""
    staged_predict_proba(model::MichiBoostClassifier, data; cat_features=nothing)

Per-iteration probabilities from a trained classifier. Equivalent to
`staged_predict(model, data; prediction_type=PredictionTypes.Probability)`.

# Returns
- **Binary**: `Matrix{Float64}` of shape `(n_samples, n_iterations)`, giving
  the positive-class probability at each iteration.
- **Multi-class**: `Array{Float64,3}` of shape
  `(n_samples, n_classes, n_iterations)`.
"""
function staged_predict_proba(m::MichiBoostClassifier, data; cat_features=nothing)
    return staged_predict(
        m, data; prediction_type=PredictionTypes.Probability, cat_features
    )
end

"""
    eval_metrics(model, pool::Pool; metrics) -> Dict{String,Vector{Float64}}

Evaluate one or more metrics on a labelled `Pool` at every boosting iteration.
Mirrors CatBoost's `eval_metrics`. Returns a dictionary keyed by canonical
metric name; each value is a vector of length `n_iterations` giving the
metric's value after `i` trees have been added.

`metrics` is a vector of `Metrics.*` tag types, CatBoost-style strings, or
Symbols, mixed freely (e.g. `[Metrics.AUC, "Logloss", :Accuracy]`). Each
metric must be valid for the model's task (regression, binary, or
multi-class), or `eval_metrics` raises an error.

# Example
```julia
result = eval_metrics(model, val_pool; metrics=[Metrics.AUC, Metrics.Logloss])
result["AUC"]      # Vector{Float64}, length == length(model.trees)
result["Logloss"]
```
"""
function eval_metrics(m::MichiBoostWrapper, pool::Pool; metrics::AbstractVector)
    m.model === nothing && error("Model has not been trained. Call fit! first.")
    pool.label === nothing && error("`eval_metrics` requires a labelled Pool")

    model = m.model
    n_iter = length(model.trees)
    raw = _staged_predict_raw(model, pool)

    y_raw = get_label(pool)
    y_eval = if model.is_multiclass
        # Build one-hot for multi-class metrics. The label-index mapping mirrors
        # the one `train` uses (sort + enumerate), so test pools with the same
        # class set produce consistent indices. Rows whose class is unseen at
        # training time are left all-zero; they contribute zero target mass.
        classes = sort(unique(y_raw))
        label_map = Dict(classes[i] => i for i in eachindex(classes))
        oh = zeros(Float64, length(y_raw), model.n_classes)
        @inbounds for i in eachindex(y_raw)
            c = get(label_map, y_raw[i], 0)
            c > 0 && (oh[i, c] = 1.0)
        end
        oh
    else
        y_raw
    end

    out = Dict{String,Vector{Float64}}()
    for metric in metrics
        metric_type = if metric isa AbstractString || metric isa Symbol
            _resolve_metric(metric)
        elseif metric isa Type && metric <: Metric
            metric
        else
            error(
                "each entry in `metrics` must be a `Metrics.*` tag, a String, or a Symbol, " *
                "got `$(typeof(metric))`",
            )
        end
        _, fn = _eval_metric(metric_type, model.is_multiclass, model.n_classes)

        values = Vector{Float64}(undef, n_iter)
        @inbounds for it in 1:n_iter
            iter_raw = if model.is_multiclass
                @view raw[:, :, it]
            else
                @view raw[:, it]
            end
            values[it] = fn(y_eval, iter_raw)
        end
        out[string(nameof(metric_type))] = values
    end

    return out
end

"""
    shrink!(model, n::Integer) -> model

Truncate the model's ensemble to its first `n` trees in place. Predictions
made after `shrink!` use only the retained trees. Use it to roll back to an
early-stopped snapshot (paired with [`get_best_iteration`](@ref)) or to
implement resume workflows.

`n` must satisfy `0 <= n <= length(model.trees)`. Returns `model` for
chaining. [`get_best_iteration`](@ref) and [`get_best_score`](@ref) keep the
values recorded during training, so `best_iteration` may exceed
`length(model.trees)` after a shrink.

# Example
```julia
clf = MichiBoostClassifier(; iterations=200, early_stopping_rounds=10)
fit!(clf, train_pool; eval_set=val_pool)
shrink!(clf, get_best_iteration(clf))   # roll back to the ES best
```
"""
function shrink!(m::MichiBoostWrapper, n::Integer)
    m.model === nothing && error("Model has not been trained. Call fit! first.")
    shrink!(m.model, n)
    return m
end

function shrink!(model::MichiBoostModel, n::Integer)
    n_current = length(model.trees)
    (n < 0 || n > n_current) && error("`n` must be in [0, $n_current], got $n")
    resize!(model.trees, n)
    return model
end

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
    trees = model.trees
    lr = model.learning_rate
    nt = Threads.nthreads()

    # Row-chunk parallelism: each thread owns a disjoint slice of rows and
    # runs *all* trees against it.  Writes into view(preds, rows) are
    # naturally non-overlapping, so no partials buffers and no final reduction
    # are needed, unlike tree-parallelism, which allocated nt × n partials
    # per call and then summed them serially.  For tiny batches we skip the
    # fork entirely; the overhead isn't worth it.
    if model.is_multiclass
        preds = repeat(model.initial_pred', n, 1)
        if n < 1024 || nt == 1
            leaf_buf = Vector{Int}(undef, n)
            for tree in trees
                predict_tree!(preds, tree, num_bins, cat_enc, lr, leaf_buf)
            end
        else
            Threads.@threads :static for k in 1:nt
                lo = div((k - 1) * n, nt) + 1
                hi = div(k * n, nt)
                _predict_chunk_mc!(preds, trees, num_bins, cat_enc, lr, lo, hi)
            end
        end
        return preds
    else
        preds = fill(model.initial_pred::Float64, n)
        if n < 1024 || nt == 1
            leaf_buf = Vector{Int}(undef, n)
            for tree in trees
                predict_tree!(preds, tree, num_bins, cat_enc, lr, leaf_buf)
            end
        else
            Threads.@threads :static for k in 1:nt
                lo = div((k - 1) * n, nt) + 1
                hi = div(k * n, nt)
                _predict_chunk!(preds, trees, num_bins, cat_enc, lr, lo, hi)
            end
        end
        return preds
    end
end

# Function-barrier helpers so the threaded body specialises on concrete
# types and keeps leaf_buf off the heap of the outer closure.
@inline function _predict_chunk!(preds, trees, num_bins, cat_enc, lr, lo, hi)
    chunk = hi - lo + 1
    leaf_buf = Vector{Int}(undef, chunk)
    pv = view(preds, lo:hi)
    nbv = view(num_bins, lo:hi, :)
    cev = view(cat_enc, lo:hi, :)
    for tree in trees
        predict_tree!(pv, tree, nbv, cev, lr, leaf_buf)
    end
    return nothing
end

@inline function _predict_chunk_mc!(preds, trees, num_bins, cat_enc, lr, lo, hi)
    chunk = hi - lo + 1
    leaf_buf = Vector{Int}(undef, chunk)
    pv = view(preds, lo:hi, :)
    nbv = view(num_bins, lo:hi, :)
    cev = view(cat_enc, lo:hi, :)
    for tree in trees
        predict_tree!(pv, tree, nbv, cev, lr, leaf_buf)
    end
    return nothing
end

"""
    feature_importance(model) -> Vector{Pair{Symbol, Float64}}

Return feature importances as `feature_name => percentage` pairs, sorted by
importance descending. Importance counts how often each feature is chosen for
a split across all trees, normalised to a percentage. Features that never
split get `0.0` and appear at the end of the returned vector.

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

"""
    isfitted(model) -> Bool

Return `true` if [`fit!`](@ref) has populated `model.model`, `false`
otherwise. Use it to guard prediction or inspection calls against untrained
wrappers.

# Example
```julia
model = MichiBoostRegressor(; iterations=100)
isfitted(model)              # false
fit!(model, X, y)
isfitted(model)              # true
```
"""
isfitted(m::MichiBoostWrapper) = m.model !== nothing

"""
    get_params(model) -> Dict{Symbol,Any}

Return a copy of the hyperparameters stored on the wrapper. Mutating the
returned `Dict` does not affect `model`. Use [`set_params!`](@ref) to update
them in place.

# Example
```julia
model = MichiBoostRegressor(; iterations=100, depth=4)
get_params(model)   # Dict(:iterations => 100, :depth => 4, :loss_function => "RMSE")
```
"""
get_params(m::MichiBoostWrapper) = copy(m.params)

"""
    set_params!(model; kwargs...) -> model

Update hyperparameters on the wrapper, returning `model` for chaining. New
keys are added; existing keys are overwritten. Changes take effect on the
next [`fit!`](@ref) call; already-trained models are not retrained.

# Example
```julia
model = MichiBoostRegressor(; iterations=100)
set_params!(model; iterations=500, depth=4)
fit!(model, X, y)   # uses the updated values
```
"""
function set_params!(m::MichiBoostWrapper; kwargs...)
    for (k, v) in kwargs
        m.params[k] = v
    end
    return m
end

"""
    get_best_iteration(model) -> Int

Return the 1-indexed iteration whose eval-metric value was best during early
stopping. When early stopping was not active, returns the total number of
iterations completed (`length(model.trees)`).
"""
function get_best_iteration(m::MichiBoostWrapper)
    m.model === nothing && error("Model has not been trained. Call fit! first.")
    return get_best_iteration(m.model)
end
get_best_iteration(model::MichiBoostModel) = model.best_iteration

"""
    get_best_score(model) -> Union{Float64,Nothing}

Return the eval-metric value at [`get_best_iteration`](@ref). Returns
`nothing` when early stopping was not active (no `eval_set` or
`early_stopping_rounds` supplied at fit time).
"""
function get_best_score(m::MichiBoostWrapper)
    m.model === nothing && error("Model has not been trained. Call fit! first.")
    return get_best_score(m.model)
end
get_best_score(model::MichiBoostModel) = model.best_score

"""
    score(model, data, y; cat_features=nothing) -> Float64

Compute a default score for the trained model on `(data, y)`.

# Arguments
- `model`: a trained [`MichiBoostRegressor`](@ref) or
  [`MichiBoostClassifier`](@ref).
- `data`: a [`Pool`](@ref), a matrix, or any `Tables.jl`-compatible source.
- `y`: target values matching the form used during training. Numeric for
  regression; original class labels (numeric or string) for classification.
- `cat_features`: categorical column indices or names. Only needed when `data`
  is not a [`Pool`](@ref).

# Returns
- **Regressor**: coefficient of determination (R²).
- **Classifier**: accuracy, the fraction of predictions matching `y`.

# Example
```julia
acc = score(clf, X_test, y_test)
r2 = score(reg, X_test, y_test)
```
"""
function score(m::MichiBoostWrapper, data, y; cat_features=nothing)
    m.model === nothing && error("Model has not been trained. Call fit! first.")
    pool = data isa Pool ? data : Pool(data; cat_features)
    length(y) == pool.n_samples || throw(
        DimensionMismatch(
            "`y` has $(length(y)) entries but `data` has $(pool.n_samples) rows"
        ),
    )
    ŷ = predict(m, pool)
    return _score(m, y, ŷ)
end

function _score(::MichiBoostRegressor, y, ŷ)
    yf = Float64.(y)
    ȳ = sum(yf) / length(yf)
    ss_res = 0.0
    ss_tot = 0.0
    @inbounds for i in eachindex(yf)
        d_res = yf[i] - ŷ[i]
        d_tot = yf[i] - ȳ
        ss_res += d_res * d_res
        ss_tot += d_tot * d_tot
    end
    return ss_tot == 0.0 ? (ss_res == 0.0 ? 1.0 : 0.0) : 1.0 - ss_res / ss_tot
end

function _score(::MichiBoostClassifier, y, ŷ)
    n = length(y)
    correct = 0
    @inbounds for i in 1:n
        correct += ŷ[i] == y[i]
    end
    return correct / n
end

"""
    shap_values(model, data; cat_features=nothing) -> Array

Compute SHAP feature attributions for each sample.

# Returns
- **Regression and binary classification**: `Matrix{Float64}` of shape
  `(n_samples, n_features)`. Each row sums approximately to
  `prediction - mean_prediction`.
- **Multi-class**: `Array{Float64,3}` of shape
  `(n_samples, n_features, n_classes)`. For each `(sample, class)` slice,
  the per-feature contributions sum approximately to that class's raw score
  minus its baseline.

# Example
```julia
shap = shap_values(model, X_test)
# shap[i, j] is the contribution of feature j to sample i's prediction
```
"""
function shap_values(m::MichiBoostWrapper, data; cat_features=nothing)
    m.model === nothing && error("Model has not been trained. Call fit! first.")
    pool = data isa Pool ? data : Pool(data; cat_features)
    return shap_values(m.model, pool)
end

# save / load

"""
    save_model(model, filepath::AbstractString)

Serialise a trained model to `filepath` using
[JLD2.jl](https://github.com/JuliaIO/JLD2.jl), an HDF5-flavoured format that
is stable across Julia versions.

Accepts either a wrapper ([`MichiBoostRegressor`](@ref),
[`MichiBoostClassifier`](@ref)) or a raw [`MichiBoostModel`](@ref).

See also [`load_model`](@ref).
"""
function save_model(m::MichiBoostWrapper, filepath::AbstractString)
    m.model === nothing && error("Model has not been trained. Call fit! first.")
    return save_model(m.model, filepath)
end

"""
    load_model(filepath::AbstractString) -> MichiBoostModel

Load a model previously saved with [`save_model`](@ref).

# Example
```julia
save_model(model, "my_model.jld2")
loaded = load_model("my_model.jld2")
predict(loaded, X_test)  # works with the raw MichiBoostModel
```
"""
function load_model end  # Actual method in io.jl

function _kfold_folds(n::Int, fold_count::Int, shuffle::Bool, rng::AbstractRNG)
    indices = shuffle ? randperm(rng, n) : collect(1:n)
    fold_size = n ÷ fold_count
    folds = Vector{Vector{Int}}(undef, fold_count)
    for k in 1:fold_count
        ts = (k - 1) * fold_size + 1
        te = k == fold_count ? n : k * fold_size
        folds[k] = collect(indices[ts:te])
    end
    return folds
end

function _stratified_folds(
    label::AbstractVector, fold_count::Int, shuffle::Bool, rng::AbstractRNG
)
    folds = [Int[] for _ in 1:fold_count]
    classes = unique(label)
    for c in classes
        c_idx = findall(==(c), label)
        if length(c_idx) < fold_count
            error(
                "Stratified CV requires every class to have at least `fold_count` " *
                "samples; class `$c` has $(length(c_idx)) but `fold_count=$fold_count`. " *
                "Stratified CV is intended for classification labels; for continuous " *
                "targets pass `stratified=false`.",
            )
        end
        if shuffle
            shuffle!(rng, c_idx)
        end
        # Round-robin distribute this class's samples across folds so each fold
        # receives ⌈n_c / fold_count⌉ or ⌊n_c / fold_count⌋ samples of class c.
        for (k, idx) in enumerate(c_idx)
            push!(folds[((k - 1) % fold_count) + 1], idx)
        end
    end
    if shuffle
        for k in 1:fold_count
            shuffle!(rng, folds[k])
        end
    end
    return folds
end

"""
    cv(pool::Pool; params=Dict(), fold_count=3, shuffle=true, stratified=false,
       random_seed=0, verbose=false, kwargs...) -> NamedTuple

Perform k-fold cross-validation on the given [`Pool`](@ref).

# Arguments
- `pool`: a [`Pool`](@ref) with labels.
- `params`: `Dict` of training hyperparameters. Keys may be `String` or
  `Symbol`.
- `fold_count`: number of folds.
- `shuffle`: shuffle indices before splitting.
- `stratified`: if `true`, each fold preserves the class proportions of the
  full label vector. Requires every class to have at least `fold_count`
  samples. Intended for classification labels.
- `random_seed`: seed for the shuffle.
- `verbose`: print per-fold results.
- `kwargs...`: additional hyperparameters, merged with `params`.

# Returns
A `NamedTuple` with fields:
- `train_loss::Vector{Float64}`: training loss per fold.
- `test_loss::Vector{Float64}`: test loss per fold.
- `mean_train_loss::Float64`
- `mean_test_loss::Float64`

# Example
```julia
pool = Pool(X; label=y)
result = cv(
    pool;
    fold_count=5,
    stratified=true,
    params=Dict(:iterations => 100, :depth => 4),
)
println("Mean test loss: ", result.mean_test_loss)
```
"""
function cv(
    pool::Pool;
    params=Dict(),
    fold_count::Int=3,
    shuffle::Bool=true,
    stratified::Bool=false,
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
    folds = if stratified
        _stratified_folds(label, fold_count, shuffle, rng)
    else
        _kfold_folds(n, fold_count, shuffle, rng)
    end
    raw_loss = get(all_params, :loss_function, "RMSE")
    loss_fn = if raw_loss isa LossFunction
        raw_loss
    else
        _to_string(raw_loss, LossKind, _loss_name, "loss_function")
    end
    boosting = _to_string(
        get(all_params, :boosting_type, "Ordered"),
        BoostingType,
        _boosting_name,
        "boosting_type",
    )

    train_losses, test_losses = Float64[], Float64[]

    for fold in 1:fold_count
        test_idx = folds[fold]
        train_idx = reduce(vcat, (folds[k] for k in 1:fold_count if k != fold))

        train_pool = slice(pool, train_idx)
        test_pool = slice(pool, test_idx)

        model = train(
            train_pool;
            iterations=Int(get(all_params, :iterations, 1000)),
            learning_rate=Float64(get(all_params, :learning_rate, 0.03)),
            depth=Int(get(all_params, :depth, 6)),
            l2_leaf_reg=Float64(get(all_params, :l2_leaf_reg, 3.0)),
            loss_function=loss_fn,
            border_count=Int(get(all_params, :border_count, 254)),
            min_data_in_leaf=Int(get(all_params, :min_data_in_leaf, 1)),
            rsm=Float64(get(all_params, :rsm, 1.0)),
            boosting_type=boosting,
            verbose=Bool(verbose),
            random_seed=random_seed,
        )

        lf = loss_fn isa LossFunction ? loss_fn : make_loss(loss_fn)

        if model.is_multiclass
            train_logits = _predict_raw(model, train_pool)
            test_logits = _predict_raw(model, test_pool)

            # Use the model's own class labels to avoid dimension mismatch when
            # a class is absent from the training fold but present in the test fold.
            n_classes = model.n_classes
            label_map = Dict(model.class_labels[i] => i for i in 1:n_classes)

            train_y = get_label(train_pool)
            train_y_onehot = zeros(Float64, length(train_y), n_classes)
            for i in eachindex(train_y)
                train_y_onehot[i, label_map[train_y[i]]] = 1.0
            end

            test_y = get_label(test_pool)
            test_y_onehot = zeros(Float64, length(test_y), n_classes)
            unseen_loss = log(Float64(n_classes))  # -log(1/n_classes): uniform prior
            unseen_count = 0
            for i in eachindex(test_y)
                col = get(label_map, test_y[i], 0)
                if col > 0
                    test_y_onehot[i, col] = 1.0
                else
                    unseen_count += 1
                end
            end

            seen_count = length(test_y) - unseen_count
            base_test_loss = if seen_count > 0
                # Compute loss only over samples whose class the model knows
                seen_mask = [get(label_map, test_y[i], 0) > 0 for i in eachindex(test_y)]
                loss(lf, test_y_onehot[seen_mask, :], test_logits[seen_mask, :])
            else
                0.0
            end
            # Blend in uniform-prior loss for unseen samples
            test_loss =
                (base_test_loss * seen_count + unseen_loss * unseen_count) / length(test_y)

            push!(train_losses, loss(lf, train_y_onehot, train_logits))
            push!(test_losses, test_loss)
        elseif model.n_classes == 2
            train_logits = _predict_raw(model, train_pool)
            test_logits = _predict_raw(model, test_pool)

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
