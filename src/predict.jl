function _prepare_features(model::MichiBoostModel, pool::Pool)
    n = pool.n_samples
    num_bins = if n_numerical(pool) > 0
        apply_borders(pool.features_numerical, model.borders)
    else
        Matrix{UInt16}(undef, n, 0)
    end
    cat_encoded = if n_categorical(pool) > 0 && model.encoder !== nothing
        encode_categorical(model.encoder, pool.features_categorical)
    else
        Matrix{Float64}(undef, n, 0)
    end
    return num_bins, cat_encoded
end

"""
    predict(model::MichiBoostModel, pool::Pool)

Low-level prediction on a fitted `MichiBoostModel`.

Returns:
- **Regression**: `Vector{Float64}` of predicted values.
- **Binary classification**: `Vector{Float64}` of P(positive class).
- **Multi-class**: `Matrix{Float64}` of probabilities (rows × classes).

Most users should call `predict(wrapper_model, data)` instead.
"""
function predict(model::MichiBoostModel, pool::Pool)
    num_bins, cat_encoded = _prepare_features(model, pool)
    n = pool.n_samples
    n_trees = length(model.trees)
    nt = Threads.nthreads()

    if model.is_multiclass
        partials = [zeros(Float64, n, model.n_classes) for _ in 1:nt]
        leaf_bufs = [Vector{Int}(undef, n) for _ in 1:nt]
        Threads.@threads :static for k in 1:nt
            lo = div((k - 1) * n_trees, nt) + 1
            hi = div(k * n_trees, nt)
            lbuf = leaf_bufs[k]
            for i in lo:hi
                predict_tree!(
                    partials[k], model.trees[i], num_bins, cat_encoded, model.learning_rate, lbuf
                )
            end
        end
        preds = repeat(model.initial_pred', n, 1)
        for t in 1:nt
            preds .+= partials[t]
        end
        return _softmax_matrix(preds)
    else
        partials = [zeros(Float64, n) for _ in 1:nt]
        leaf_bufs = [Vector{Int}(undef, n) for _ in 1:nt]
        Threads.@threads :static for k in 1:nt
            lo = div((k - 1) * n_trees, nt) + 1
            hi = div(k * n_trees, nt)
            lbuf = leaf_bufs[k]
            for i in lo:hi
                predict_tree!(
                    partials[k], model.trees[i], num_bins, cat_encoded, model.learning_rate, lbuf
                )
            end
        end
        preds = fill(model.initial_pred::Float64, n)
        for t in 1:nt
            preds .+= partials[t]
        end
        return model.n_classes == 2 ? _sigmoid.(preds) : preds
    end
end

"""
    predict_classes(model::MichiBoostModel, pool::Pool)

Return predicted class labels from a fitted `MichiBoostModel`.

- **Binary**: returns the label with probability ≥ 0.5.
- **Multi-class**: returns the label with the highest probability.
"""
function predict_classes(model::MichiBoostModel, pool::Pool)
    preds = predict(model, pool)
    if model.is_multiclass
        return [model.class_labels[argmax(preds[i, :])] for i in axes(preds, 1)]
    elseif model.n_classes == 2
        return [
            preds[i] >= 0.5 ? model.class_labels[2] : model.class_labels[1] for
            i in eachindex(preds)
        ]
    else
        return error("predict_classes is only for classification models")
    end
end

"""
    feature_importance(model::MichiBoostModel) -> Vector{Pair{Symbol, Float64}}

Compute feature importances based on split frequency across all trees.

Each pair is `feature_name => percentage`, sorted descending.  Percentages sum
to 100.  Features never used in any split get 0.
"""
function feature_importance(model::MichiBoostModel)
    n_num = length(model.borders)
    n_cat = if model.encoder !== nothing
        length(model.encoder.category_stats)
    else
        0
    end
    importance = Dict{Symbol,Float64}()

    for tree in model.trees, k in 1:tree.depth
        name = if tree.split_feature_types[k] == :numerical
            j = tree.split_feature_indices[k]
            if j <= length(model.numerical_feature_indices)
                model.feature_names[model.numerical_feature_indices[j]]
            else
                Symbol("num_$j")
            end
        else
            j = tree.split_feature_indices[k]
            if j <= length(model.categorical_feature_indices)
                model.feature_names[model.categorical_feature_indices[j]]
            else
                Symbol("cat_$j")
            end
        end
        importance[name] = get(importance, name, 0.0) + 1.0
    end

    total = max(sum(values(importance); init=0.0), 1e-10)

    # Collect all feature names in original column order
    all_names = model.feature_names

    result = Pair{Symbol,Float64}[]
    for (name, imp) in sort(collect(importance); by=x -> -x[2])
        push!(result, name => 100.0 * imp / total)
    end
    for name in all_names
        haskey(importance, name) || push!(result, name => 0.0)
    end
    return result
end
