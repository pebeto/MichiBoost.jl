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
                    partials[k],
                    model.trees[i],
                    num_bins,
                    cat_encoded,
                    model.learning_rate,
                    lbuf,
                )
            end
        end
        preds = repeat(model.initial_pred', n, 1)
        for t in 1:nt
            preds .+= partials[t]
        end
        return if model.custom_loss !== nothing
            link_inverse(model.custom_loss, preds)
        else
            _softmax_matrix(preds)
        end
    else
        partials = [zeros(Float64, n) for _ in 1:nt]
        leaf_bufs = [Vector{Int}(undef, n) for _ in 1:nt]
        Threads.@threads :static for k in 1:nt
            lo = div((k - 1) * n_trees, nt) + 1
            hi = div(k * n_trees, nt)
            lbuf = leaf_bufs[k]
            for i in lo:hi
                predict_tree!(
                    partials[k],
                    model.trees[i],
                    num_bins,
                    cat_encoded,
                    model.learning_rate,
                    lbuf,
                )
            end
        end
        preds = fill(model.initial_pred::Float64, n)
        for t in 1:nt
            preds .+= partials[t]
        end
        if model.custom_loss !== nothing
            return link_inverse(model.custom_loss, preds)
        end
        return model.n_classes == 2 ? _sigmoid.(preds) : preds
    end
end

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

function feature_importance(model::MichiBoostModel)
    importance = Dict{Symbol,Float64}()

    for tree in model.trees, k in 1:(tree.depth)
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

    result = Pair{Symbol,Float64}[]
    for (name, imp) in sort(collect(importance); by=x -> -x[2])
        push!(result, name => 100.0 * imp / total)
    end
    for name in model.feature_names
        haskey(importance, name) || push!(result, name => 0.0)
    end
    return result
end
