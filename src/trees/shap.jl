"""
    shap_values(model::MichiBoostModel, pool::Pool) -> Array

Compute SHAP feature attributions for each sample in `pool`.

Returns:
- **Regression / Binary**: `Matrix{Float64}` of shape `(n_samples, n_features)`.
- **Multiclass**: `Array{Float64,3}` of shape `(n_samples, n_features, n_classes)`.

Each row sums (approximately) to `prediction - expected_prediction`.
"""
function shap_values(model::MichiBoostModel, pool::Pool)
    num_bins, cat_encoded = _prepare_features(model, pool)
    n = pool.n_samples
    n_features = pool.n_features
    lr = model.learning_rate

    # Compute leaf indices for every sample across all trees
    leaf_indices = zeros(Int, n)

    if model.is_multiclass
        n_classes = model.n_classes
        shap = zeros(Float64, n, n_features, n_classes)

        for tree in model.trees
            # Compute leaf index per sample
            fill!(leaf_indices, 0)
            @inbounds for k in 1:tree.depth
                feat_idx = tree.split_feature_indices[k]
                thr = tree.split_thresholds[k]
                if tree.split_feature_types[k] == :numerical
                    thr_u = UInt16(thr)
                    for i in 1:n
                        leaf_indices[i] = (leaf_indices[i] << 1) | Int(num_bins[i, feat_idx] > thr_u)
                    end
                else
                    for i in 1:n
                        leaf_indices[i] = (leaf_indices[i] << 1) | Int(cat_encoded[i, feat_idx] > thr)
                    end
                end
            end
            for i in 1:n
                _shap_tree!(
                    view(shap, i, :, :),
                    tree,
                    leaf_indices[i],
                    lr,
                    model.numerical_feature_indices,
                    model.categorical_feature_indices,
                )
            end
        end
        return shap
    else
        shap = zeros(Float64, n, n_features)

        for tree in model.trees
            fill!(leaf_indices, 0)
            @inbounds for k in 1:tree.depth
                feat_idx = tree.split_feature_indices[k]
                thr = tree.split_thresholds[k]
                if tree.split_feature_types[k] == :numerical
                    thr_u = UInt16(thr)
                    for i in 1:n
                        leaf_indices[i] = (leaf_indices[i] << 1) | Int(num_bins[i, feat_idx] > thr_u)
                    end
                else
                    for i in 1:n
                        leaf_indices[i] = (leaf_indices[i] << 1) | Int(cat_encoded[i, feat_idx] > thr)
                    end
                end
            end
            for i in 1:n
                _shap_tree!(
                    view(shap, i, :),
                    tree,
                    leaf_indices[i],
                    lr,
                    model.numerical_feature_indices,
                    model.categorical_feature_indices,
                )
            end
        end
        return shap
    end
end
