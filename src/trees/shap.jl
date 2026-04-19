"""
    shap_values(model::MichiBoostModel, pool::Pool) -> Array

Compute SHAP feature attributions for each sample in `pool`.

Returns:
- **Regression / Binary**: `Matrix{Float64}` of shape `(n_samples, n_features)`.
- **Multiclass**: `Array{Float64,3}` of shape `(n_samples, n_features, n_classes)`.

Each row sums exactly to `raw_prediction - E[raw_prediction]`, where the
expectation is taken under a uniform distribution over tree leaves.
"""
function shap_values(model::MichiBoostModel, pool::Pool)
    num_bins, cat_encoded = _prepare_features(model, pool)
    n = pool.n_samples
    n_features = pool.n_features
    lr = model.learning_rate
    trees = model.trees

    if model.is_multiclass
        n_classes = model.n_classes
        shap = zeros(Float64, n, n_features, n_classes)

        Threads.@threads :static for i in 1:n
            for tree in trees
                leaf_idx = 0
                @inbounds for k in 1:tree.depth
                    feat_idx = tree.split_feature_indices[k]
                    thr = tree.split_thresholds[k]
                    if tree.split_feature_types[k] == :numerical
                        leaf_idx = (leaf_idx << 1) | Int(num_bins[i, feat_idx] > UInt16(thr))
                    else
                        leaf_idx = (leaf_idx << 1) | Int(cat_encoded[i, feat_idx] > thr)
                    end
                end
                _shap_tree!(
                    view(shap, i, :, :),
                    tree,
                    leaf_idx,
                    lr,
                    model.numerical_feature_indices,
                    model.categorical_feature_indices,
                )
            end
        end
        return shap
    else
        shap = zeros(Float64, n, n_features)

        Threads.@threads :static for i in 1:n
            for tree in trees
                leaf_idx = 0
                @inbounds for k in 1:tree.depth
                    feat_idx = tree.split_feature_indices[k]
                    thr = tree.split_thresholds[k]
                    if tree.split_feature_types[k] == :numerical
                        leaf_idx = (leaf_idx << 1) | Int(num_bins[i, feat_idx] > UInt16(thr))
                    else
                        leaf_idx = (leaf_idx << 1) | Int(cat_encoded[i, feat_idx] > thr)
                    end
                end
                _shap_tree!(
                    view(shap, i, :),
                    tree,
                    leaf_idx,
                    lr,
                    model.numerical_feature_indices,
                    model.categorical_feature_indices,
                )
            end
        end
        return shap
    end
end
