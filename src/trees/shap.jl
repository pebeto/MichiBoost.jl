"""
    _shap_tree!(shap_row, tree, leaf_idx, lr)

Add the SHAP contributions of one `SymmetricTree` for one sample (identified
by its 0-based `leaf_idx`) into `shap_row` (length = n_original_features).
`lr` is the learning rate used to scale leaf values.
"""
function _shap_tree!(
    shap_row::AbstractVector{Float64},
    tree::SymmetricTree,
    leaf_idx::Int,
    lr::Float64,
    num_feat_indices::Vector{Int},
    cat_feat_indices::Vector{Int},
)
    depth = tree.depth
    n_leaves = 1 << depth

    for k in 1:depth
        # Which bit position corresponds to level k?
        # Level 1 is the most-significant bit (depth-1 shift), level depth is bit 0.
        bit_pos = depth - k
        went_right = (leaf_idx >> bit_pos) & 1

        # Mean leaf value over left subtree (bit_pos = 0) and right (bit_pos = 1).
        # The mask for "right" leaves at this level: bit bit_pos is set.
        mask = 1 << bit_pos
        sum_left = 0.0
        sum_right = 0.0
        count_left = 0
        count_right = 0
        for leaf in 0:(n_leaves - 1)
            if (leaf & mask) == 0
                sum_left += tree.leaf_values[leaf + 1]
                count_left += 1
            else
                sum_right += tree.leaf_values[leaf + 1]
                count_right += 1
            end
        end
        mean_left = sum_left / count_left
        mean_right = sum_right / count_right

        contribution = lr * (went_right == 1 ? mean_right - mean_left : mean_left - mean_right) / 2.0

        # Map the split feature index back to the original feature column.
        feat_idx = tree.split_feature_indices[k]
        orig_col = if tree.split_feature_types[k] == :numerical
            num_feat_indices[feat_idx]
        else
            cat_feat_indices[feat_idx]
        end
        shap_row[orig_col] += contribution
    end
    return nothing
end

"""
    _shap_tree_mc!(shap_row, tree, leaf_idx, lr, n_classes)

Multiclass variant — `shap_row` has shape `(n_original_features, n_classes)`.
"""
function _shap_tree_mc!(
    shap_row::AbstractMatrix{Float64},
    tree::SymmetricTreeMultiClass,
    leaf_idx::Int,
    lr::Float64,
    num_feat_indices::Vector{Int},
    cat_feat_indices::Vector{Int},
)
    depth = tree.depth
    n_leaves = 1 << depth
    n_classes = size(tree.leaf_values, 2)

    for k in 1:depth
        bit_pos = depth - k
        went_right = (leaf_idx >> bit_pos) & 1
        mask = 1 << bit_pos

        sum_left = zeros(Float64, n_classes)
        sum_right = zeros(Float64, n_classes)
        count_left = 0
        count_right = 0
        for leaf in 0:(n_leaves - 1)
            if (leaf & mask) == 0
                for c in 1:n_classes
                    sum_left[c] += tree.leaf_values[leaf + 1, c]
                end
                count_left += 1
            else
                for c in 1:n_classes
                    sum_right[c] += tree.leaf_values[leaf + 1, c]
                end
                count_right += 1
            end
        end

        feat_idx = tree.split_feature_indices[k]
        orig_col = if tree.split_feature_types[k] == :numerical
            num_feat_indices[feat_idx]
        else
            cat_feat_indices[feat_idx]
        end

        for c in 1:n_classes
            mean_left = sum_left[c] / count_left
            mean_right = sum_right[c] / count_right
            contribution = lr * (went_right == 1 ? mean_right - mean_left : mean_left - mean_right) / 2.0
            shap_row[orig_col, c] += contribution
        end
    end
    return nothing
end

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
                _shap_tree_mc!(
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
