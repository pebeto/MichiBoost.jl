"""
    _shap_tree!(shap_row, tree, leaf_idx, lr, n_classes)

Multiclass variant — `shap_row` has shape `(n_original_features, n_classes)`.
"""
function _shap_tree!(
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
