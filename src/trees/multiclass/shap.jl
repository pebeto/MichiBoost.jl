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
    n_classes = size(tree.leaf_values, 2)
    path_prefix = 0

    for k in 1:depth
        bit_pos = depth - k
        half = 1 << bit_pos
        lo = path_prefix << (bit_pos + 1)

        feat_idx = tree.split_feature_indices[k]
        orig_col = if tree.split_feature_types[k] == :numerical
            num_feat_indices[feat_idx]
        else
            cat_feat_indices[feat_idx]
        end

        went_right = (leaf_idx >> bit_pos) & 1

        for c in 1:n_classes
            sum_left  = sum(@view tree.leaf_values[lo+1      : lo+half,    c])
            sum_right = sum(@view tree.leaf_values[lo+half+1 : lo+2*half,  c])
            mean_left  = sum_left  / half
            mean_right = sum_right / half
            contribution = lr * (went_right == 1 ? mean_right - mean_left : mean_left - mean_right) / 2.0
            shap_row[orig_col, c] += contribution
        end

        path_prefix = (path_prefix << 1) | went_right
    end
    return nothing
end
