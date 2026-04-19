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
    path_prefix = 0  # tracks the 0-indexed first leaf of the current subtree

    for k in 1:depth
        bit_pos = depth - k
        half = 1 << bit_pos                # leaves in each child subtree
        lo = path_prefix << (bit_pos + 1)  # 0-indexed start of current subtree

        # Path-conditioned means: average only over leaves reachable from here.
        sum_left  = sum(@view tree.leaf_values[lo+1      : lo+half])
        sum_right = sum(@view tree.leaf_values[lo+half+1 : lo+2*half])
        mean_left  = sum_left  / half
        mean_right = sum_right / half

        went_right = (leaf_idx >> bit_pos) & 1
        contribution = lr * (went_right == 1 ? mean_right - mean_left : mean_left - mean_right) / 2.0

        feat_idx = tree.split_feature_indices[k]
        orig_col = if tree.split_feature_types[k] == :numerical
            num_feat_indices[feat_idx]
        else
            cat_feat_indices[feat_idx]
        end
        shap_row[orig_col] += contribution

        path_prefix = (path_prefix << 1) | went_right
    end
    return nothing
end
