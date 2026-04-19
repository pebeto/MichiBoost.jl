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
