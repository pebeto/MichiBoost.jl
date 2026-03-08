"""Add tree predictions into `out` in-place (no allocation)."""
function predict_tree!(out::AbstractVector{Float64},
                       tree::SymmetricTree,
                       num_bins::AbstractMatrix{UInt16},
                       cat_encoded::AbstractMatrix{Float64},
                       lr::Float64)
    @inbounds for i in eachindex(out)
        out[i] += lr * tree.leaf_values[_leaf_index(tree, num_bins, cat_encoded, i) + 1]
    end
    return nothing
end

"""Add tree predictions into `out` in-place (multiclass, no allocation)."""
function predict_tree_mc!(out::AbstractMatrix{Float64},
                          tree::SymmetricTreeMultiClass,
                          num_bins::AbstractMatrix{UInt16},
                          cat_encoded::AbstractMatrix{Float64},
                          lr::Float64)
    n_classes = size(tree.leaf_values, 2)
    @inbounds for i in axes(out, 1)
        leaf = _leaf_index(tree, num_bins, cat_encoded, i) + 1
        for c in 1:n_classes
            out[i, c] += lr * tree.leaf_values[leaf, c]
        end
    end
    return nothing
end

function predict_tree(tree::SymmetricTree, num_bins, cat_encoded)
    n = size(num_bins, 1)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        out[i] = tree.leaf_values[_leaf_index(tree, num_bins, cat_encoded, i) + 1]
    end
    return out
end

function predict_tree_multiclass(tree::SymmetricTreeMultiClass, num_bins, cat_encoded)
    n = size(num_bins, 1)
    nc = size(tree.leaf_values, 2)
    out = Matrix{Float64}(undef, n, nc)
    @inbounds for i in 1:n
        leaf = _leaf_index(tree, num_bins, cat_encoded, i) + 1
        for c in 1:nc; out[i, c] = tree.leaf_values[leaf, c]; end
    end
    return out
end

"""Compute the 0-based leaf index for sample `i`."""
@inline function _leaf_index(tree, num_bins, cat_encoded, i)
    leaf_idx = 0
    for k in 1:tree.depth
        feat_idx = tree.split_feature_indices[k]
        threshold = tree.split_thresholds[k]
        bit = if tree.split_feature_types[k] == :numerical
            num_bins[i, feat_idx] > UInt16(threshold) ? 1 : 0
        else
            cat_encoded[i, feat_idx] > threshold ? 1 : 0
        end
        leaf_idx = (leaf_idx << 1) | bit
    end
    return leaf_idx
end
