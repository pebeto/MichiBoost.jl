"""Add tree predictions into `out` in-place (multiclass)."""
function predict_tree!(
    out::AbstractMatrix{Float64},
    tree::SymmetricTreeMultiClass,
    num_bins::AbstractMatrix{UInt16},
    cat_encoded::AbstractMatrix{Float64},
    lr::Float64,
    leaf_indices::Vector{Int},
)
    n = size(out, 1)
    n_classes = size(tree.leaf_values, 2)
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
    @inbounds for i in 1:n
        leaf = leaf_indices[i] + 1
        for c in 1:n_classes
            out[i, c] += lr * tree.leaf_values[leaf, c]
        end
    end
    return nothing
end

function predict_tree(tree::SymmetricTreeMultiClass, num_bins, cat_encoded)
    n = size(num_bins, 1)
    nc = size(tree.leaf_values, 2)
    leaf_indices = zeros(Int, n)
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
    out = Matrix{Float64}(undef, n, nc)
    @inbounds for i in 1:n
        leaf = leaf_indices[i] + 1
        for c in 1:nc
            out[i, c] = tree.leaf_values[leaf, c]
        end
    end
    return out
end
