function build_symmetric_tree(
    gradients::AbstractVector{Float64},
    hessians::AbstractVector{Float64},
    num_bins::AbstractMatrix{UInt16},
    cat_encoded::AbstractMatrix{Float64},
    sample_indices::Vector{Int},
    depth::Int,
    n_num::Int,
    n_cat::Int,
    qf::QuantizedFeatures;
    l2_leaf_reg::Float64=3.0,
    min_data_in_leaf::Int=1,
    rsm::Float64=1.0,
    rng::AbstractRNG=MersenneTwister(),
    buffers::Vector{SplitBuffers}=[
        SplitBuffers(1 << depth, maximum(qf.n_bins; init=1) + 1, length(sample_indices))
        for _ in 1:Threads.maxthreadid()
    ],
    cat_sorted_vals::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
)
    n_features = n_num + n_cat
    n_sampled = max(1, round(Int, rsm * n_features))

    split_features, split_types, split_thresholds = Int[], Symbol[], Float64[]
    n = length(sample_indices)
    copyto!(view(buffers[1].indices, 1:n), sample_indices)
    leaf_groups = [view(buffers[1].indices, 1:n)]

    for _ in 1:depth
        sampled = randperm(rng, n_features)[1:n_sampled]
        sampled_num = filter(i -> i <= n_num, sampled)
        sampled_cat = filter(i -> i > n_num, sampled) .- n_num

        best = _find_best_split_across_leaves(
            gradients,
            hessians,
            num_bins,
            cat_encoded,
            leaf_groups,
            sampled_num,
            sampled_cat,
            qf,
            buffers,
            cat_sorted_vals;
            l2_leaf_reg,
            min_data_in_leaf,
        )
        leaf_groups = _apply_split!(
            split_features,
            split_types,
            split_thresholds,
            leaf_groups,
            best,
            num_bins,
            cat_encoded,
            n_num,
            buffers[1],
        )
    end

    n_leaves = 1 << depth
    leaf_values = Vector{Float64}(undef, n_leaves)
    @inbounds for l in 1:n_leaves
        group = leaf_groups[l]
        if isempty(group)
            leaf_values[l] = 0.0
        else
            g_sum, h_sum = 0.0, 0.0
            for idx in group
                g_sum += gradients[idx]
                h_sum += hessians[idx]
            end
            leaf_values[l] = g_sum / (h_sum + l2_leaf_reg)
        end
    end
    return SymmetricTree(depth, split_features, split_types, split_thresholds, leaf_values)
end

# Shared utility used by both build_symmetric_tree and build_symmetric_tree_multiclass
function _apply_split!(
    split_features,
    split_types,
    split_thresholds,
    leaf_groups,
    best::SplitCandidate,
    num_bins,
    cat_encoded,
    n_num,
    buf,
)
    if best.feature_index == 0
        push!(split_features, 1)
        push!(split_types, n_num > 0 ? :numerical : :categorical)
        push!(split_thresholds, 0.0)
        new_groups = Vector{SubArray}(undef, 2 * length(leaf_groups))
        for (li, group) in enumerate(leaf_groups)
            new_groups[2 * li - 1] = group
            new_groups[2 * li] = view(buf.indices, 1:0)
        end
        return new_groups
    end

    push!(split_features, best.feature_index)
    push!(split_types, best.feature_type)
    push!(split_thresholds, best.threshold)

    new_groups = Vector{Any}(undef, 2 * length(leaf_groups))
    offset = 1
    @inbounds for (li, group) in enumerate(leaf_groups)
        n = length(group)
        left_buf = view(buf.indices, offset:(offset + n - 1))
        right_buf = view(buf.indices_tmp, offset:(offset + n - 1))
        lc = 0
        rc = 0
        for idx in group
            if _goes_right(best, num_bins, cat_encoded, idx)
                rc += 1
                right_buf[rc] = idx
            else
                lc += 1
                left_buf[lc] = idx
            end
        end
        for i in 1:rc
            buf.indices[offset + lc + i - 1] = right_buf[i]
        end
        new_groups[2 * li - 1] = view(buf.indices, offset:(offset + lc - 1))
        new_groups[2 * li] = view(buf.indices, (offset + lc):(offset + n - 1))
        offset += n
    end
    return new_groups
end

@inline function _goes_right(split::SplitCandidate, num_bins, cat_encoded, idx)
    if split.feature_type == :numerical
        return num_bins[idx, split.feature_index] > UInt16(split.threshold)
    else
        return cat_encoded[idx, split.feature_index] > split.threshold
    end
end
