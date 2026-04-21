function build_symmetric_tree(
    gradients::AbstractMatrix{Float64},
    hessians::AbstractMatrix{Float64},
    num_bins::AbstractMatrix{UInt16},
    cat_encoded::AbstractMatrix{Float64},
    sample_indices::Vector{Int},
    depth::Int,
    n_num::Int,
    n_cat::Int,
    qf::QuantizedFeatures,
    n_classes::Int;
    l2_leaf_reg::Float64=3.0,
    min_data_in_leaf::Int=1,
    rsm::Float64=1.0,
    rng::AbstractRNG=MersenneTwister(),
    buffers::Vector{SplitBuffersMC}=[
        SplitBuffersMC(1 << depth, maximum(qf.n_bins; init=1) + 1, n_classes, length(sample_indices))
        for _ in 1:Threads.maxthreadid()
    ],
    cat_sorted_vals::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
    hist_cache::HistCacheMC=HistCacheMC(1 << depth, qf.n_bins, cat_sorted_vals, n_classes),
)
    n_features = n_num + n_cat
    n_sampled = max(1, round(Int, rsm * n_features))

    split_features, split_types, split_thresholds = Int[], Symbol[], Float64[]
    n = length(sample_indices)
    copyto!(view(buffers[1].indices, 1:n), sample_indices)
    leaf_groups = [view(buffers[1].indices, 1:n)]

    reset_hist_cache!(hist_cache)

    for _ in 1:depth
        sampled = randperm(rng, n_features)[1:n_sampled]
        sampled_num = filter(i -> i <= n_num, sampled)
        sampled_cat = filter(i -> i > n_num, sampled) .- n_num

        best = _find_best_split_across_leaves_mc(
            gradients,
            hessians,
            num_bins,
            cat_encoded,
            leaf_groups,
            sampled_num,
            sampled_cat,
            qf,
            n_classes,
            buffers,
            cat_sorted_vals,
            hist_cache;
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
        rotate_hist_cache!(hist_cache)
    end

    n_leaves = 1 << depth
    leaf_values = Matrix{Float64}(undef, n_leaves, n_classes)
    # Each thread writes to a disjoint row of leaf_values.
    Threads.@threads :static for l in 1:n_leaves
        group = leaf_groups[l]
        if isempty(group)
            @inbounds for c in 1:n_classes
                leaf_values[l, c] = 0.0
            end
        else
            _fill_leaf_values_mc!(leaf_values, l, group, gradients, hessians, n_classes, l2_leaf_reg)
        end
    end
    return SymmetricTreeMultiClass(depth, split_features, split_types, split_thresholds, leaf_values)
end

# Function-barrier helper for the multiclass Newton step: without this, the
# @threads closure captures g_sum / h_sum in a Core.Box and every `+=`
# allocates a fresh boxed Float64 — up to ~1.1M per boosting round.
@inline function _fill_leaf_values_mc!(
    leaf_values, l, group, gradients, hessians, n_classes, l2_leaf_reg,
)
    n_leaf = length(group)
    @inbounds for c in 1:n_classes
        g_sum = 0.0
        h_sum = 0.0
        for k in 1:n_leaf
            idx = group[k]
            g_sum += gradients[idx, c]
            h_sum += hessians[idx, c]
        end
        leaf_values[l, c] = g_sum / (h_sum + l2_leaf_reg)
    end
    return nothing
end
