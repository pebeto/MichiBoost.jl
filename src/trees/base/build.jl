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
    buffers::Vector{SplitBuffers},
    cat_sorted_vals::Vector{Vector{Float64}},
    hist_cache::HistCache,
    leaf_refine_values::Union{Nothing,AbstractVector{Float64}}=nothing,
    leaf_refine_weights::Union{Nothing,AbstractVector{Float64}}=nothing,
    leaf_reduce::Union{Nothing,Function}=nothing,
    monotone::Union{Nothing,Vector{Int}}=nothing,
)
    n_features = n_num + n_cat
    n_sampled = max(1, round(Int, rsm * n_features))

    split_features, split_types, split_thresholds = Int[], Symbol[], Float64[]
    n = length(sample_indices)
    copyto!(view(buffers[1].indices, 1:n), sample_indices)
    leaf_groups = [view(buffers[1].indices, 1:n)]

    reset_hist_cache!(hist_cache)

    # Per-node value bounds for monotonicity. Each split on a constrained
    # feature narrows the children's bounds around a midpoint so the whole left
    # subtree stays below the whole right subtree (or above, for a -1 sign).
    # Final leaf values are clamped into these bounds, which guarantees the tree
    # is monotone in the constrained feature. See `_propagate_monotone_bounds`.
    use_mono = monotone !== nothing
    lo = use_mono ? Float64[-Inf] : Float64[]
    hi = use_mono ? Float64[Inf] : Float64[]

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
            cat_sorted_vals,
            hist_cache;
            l2_leaf_reg,
            min_data_in_leaf,
        )
        parent_lo, parent_hi = lo, hi
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
        if use_mono
            lo, hi = _propagate_monotone_bounds(
                parent_lo,
                parent_hi,
                leaf_groups,
                best,
                monotone,
                gradients,
                hessians,
                l2_leaf_reg,
            )
        end
    end

    n_leaves = 1 << depth
    leaf_values = Vector{Float64}(undef, n_leaves)
    refine = leaf_refine_values !== nothing
    # Leaves write to disjoint leaf_values[l] slots, so this loop parallelises
    # safely.  The refine path allocates its small scratch vectors locally
    # per-leaf to keep that safety without a per-thread scratch pool.
    Threads.@threads :static for l in 1:n_leaves
        group = leaf_groups[l]
        v = if isempty(group)
            0.0
        elseif refine
            _leaf_value_refine(group, leaf_refine_values, leaf_refine_weights, leaf_reduce)
        else
            _leaf_value_newton(group, gradients, hessians, l2_leaf_reg)
        end
        leaf_values[l] = use_mono ? clamp(v, lo[l], hi[l]) : v
    end
    return SymmetricTree(depth, split_features, split_types, split_thresholds, leaf_values)
end

# Narrow each child node's value bounds so a monotone split keeps the left
# subtree below the right (sign +1) or above it (sign -1). `best.feature_index`
# is the numerical-block index, matching the `monotone` vector; categorical and
# degenerate (feature 0) splits pass the parent bounds through unchanged.
function _propagate_monotone_bounds(
    lo, hi, leaf_groups, best::SplitCandidate, monotone, gradients, hessians, l2_leaf_reg
)
    n_parents = length(lo)
    new_lo = Vector{Float64}(undef, 2 * n_parents)
    new_hi = Vector{Float64}(undef, 2 * n_parents)
    sign =
        (best.feature_index != 0 && !best.is_categorical) ? monotone[best.feature_index] : 0
    @inbounds for p in 1:n_parents
        plo, phi = lo[p], hi[p]
        if sign == 0
            new_lo[2p - 1] = plo
            new_hi[2p - 1] = phi
            new_lo[2p] = plo
            new_hi[2p] = phi
        else
            left, right = leaf_groups[2p - 1], leaf_groups[2p]
            vl = if isempty(left)
                0.0
            else
                _leaf_value_newton(left, gradients, hessians, l2_leaf_reg)
            end
            vr = if isempty(right)
                0.0
            else
                _leaf_value_newton(right, gradients, hessians, l2_leaf_reg)
            end
            mid = 0.5 * (clamp(vl, plo, phi) + clamp(vr, plo, phi))
            if sign > 0
                new_lo[2p - 1], new_hi[2p - 1] = plo, mid
                new_lo[2p], new_hi[2p] = mid, phi
            else
                new_lo[2p - 1], new_hi[2p - 1] = mid, phi
                new_lo[2p], new_hi[2p] = plo, mid
            end
        end
    end
    return new_lo, new_hi
end

# Function-barrier helpers: keep g_sum / h_sum as plain locals rather than
# letting the @threads closure box them as Core.Box (which happens when the
# arithmetic sits inline inside the threaded body).
@inline function _leaf_value_newton(group, gradients, hessians, l2_leaf_reg)
    g_sum = 0.0
    h_sum = 0.0
    n_leaf = length(group)
    @inbounds for j in 1:n_leaf
        idx = group[j]
        g_sum += gradients[idx]
        h_sum += hessians[idx]
    end
    return g_sum / (h_sum + l2_leaf_reg)
end

@inline function _leaf_value_refine(group, leaf_refine_values, leaf_refine_weights, reduce)
    n_leaf = length(group)
    local_vals = Vector{Float64}(undef, n_leaf)
    local_ws = Vector{Float64}(undef, n_leaf)
    @inbounds for j in 1:n_leaf
        idx = group[j]
        local_vals[j] = leaf_refine_values[idx]
        local_ws[j] = leaf_refine_weights === nothing ? 1.0 : leaf_refine_weights[idx]
    end
    return if reduce === nothing
        weighted_median(local_vals, local_ws)
    else
        reduce(local_vals, local_ws)
    end
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
        new_groups = Vector{LeafGroupView}(undef, 2 * length(leaf_groups))
        @inbounds for li in 1:length(leaf_groups)
            new_groups[2 * li - 1] = leaf_groups[li]
            new_groups[2 * li] = view(buf.indices, 1:0)
        end
        return new_groups
    end

    push!(split_features, best.feature_index)
    push!(split_types, best.is_categorical ? :categorical : :numerical)
    push!(split_thresholds, best.threshold)

    n_leaves = length(leaf_groups)
    new_groups = Vector{LeafGroupView}(undef, 2 * n_leaves)

    # Prefix-sum of leaf sizes so every leaf has a disjoint, preallocated
    # [offset, offset+n) region inside buf.indices / buf.indices_tmp.  With
    # non-overlapping ranges the per-leaf partition can run on separate
    # threads without any synchronisation.
    offsets = Vector{Int}(undef, n_leaves + 1)
    offsets[1] = 1
    @inbounds for li in 1:n_leaves
        offsets[li + 1] = offsets[li] + length(leaf_groups[li])
    end
    leaf_lc = Vector{Int}(undef, n_leaves)

    Threads.@threads :static for li in 1:n_leaves
        group = leaf_groups[li]
        n = length(group)
        offset = offsets[li]
        left_buf = view(buf.indices, offset:(offset + n - 1))
        right_buf = view(buf.indices_tmp, offset:(offset + n - 1))
        lc = 0
        rc = 0
        @inbounds for k in 1:n
            idx = group[k]
            if _goes_right(best, num_bins, cat_encoded, idx)
                rc += 1
                right_buf[rc] = idx
            else
                lc += 1
                left_buf[lc] = idx
            end
        end
        @inbounds for i in 1:rc
            buf.indices[offset + lc + i - 1] = right_buf[i]
        end
        leaf_lc[li] = lc
    end

    @inbounds for li in 1:n_leaves
        offset = offsets[li]
        n = offsets[li + 1] - offsets[li]
        lc = leaf_lc[li]
        new_groups[2 * li - 1] = view(buf.indices, offset:(offset + lc - 1))
        new_groups[2 * li] = view(buf.indices, (offset + lc):(offset + n - 1))
    end
    return new_groups
end

@inline function _goes_right(split::SplitCandidate, num_bins, cat_encoded, idx)
    if split.is_categorical
        return cat_encoded[idx, split.feature_index] > split.threshold
    else
        return num_bins[idx, split.feature_index] > UInt16(split.threshold)
    end
end
