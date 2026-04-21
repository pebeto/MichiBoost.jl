"""
    _fill_num_leaf!(hist_g, hist_h, hist_c, li, group, j, gradients, hessians, num_bins, buf)

Fill row `li` of a numerical histogram from the samples in `group`, returning
the (g_sum, h_sum, n) totals. Exists as a standalone function so Julia
specializes on the concrete type of `group` (function barrier) — calling it
from a loop with `group::Any` keeps the inner loop typed.
"""
function _fill_num_leaf!(
    hist_g, hist_h, hist_c, li::Int, group, j::Int,
    gradients, hessians, num_bins, buf,
)
    n = length(group)
    g_sum = 0.0
    h_sum = 0.0
    if n >= 512
        local_g = buf.local_gradients
        local_h = buf.local_hessians
        local_bins = buf.local_bins
        @inbounds for i in 1:n
            idx = group[i]
            local_g[i] = gradients[idx]
            local_h[i] = hessians[idx]
            local_bins[i] = num_bins[idx, j]
        end
        @inbounds for i in 1:n
            b = Int(local_bins[i]) + 1
            hist_g[li, b] += local_g[i]
            hist_h[li, b] += local_h[i]
            hist_c[li, b] += 1
            g_sum += local_g[i]
            h_sum += local_h[i]
        end
    else
        @inbounds for i in 1:n
            idx = group[i]
            b = Int(num_bins[idx, j]) + 1
            gi = gradients[idx]
            hi = hessians[idx]
            hist_g[li, b] += gi
            hist_h[li, b] += hi
            hist_c[li, b] += 1
            g_sum += gi
            h_sum += hi
        end
    end
    return g_sum, h_sum, n
end

function _fill_cat_leaf!(
    hist_g, hist_h, hist_c, li::Int, group, j::Int,
    gradients, hessians, cat_encoded, sorted_vals, buf,
)
    n = length(group)
    g_sum = 0.0
    h_sum = 0.0
    if n >= 512
        local_g = buf.local_gradients
        local_h = buf.local_hessians
        local_cat = buf.local_cat_values
        @inbounds for i in 1:n
            idx = group[i]
            local_g[i] = gradients[idx]
            local_h[i] = hessians[idx]
            local_cat[i] = cat_encoded[idx, j]
        end
        @inbounds for i in 1:n
            b = searchsortedfirst(sorted_vals, local_cat[i])
            hist_g[li, b] += local_g[i]
            hist_h[li, b] += local_h[i]
            hist_c[li, b] += 1
            g_sum += local_g[i]
            h_sum += local_h[i]
        end
    else
        @inbounds for i in 1:n
            idx = group[i]
            b = searchsortedfirst(sorted_vals, cat_encoded[idx, j])
            gi = gradients[idx]
            hi = hessians[idx]
            hist_g[li, b] += gi
            hist_h[li, b] += hi
            hist_c[li, b] += 1
            g_sum += gi
            h_sum += hi
        end
    end
    return g_sum, h_sum, n
end

function _fill_num_hist!(
    buf, hist_g, hist_h, hist_c, leaf_groups, j, gradients, hessians,
    num_bins, nb, has_parent::Bool, n_samples_level::Int,
)
    nb1 = nb + 1
    n_leaves = length(leaf_groups)

    # Break-even: without trick costs n_leaves*nb1 + n_samples; with trick costs
    # ~(3/2)*n_leaves*nb1 + n_samples/2. Trick wins when n_samples > n_leaves*nb1,
    # otherwise per-parent fixed overhead dominates and from-scratch is cheaper.
    use_subtraction = has_parent && n_leaves >= 2 && n_samples_level > n_leaves * nb1

    if use_subtraction
        n_parents = n_leaves ÷ 2
        scratch_g = buf.parent_hist_g_scratch
        scratch_h = buf.parent_hist_h_scratch
        scratch_c = buf.parent_hist_c_scratch
        total_g = buf.total_g
        total_h = buf.total_h
        total_n = buf.total_n
        @inbounds for p in n_parents:-1:1
            parent_g_sum = 0.0
            parent_h_sum = 0.0
            parent_n_sum = 0
            for b in 1:nb1
                gv = hist_g[p, b]
                hv = hist_h[p, b]
                cv = hist_c[p, b]
                scratch_g[b] = gv
                scratch_h[b] = hv
                scratch_c[b] = cv
                parent_g_sum += gv
                parent_h_sum += hv
                parent_n_sum += cv
            end

            left_li = 2p - 1
            right_li = 2p
            left_group = leaf_groups[left_li]
            right_group = leaf_groups[right_li]
            if length(left_group) <= length(right_group)
                smaller_li, larger_li = left_li, right_li
                smaller_group = left_group
            else
                smaller_li, larger_li = right_li, left_li
                smaller_group = right_group
            end

            for b in 1:nb1
                hist_g[smaller_li, b] = 0.0
                hist_h[smaller_li, b] = 0.0
                hist_c[smaller_li, b] = 0
            end

            g_small, h_small, n_small = _fill_num_leaf!(
                hist_g, hist_h, hist_c, smaller_li, smaller_group, j,
                gradients, hessians, num_bins, buf,
            )

            for b in 1:nb1
                hist_g[larger_li, b] = scratch_g[b] - hist_g[smaller_li, b]
                hist_h[larger_li, b] = scratch_h[b] - hist_h[smaller_li, b]
                hist_c[larger_li, b] = scratch_c[b] - hist_c[smaller_li, b]
            end

            total_g[smaller_li] = g_small
            total_h[smaller_li] = h_small
            total_n[smaller_li] = n_small
            total_g[larger_li] = parent_g_sum - g_small
            total_h[larger_li] = parent_h_sum - h_small
            total_n[larger_li] = parent_n_sum - n_small
        end
    else
        total_g = buf.total_g
        total_h = buf.total_h
        total_n = buf.total_n
        @inbounds for li in 1:n_leaves
            for b in 1:nb1
                hist_g[li, b] = 0.0
                hist_h[li, b] = 0.0
                hist_c[li, b] = 0
            end
            group = leaf_groups[li]
            g_sum, h_sum, n = _fill_num_leaf!(
                hist_g, hist_h, hist_c, li, group, j,
                gradients, hessians, num_bins, buf,
            )
            total_g[li] = g_sum
            total_h[li] = h_sum
            total_n[li] = n
        end
    end
    return nothing
end

function _fill_cat_hist!(
    buf, hist_g, hist_h, hist_c, leaf_groups, j, gradients, hessians,
    cat_encoded, sorted_vals, nv, has_parent::Bool, n_samples_level::Int,
)
    n_leaves = length(leaf_groups)
    use_subtraction = has_parent && n_leaves >= 2 && n_samples_level > n_leaves * nv

    if use_subtraction
        n_parents = n_leaves ÷ 2
        scratch_g = buf.parent_hist_g_scratch
        scratch_h = buf.parent_hist_h_scratch
        scratch_c = buf.parent_hist_c_scratch
        total_g = buf.total_g
        total_h = buf.total_h
        total_n = buf.total_n
        @inbounds for p in n_parents:-1:1
            parent_g_sum = 0.0
            parent_h_sum = 0.0
            parent_n_sum = 0
            for b in 1:nv
                gv = hist_g[p, b]
                hv = hist_h[p, b]
                cv = hist_c[p, b]
                scratch_g[b] = gv
                scratch_h[b] = hv
                scratch_c[b] = cv
                parent_g_sum += gv
                parent_h_sum += hv
                parent_n_sum += cv
            end

            left_li = 2p - 1
            right_li = 2p
            left_group = leaf_groups[left_li]
            right_group = leaf_groups[right_li]
            if length(left_group) <= length(right_group)
                smaller_li, larger_li = left_li, right_li
                smaller_group = left_group
            else
                smaller_li, larger_li = right_li, left_li
                smaller_group = right_group
            end

            for b in 1:nv
                hist_g[smaller_li, b] = 0.0
                hist_h[smaller_li, b] = 0.0
                hist_c[smaller_li, b] = 0
            end

            g_small, h_small, n_small = _fill_cat_leaf!(
                hist_g, hist_h, hist_c, smaller_li, smaller_group, j,
                gradients, hessians, cat_encoded, sorted_vals, buf,
            )

            for b in 1:nv
                hist_g[larger_li, b] = scratch_g[b] - hist_g[smaller_li, b]
                hist_h[larger_li, b] = scratch_h[b] - hist_h[smaller_li, b]
                hist_c[larger_li, b] = scratch_c[b] - hist_c[smaller_li, b]
            end

            total_g[smaller_li] = g_small
            total_h[smaller_li] = h_small
            total_n[smaller_li] = n_small
            total_g[larger_li] = parent_g_sum - g_small
            total_h[larger_li] = parent_h_sum - h_small
            total_n[larger_li] = parent_n_sum - n_small
        end
    else
        total_g = buf.total_g
        total_h = buf.total_h
        total_n = buf.total_n
        @inbounds for li in 1:n_leaves
            for b in 1:nv
                hist_g[li, b] = 0.0
                hist_h[li, b] = 0.0
                hist_c[li, b] = 0
            end
            group = leaf_groups[li]
            g_sum, h_sum, n = _fill_cat_leaf!(
                hist_g, hist_h, hist_c, li, group, j,
                gradients, hessians, cat_encoded, sorted_vals, buf,
            )
            total_g[li] = g_sum
            total_h[li] = h_sum
            total_n[li] = n
        end
    end
    return nothing
end

function _sweep_gain(
    hist_g, hist_h, hist_c, total_g, total_h, total_n,
    left_g, left_h, left_c, n_leaves, nb_or_nv, l2_leaf_reg, min_data_in_leaf,
)
    best_gain = -Inf
    best_b = -1
    @inbounds for li in 1:n_leaves
        left_g[li] = hist_g[li, 1]
        left_h[li] = hist_h[li, 1]
        left_c[li] = hist_c[li, 1]
    end
    @inbounds for b in 2:nb_or_nv
        @inbounds for li in 1:n_leaves
            left_g[li] += hist_g[li, b]
            left_h[li] += hist_h[li, b]
            left_c[li] += hist_c[li, b]
        end
        gain = 0.0
        @inbounds for li in 1:n_leaves
            total_n[li] == 0 && continue
            lc = left_c[li]
            rc = total_n[li] - lc
            (lc < min_data_in_leaf || rc < min_data_in_leaf) && continue
            rg = total_g[li] - left_g[li]
            rh = total_h[li] - left_h[li]
            gain +=
                left_g[li]^2 / (left_h[li] + l2_leaf_reg) +
                rg^2 / (rh + l2_leaf_reg) -
                total_g[li]^2 / (total_h[li] + l2_leaf_reg)
        end
        if gain > best_gain
            best_gain = gain
            best_b = b
        end
    end
    return best_gain, best_b
end

function _find_best_split_across_leaves(
    gradients,
    hessians,
    num_bins,
    cat_encoded,
    leaf_groups,
    sampled_num::AbstractVector{Int},
    sampled_cat::AbstractVector{Int},
    qf,
    bufs::Vector{SplitBuffers},
    cat_sorted_vals::Vector{Vector{Float64}},
    cache::HistCache;
    l2_leaf_reg=3.0,
    min_data_in_leaf=1,
)
    n_leaves = length(leaf_groups)
    thread_bests = fill(SplitCandidate(0, false, 0.0, -Inf), Threads.maxthreadid())

    n_samples_level = 0
    @inbounds for li in 1:n_leaves
        n_samples_level += length(leaf_groups[li])
    end

    Threads.@threads :static for j in sampled_num
        nb = qf.n_bins[j]
        nb <= 1 && continue
        tid = Threads.threadid()
        buf = bufs[tid]
        hist_g = cache.num_hist_g[j]
        hist_h = cache.num_hist_h[j]
        hist_c = cache.num_hist_c[j]
        has_parent = cache.num_hist_valid[j]
        _fill_num_hist!(
            buf, hist_g, hist_h, hist_c, leaf_groups, j,
            gradients, hessians, num_bins, nb, has_parent, n_samples_level,
        )
        cache.num_hist_filled[j] = true
        gain, b = _sweep_gain(
            hist_g, hist_h, hist_c, buf.total_g, buf.total_h, buf.total_n,
            buf.left_g, buf.left_h, buf.left_c,
            n_leaves, nb + 1, l2_leaf_reg, min_data_in_leaf,
        )
        if gain > thread_bests[tid].gain
            thread_bests[tid] = SplitCandidate(j, false, Float64(b - 1), gain)
        end
    end

    Threads.@threads :static for j in sampled_cat
        if isempty(cat_sorted_vals)
            sorted_vals_local = _collect_cat_vals(leaf_groups, cat_encoded, j)
            nv = length(sorted_vals_local)
            nv <= 1 && continue
            tid = Threads.threadid()
            buf = bufs[tid]
            hist_g = cache.cat_hist_g[j]
            hist_h = cache.cat_hist_h[j]
            hist_c = cache.cat_hist_c[j]
            _fill_cat_hist!(
                buf, hist_g, hist_h, hist_c, leaf_groups, j,
                gradients, hessians, cat_encoded, sorted_vals_local, nv, false, n_samples_level,
            )
            cache.cat_hist_filled[j] = false
            gain, b = _sweep_gain(
                hist_g, hist_h, hist_c, buf.total_g, buf.total_h, buf.total_n,
                buf.left_g, buf.left_h, buf.left_c,
                n_leaves, nv, l2_leaf_reg, min_data_in_leaf,
            )
            if gain > thread_bests[tid].gain && b > 0
                threshold = (sorted_vals_local[b - 1] + sorted_vals_local[b]) / 2.0
                thread_bests[tid] = SplitCandidate(j, true, threshold, gain)
            end
        else
            sorted_vals = cat_sorted_vals[j]
            nv = length(sorted_vals)
            nv <= 1 && continue
            tid = Threads.threadid()
            buf = bufs[tid]
            hist_g = cache.cat_hist_g[j]
            hist_h = cache.cat_hist_h[j]
            hist_c = cache.cat_hist_c[j]
            has_parent = cache.cat_hist_valid[j]
            _fill_cat_hist!(
                buf, hist_g, hist_h, hist_c, leaf_groups, j,
                gradients, hessians, cat_encoded, sorted_vals, nv, has_parent, n_samples_level,
            )
            cache.cat_hist_filled[j] = true
            gain, b = _sweep_gain(
                hist_g, hist_h, hist_c, buf.total_g, buf.total_h, buf.total_n,
                buf.left_g, buf.left_h, buf.left_c,
                n_leaves, nv, l2_leaf_reg, min_data_in_leaf,
            )
            if gain > thread_bests[tid].gain && b > 0
                threshold = (sorted_vals[b - 1] + sorted_vals[b]) / 2.0
                thread_bests[tid] = SplitCandidate(j, true, threshold, gain)
            end
        end
    end

    return argmax(c -> c.gain, thread_bests)
end

function _collect_cat_vals(leaf_groups, cat_encoded, j)
    all_vals = Set{Float64}()
    for group in leaf_groups, idx in group
        push!(all_vals, cat_encoded[idx, j])
    end
    return sort(collect(all_vals))
end
