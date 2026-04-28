"""
Fill row `li` of a multiclass numerical histogram and accumulate per-class
totals into `total_g[:, li]` / `total_h[:, li]` (zeroed inside). Returns `n`.
Helper form provides a function barrier so Julia specializes on `group`'s
concrete type — inlining the body into the caller would leave the hot loop
iterating over `Any`.
"""
function _fill_num_leaf_mc!(
    hist_g, hist_h, hist_c, total_g, total_h, li::Int, group, j::Int,
    gradients, hessians, num_bins, buf, n_classes::Int,
)
    @inbounds for c in 1:n_classes
        total_g[c, li] = 0.0
        total_h[c, li] = 0.0
    end
    n = length(group)
    if n >= 512
        local_bins = buf.local_bins
        local_g = buf.local_gradients_mc
        local_h = buf.local_hessians_mc
        @inbounds for i in 1:n
            idx = group[i]
            local_bins[i] = num_bins[idx, j]
            for c in 1:n_classes
                local_g[c, i] = gradients[idx, c]
                local_h[c, i] = hessians[idx, c]
            end
        end
        @inbounds for i in 1:n
            b = Int(local_bins[i]) + 1
            hist_c[li, b] += 1
            for c in 1:n_classes
                gv = local_g[c, i]
                hv = local_h[c, i]
                hist_g[c, li, b] += gv
                hist_h[c, li, b] += hv
                total_g[c, li] += gv
                total_h[c, li] += hv
            end
        end
    else
        @inbounds for i in 1:n
            idx = group[i]
            b = Int(num_bins[idx, j]) + 1
            hist_c[li, b] += 1
            for c in 1:n_classes
                gv = gradients[idx, c]
                hv = hessians[idx, c]
                hist_g[c, li, b] += gv
                hist_h[c, li, b] += hv
                total_g[c, li] += gv
                total_h[c, li] += hv
            end
        end
    end
    return n
end

function _fill_cat_leaf_mc!(
    hist_g, hist_h, hist_c, total_g, total_h, li::Int, group, j::Int,
    gradients, hessians, cat_encoded, sorted_vals, buf, n_classes::Int,
)
    @inbounds for c in 1:n_classes
        total_g[c, li] = 0.0
        total_h[c, li] = 0.0
    end
    n = length(group)
    if n >= 512
        local_cat = buf.local_cat_values
        local_g = buf.local_gradients_mc
        local_h = buf.local_hessians_mc
        @inbounds for i in 1:n
            idx = group[i]
            local_cat[i] = cat_encoded[idx, j]
            for c in 1:n_classes
                local_g[c, i] = gradients[idx, c]
                local_h[c, i] = hessians[idx, c]
            end
        end
        @inbounds for i in 1:n
            b = searchsortedfirst(sorted_vals, local_cat[i])
            hist_c[li, b] += 1
            for c in 1:n_classes
                gv = local_g[c, i]
                hv = local_h[c, i]
                hist_g[c, li, b] += gv
                hist_h[c, li, b] += hv
                total_g[c, li] += gv
                total_h[c, li] += hv
            end
        end
    else
        @inbounds for i in 1:n
            idx = group[i]
            b = searchsortedfirst(sorted_vals, cat_encoded[idx, j])
            hist_c[li, b] += 1
            for c in 1:n_classes
                gv = gradients[idx, c]
                hv = hessians[idx, c]
                hist_g[c, li, b] += gv
                hist_h[c, li, b] += hv
                total_g[c, li] += gv
                total_h[c, li] += hv
            end
        end
    end
    return n
end

function _fill_num_hist_mc!(
    buf, hist_g, hist_h, hist_c, leaf_groups, j,
    gradients, hessians, num_bins, nb, n_classes, has_parent::Bool, n_samples_level::Int,
)
    nb1 = nb + 1
    n_leaves = length(leaf_groups)
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
            parent_n_sum = 0
            for b in 1:nb1
                cv = hist_c[p, b]
                scratch_c[b] = cv
                parent_n_sum += cv
                for c in 1:n_classes
                    scratch_g[c, b] = hist_g[c, p, b]
                    scratch_h[c, b] = hist_h[c, p, b]
                end
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
                hist_c[smaller_li, b] = 0
                for c in 1:n_classes
                    hist_g[c, smaller_li, b] = 0.0
                    hist_h[c, smaller_li, b] = 0.0
                end
            end

            n_small = _fill_num_leaf_mc!(
                hist_g, hist_h, hist_c, total_g, total_h, smaller_li, smaller_group, j,
                gradients, hessians, num_bins, buf, n_classes,
            )

            total_n[smaller_li] = n_small
            total_n[larger_li] = parent_n_sum - n_small
            for c in 1:n_classes
                parent_g_c = 0.0
                parent_h_c = 0.0
                for b in 1:nb1
                    pg = scratch_g[c, b]
                    ph = scratch_h[c, b]
                    sg = hist_g[c, smaller_li, b]
                    sh = hist_h[c, smaller_li, b]
                    hist_g[c, larger_li, b] = pg - sg
                    hist_h[c, larger_li, b] = ph - sh
                    parent_g_c += pg
                    parent_h_c += ph
                end
                total_g[c, larger_li] = parent_g_c - total_g[c, smaller_li]
                total_h[c, larger_li] = parent_h_c - total_h[c, smaller_li]
            end
            for b in 1:nb1
                hist_c[larger_li, b] = scratch_c[b] - hist_c[smaller_li, b]
            end
        end
    else
        total_g = buf.total_g
        total_h = buf.total_h
        total_n = buf.total_n
        @inbounds for li in 1:n_leaves
            for b in 1:nb1
                hist_c[li, b] = 0
                for c in 1:n_classes
                    hist_g[c, li, b] = 0.0
                    hist_h[c, li, b] = 0.0
                end
            end
            group = leaf_groups[li]
            n = _fill_num_leaf_mc!(
                hist_g, hist_h, hist_c, total_g, total_h, li, group, j,
                gradients, hessians, num_bins, buf, n_classes,
            )
            total_n[li] = n
        end
    end
    return nothing
end

function _fill_cat_hist_mc!(
    buf, hist_g, hist_h, hist_c, leaf_groups, j,
    gradients, hessians, cat_encoded, sorted_vals, nv, n_classes, has_parent::Bool,
    n_samples_level::Int,
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
            parent_n_sum = 0
            for b in 1:nv
                cv = hist_c[p, b]
                scratch_c[b] = cv
                parent_n_sum += cv
                for c in 1:n_classes
                    scratch_g[c, b] = hist_g[c, p, b]
                    scratch_h[c, b] = hist_h[c, p, b]
                end
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
                hist_c[smaller_li, b] = 0
                for c in 1:n_classes
                    hist_g[c, smaller_li, b] = 0.0
                    hist_h[c, smaller_li, b] = 0.0
                end
            end

            n_small = _fill_cat_leaf_mc!(
                hist_g, hist_h, hist_c, total_g, total_h, smaller_li, smaller_group, j,
                gradients, hessians, cat_encoded, sorted_vals, buf, n_classes,
            )

            total_n[smaller_li] = n_small
            total_n[larger_li] = parent_n_sum - n_small
            for c in 1:n_classes
                parent_g_c = 0.0
                parent_h_c = 0.0
                for b in 1:nv
                    pg = scratch_g[c, b]
                    ph = scratch_h[c, b]
                    sg = hist_g[c, smaller_li, b]
                    sh = hist_h[c, smaller_li, b]
                    hist_g[c, larger_li, b] = pg - sg
                    hist_h[c, larger_li, b] = ph - sh
                    parent_g_c += pg
                    parent_h_c += ph
                end
                total_g[c, larger_li] = parent_g_c - total_g[c, smaller_li]
                total_h[c, larger_li] = parent_h_c - total_h[c, smaller_li]
            end
            for b in 1:nv
                hist_c[larger_li, b] = scratch_c[b] - hist_c[smaller_li, b]
            end
        end
    else
        total_g = buf.total_g
        total_h = buf.total_h
        total_n = buf.total_n
        @inbounds for li in 1:n_leaves
            for b in 1:nv
                hist_c[li, b] = 0
                for c in 1:n_classes
                    hist_g[c, li, b] = 0.0
                    hist_h[c, li, b] = 0.0
                end
            end
            group = leaf_groups[li]
            n = _fill_cat_leaf_mc!(
                hist_g, hist_h, hist_c, total_g, total_h, li, group, j,
                gradients, hessians, cat_encoded, sorted_vals, buf, n_classes,
            )
            total_n[li] = n
        end
    end
    return nothing
end

function _sweep_gain_mc(
    hist_g, hist_h, hist_c, total_g, total_h, total_n,
    left_g, left_h, left_c, total_score,
    n_leaves, nb_or_nv, l2_leaf_reg, min_data_in_leaf, n_classes,
)
    best_gain = -Inf
    best_b = -1

    # The third gain term `total_g²/(total_h + λ)` depends only on (c, li), so
    # lift it out of the b-sweep — at nb=256, leaves=64, classes=7 the inner
    # version recomputes ~115k divisions per feature per level.
    @inbounds for li in 1:n_leaves
        if total_n[li] == 0
            for c in 1:n_classes
                total_score[c, li] = 0.0
            end
        else
            for c in 1:n_classes
                total_score[c, li] =
                    total_g[c, li]^2 / (total_h[c, li] + l2_leaf_reg)
            end
        end
    end

    @inbounds for li in 1:n_leaves
        left_c[li] = hist_c[li, 1]
        for c in 1:n_classes
            left_g[c, li] = hist_g[c, li, 1]
            left_h[c, li] = hist_h[c, li, 1]
        end
    end
    @inbounds for b in 2:nb_or_nv
        @inbounds for li in 1:n_leaves
            left_c[li] += hist_c[li, b]
            for c in 1:n_classes
                left_g[c, li] += hist_g[c, li, b]
                left_h[c, li] += hist_h[c, li, b]
            end
        end
        gain = 0.0
        @inbounds for li in 1:n_leaves
            total_n[li] == 0 && continue
            lc = left_c[li]
            rc = total_n[li] - lc
            (lc < min_data_in_leaf || rc < min_data_in_leaf) && continue
            for c in 1:n_classes
                rg = total_g[c, li] - left_g[c, li]
                rh = total_h[c, li] - left_h[c, li]
                gain +=
                    left_g[c, li]^2 / (left_h[c, li] + l2_leaf_reg) +
                    rg^2 / (rh + l2_leaf_reg) -
                    total_score[c, li]
            end
        end
        if gain > best_gain
            best_gain = gain
            best_b = b
        end
    end
    return best_gain, best_b
end

function _find_best_split_across_leaves_mc(
    gradients,
    hessians,
    num_bins,
    cat_encoded,
    leaf_groups,
    sampled_num::AbstractVector{Int},
    sampled_cat::AbstractVector{Int},
    qf,
    n_classes,
    bufs::Vector{SplitBuffersMC},
    cat_sorted_vals::Vector{Vector{Float64}},
    cache::HistCacheMC;
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
        _fill_num_hist_mc!(
            buf, hist_g, hist_h, hist_c, leaf_groups, j,
            gradients, hessians, num_bins, nb, n_classes, has_parent, n_samples_level,
        )
        cache.num_hist_filled[j] = true
        gain, b = _sweep_gain_mc(
            hist_g, hist_h, hist_c, buf.total_g, buf.total_h, buf.total_n,
            buf.left_g, buf.left_h, buf.left_c, buf.total_score,
            n_leaves, nb + 1, l2_leaf_reg, min_data_in_leaf, n_classes,
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
            _fill_cat_hist_mc!(
                buf, hist_g, hist_h, hist_c, leaf_groups, j,
                gradients, hessians, cat_encoded, sorted_vals_local, nv, n_classes, false,
                n_samples_level,
            )
            cache.cat_hist_filled[j] = false
            gain, b = _sweep_gain_mc(
                hist_g, hist_h, hist_c, buf.total_g, buf.total_h, buf.total_n,
                buf.left_g, buf.left_h, buf.left_c, buf.total_score,
                n_leaves, nv, l2_leaf_reg, min_data_in_leaf, n_classes,
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
            _fill_cat_hist_mc!(
                buf, hist_g, hist_h, hist_c, leaf_groups, j,
                gradients, hessians, cat_encoded, sorted_vals, nv, n_classes, has_parent,
                n_samples_level,
            )
            cache.cat_hist_filled[j] = true
            gain, b = _sweep_gain_mc(
                hist_g, hist_h, hist_c, buf.total_g, buf.total_h, buf.total_n,
                buf.left_g, buf.left_h, buf.left_c, buf.total_score,
                n_leaves, nv, l2_leaf_reg, min_data_in_leaf, n_classes,
            )
            if gain > thread_bests[tid].gain && b > 0
                threshold = (sorted_vals[b - 1] + sorted_vals[b]) / 2.0
                thread_bests[tid] = SplitCandidate(j, true, threshold, gain)
            end
        end
    end

    return argmax(c -> c.gain, thread_bests)
end
