function _fill_num_hist_mc!(buf, li, group, j, gradients, hessians, num_bins, nb, n_classes)
    nb1 = nb + 1
    @inbounds for b in 1:nb1
        buf.hist_c[li, b] = 0
        for c in 1:n_classes
            buf.hist_g[li, b, c] = 0.0
            buf.hist_h[li, b, c] = 0.0
        end
    end
    @inbounds for c in 1:n_classes
        buf.total_g[li, c] = 0.0
        buf.total_h[li, c] = 0.0
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
                local_g[i, c] = gradients[idx, c]
                local_h[i, c] = hessians[idx, c]
            end
        end
        @inbounds for i in 1:n
            b = Int(local_bins[i]) + 1
            buf.hist_c[li, b] += 1
            for c in 1:n_classes
                buf.hist_g[li, b, c] += local_g[i, c]
                buf.hist_h[li, b, c] += local_h[i, c]
                buf.total_g[li, c] += local_g[i, c]
                buf.total_h[li, c] += local_h[i, c]
            end
        end
    else
        @inbounds for idx in group
            b = Int(num_bins[idx, j]) + 1
            buf.hist_c[li, b] += 1
            for c in 1:n_classes
                buf.hist_g[li, b, c] += gradients[idx, c]
                buf.hist_h[li, b, c] += hessians[idx, c]
                buf.total_g[li, c] += gradients[idx, c]
                buf.total_h[li, c] += hessians[idx, c]
            end
        end
    end
    buf.total_n[li] = n
end

function _fill_cat_hist_mc!(
    buf, li, group, j, gradients, hessians, cat_encoded, sorted_vals, nv, n_classes
)
    @inbounds for b in 1:nv
        buf.hist_c[li, b] = 0
        for c in 1:n_classes
            buf.hist_g[li, b, c] = 0.0
            buf.hist_h[li, b, c] = 0.0
        end
    end
    @inbounds for c in 1:n_classes
        buf.total_g[li, c] = 0.0
        buf.total_h[li, c] = 0.0
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
                local_g[i, c] = gradients[idx, c]
                local_h[i, c] = hessians[idx, c]
            end
        end
        @inbounds for i in 1:n
            b = searchsortedfirst(sorted_vals, local_cat[i])
            buf.hist_c[li, b] += 1
            for c in 1:n_classes
                buf.hist_g[li, b, c] += local_g[i, c]
                buf.hist_h[li, b, c] += local_h[i, c]
                buf.total_g[li, c] += local_g[i, c]
                buf.total_h[li, c] += local_h[i, c]
            end
        end
    else
        @inbounds for idx in group
            b = searchsortedfirst(sorted_vals, cat_encoded[idx, j])
            buf.hist_c[li, b] += 1
            for c in 1:n_classes
                buf.hist_g[li, b, c] += gradients[idx, c]
                buf.hist_h[li, b, c] += hessians[idx, c]
                buf.total_g[li, c] += gradients[idx, c]
                buf.total_h[li, c] += hessians[idx, c]
            end
        end
    end
    buf.total_n[li] = n
end

function _sweep_gain_mc(buf, n_leaves, nb_or_nv, l2_leaf_reg, min_data_in_leaf, n_classes)
    best_gain = -Inf
    best_b = -1
    @inbounds for li in 1:n_leaves
        buf.left_c[li] = buf.hist_c[li, 1]
        @inbounds for c in 1:n_classes
            buf.left_g[li, c] = buf.hist_g[li, 1, c]
            buf.left_h[li, c] = buf.hist_h[li, 1, c]
        end
    end
    @inbounds for b in 2:nb_or_nv
        @inbounds for li in 1:n_leaves
            buf.left_c[li] += buf.hist_c[li, b]
            @inbounds for c in 1:n_classes
                buf.left_g[li, c] += buf.hist_g[li, b, c]
                buf.left_h[li, c] += buf.hist_h[li, b, c]
            end
        end
        gain = 0.0
        @inbounds for li in 1:n_leaves
            buf.total_n[li] == 0 && continue
            lc = buf.left_c[li]
            rc = buf.total_n[li] - lc
            (lc < min_data_in_leaf || rc < min_data_in_leaf) && continue
            @inbounds for c in 1:n_classes
                rg = buf.total_g[li, c] - buf.left_g[li, c]
                rh = buf.total_h[li, c] - buf.left_h[li, c]
                gain +=
                    buf.left_g[li, c]^2 / (buf.left_h[li, c] + l2_leaf_reg) +
                    rg^2 / (rh + l2_leaf_reg) -
                    buf.total_g[li, c]^2 / (buf.total_h[li, c] + l2_leaf_reg)
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
    cat_sorted_vals::Vector{Vector{Float64}};
    l2_leaf_reg=3.0,
    min_data_in_leaf=1,
)
    n_leaves = length(leaf_groups)
    thread_bests = fill(SplitCandidate(0, :numerical, 0.0, -Inf), Threads.maxthreadid())

    Threads.@threads :static for j in sampled_num
        nb = qf.n_bins[j]
        nb <= 1 && continue
        tid = Threads.threadid()
        buf = bufs[tid]
        for (li, group) in enumerate(leaf_groups)
            _fill_num_hist_mc!(buf, li, group, j, gradients, hessians, num_bins, nb, n_classes)
        end
        gain, b = _sweep_gain_mc(buf, n_leaves, nb + 1, l2_leaf_reg, min_data_in_leaf, n_classes)
        if gain > thread_bests[tid].gain
            thread_bests[tid] = SplitCandidate(j, :numerical, Float64(b - 1), gain)
        end
    end

    Threads.@threads :static for j in sampled_cat
        sorted_vals = if isempty(cat_sorted_vals)
            all_vals = Set{Float64}()
            for group in leaf_groups, idx in group
                push!(all_vals, cat_encoded[idx, j])
            end
            sort(collect(all_vals))
        else
            cat_sorted_vals[j]
        end
        nv = length(sorted_vals)
        nv <= 1 && continue
        tid = Threads.threadid()
        buf = bufs[tid]
        for (li, group) in enumerate(leaf_groups)
            _fill_cat_hist_mc!(
                buf, li, group, j, gradients, hessians, cat_encoded, sorted_vals, nv, n_classes
            )
        end
        gain, b = _sweep_gain_mc(buf, n_leaves, nv, l2_leaf_reg, min_data_in_leaf, n_classes)
        if gain > thread_bests[tid].gain && b > 0
            threshold = (sorted_vals[b - 1] + sorted_vals[b]) / 2.0
            thread_bests[tid] = SplitCandidate(j, :categorical, threshold, gain)
        end
    end

    return argmax(c -> c.gain, thread_bests)
end
