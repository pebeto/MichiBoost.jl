function build_symmetric_tree(gradients::AbstractVector{Float64},
                              hessians::AbstractVector{Float64},
                              num_bins::AbstractMatrix{UInt16},
                              cat_encoded::AbstractMatrix{Float64},
                              sample_indices::Vector{Int},
                              depth::Int,
                              n_num::Int,
                              n_cat::Int,
                              qf::QuantizedFeatures;
                              l2_leaf_reg::Float64=3.0,
                              min_data_in_leaf::Int=1)
    split_features, split_types, split_thresholds = Int[], Symbol[], Float64[]
    leaf_groups = [sample_indices]

    for _ in 1:depth
        best = _find_best_split_across_leaves(
            gradients, hessians, num_bins, cat_encoded,
            leaf_groups, n_num, n_cat, qf; l2_leaf_reg, min_data_in_leaf)
        leaf_groups = _apply_split!(split_features, split_types, split_thresholds,
                                    leaf_groups, best, num_bins, cat_encoded, n_num)
    end

    n_leaves = 1 << depth
    leaf_values = Vector{Float64}(undef, n_leaves)
    for l in 1:n_leaves
        group = leaf_groups[l]
        if isempty(group)
            leaf_values[l] = 0.0
        else
            g_sum, h_sum = 0.0, 0.0
            for idx in group; g_sum += gradients[idx]; h_sum += hessians[idx]; end
            leaf_values[l] = g_sum / (h_sum + l2_leaf_reg)
        end
    end

    return SymmetricTree(depth, split_features, split_types, split_thresholds, leaf_values)
end

function build_symmetric_tree_multiclass(gradients::AbstractMatrix{Float64},
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
                                         min_data_in_leaf::Int=1)
    split_features, split_types, split_thresholds = Int[], Symbol[], Float64[]
    leaf_groups = [sample_indices]

    for _ in 1:depth
        best = _find_best_split_across_leaves_mc(
            gradients, hessians, num_bins, cat_encoded,
            leaf_groups, n_num, n_cat, qf, n_classes; l2_leaf_reg, min_data_in_leaf)
        leaf_groups = _apply_split!(split_features, split_types, split_thresholds,
                                    leaf_groups, best, num_bins, cat_encoded, n_num)
    end

    n_leaves = 1 << depth
    leaf_values = Matrix{Float64}(undef, n_leaves, n_classes)
    for l in 1:n_leaves
        group = leaf_groups[l]
        if isempty(group)
            leaf_values[l, :] .= 0.0
        else
            for c in 1:n_classes
                g_sum, h_sum = 0.0, 0.0
                for idx in group; g_sum += gradients[idx, c]; h_sum += hessians[idx, c]; end
                leaf_values[l, c] = g_sum / (h_sum + l2_leaf_reg)
            end
        end
    end

    return SymmetricTreeMultiClass(depth, split_features, split_types, split_thresholds,
                                   leaf_values)
end

function _apply_split!(split_features, split_types, split_thresholds,
                       leaf_groups, best::SplitCandidate, num_bins, cat_encoded, n_num)
    if best.feature_index == 0
        push!(split_features, 1)
        push!(split_types, n_num > 0 ? :numerical : :categorical)
        push!(split_thresholds, 0.0)
        new_groups = Vector{Int}[]
        for group in leaf_groups
            push!(new_groups, group); push!(new_groups, Int[])
        end
        return new_groups
    end

    push!(split_features, best.feature_index)
    push!(split_types, best.feature_type)
    push!(split_thresholds, best.threshold)

    new_groups = Vector{Int}[]
    for group in leaf_groups
        left, right = Int[], Int[]
        for idx in group
            if _goes_right(best, num_bins, cat_encoded, idx)
                push!(right, idx)
            else
                push!(left, idx)
            end
        end
        push!(new_groups, left); push!(new_groups, right)
    end
    return new_groups
end

@inline function _goes_right(split::SplitCandidate, num_bins, cat_encoded, idx)
    return split.feature_type == :numerical ?
        num_bins[idx, split.feature_index] > UInt16(split.threshold) :
        cat_encoded[idx, split.feature_index] > split.threshold
end

function _find_best_split_across_leaves(gradients, hessians, num_bins, cat_encoded,
                                         leaf_groups, n_num, n_cat, qf;
                                         l2_leaf_reg=3.0, min_data_in_leaf=1)
    best = SplitCandidate(0, :numerical, 0.0, -Inf)

    # Numerical features — build per-leaf histograms, then sweep
    for j in 1:n_num
        nb = qf.n_bins[j]
        nb <= 1 && continue

        # Build one histogram per leaf
        leaf_hists_g = [zeros(Float64, nb + 1) for _ in leaf_groups]
        leaf_hists_h = [zeros(Float64, nb + 1) for _ in leaf_groups]
        leaf_hists_c = [zeros(Int, nb + 1) for _ in leaf_groups]
        leaf_total_g = zeros(Float64, length(leaf_groups))
        leaf_total_h = zeros(Float64, length(leaf_groups))
        leaf_total_n = zeros(Int, length(leaf_groups))

        for (li, group) in enumerate(leaf_groups)
            for idx in group
                b = Int(num_bins[idx, j]) + 1
                leaf_hists_g[li][b] += gradients[idx]
                leaf_hists_h[li][b] += hessians[idx]
                leaf_hists_c[li][b] += 1
                leaf_total_g[li] += gradients[idx]
                leaf_total_h[li] += hessians[idx]
                leaf_total_n[li] += 1
            end
        end

        # Sweep each bin threshold, summing gain across all leaves
        # Initialize left accumulators with NaN bin (index 1)
        left_g = [leaf_hists_g[li][1] for li in eachindex(leaf_groups)]
        left_h = [leaf_hists_h[li][1] for li in eachindex(leaf_groups)]
        left_c = [leaf_hists_c[li][1] for li in eachindex(leaf_groups)]

        for b in 2:(nb + 1)
            for li in eachindex(leaf_groups)
                left_g[li] += leaf_hists_g[li][b]
                left_h[li] += leaf_hists_h[li][b]
                left_c[li] += leaf_hists_c[li][b]
            end

            gain = 0.0
            valid = true
            for li in eachindex(leaf_groups)
                lc = left_c[li]
                rc = leaf_total_n[li] - lc
                # Skip leaves that are empty or too small to split
                leaf_total_n[li] == 0 && continue
                if lc < min_data_in_leaf || rc < min_data_in_leaf
                    # This leaf can't be split at this threshold, but others might
                    # Just add 0 gain for this leaf (no improvement)
                    continue
                end
                rg = leaf_total_g[li] - left_g[li]
                rh = leaf_total_h[li] - left_h[li]
                gain += left_g[li]^2 / (left_h[li] + l2_leaf_reg) +
                        rg^2 / (rh + l2_leaf_reg) -
                        leaf_total_g[li]^2 / (leaf_total_h[li] + l2_leaf_reg)
            end

            gain > best.gain && (best = SplitCandidate(j, :numerical, Float64(b - 1), gain))
        end
    end

    # Categorical features — collect all unique values, sweep
    for j in 1:n_cat
        all_vals = Set{Float64}()
        for group in leaf_groups, idx in group
            push!(all_vals, cat_encoded[idx, j])
        end
        sorted_vals = sort(collect(all_vals))
        length(sorted_vals) <= 1 && continue

        for s in 1:(length(sorted_vals) - 1)
            threshold = (sorted_vals[s] + sorted_vals[s + 1]) / 2.0
            gain = 0.0
            for group in leaf_groups
                isempty(group) && continue
                lg, lh, lc = 0.0, 0.0, 0
                tg, th = 0.0, 0.0
                for idx in group
                    tg += gradients[idx]; th += hessians[idx]
                    if cat_encoded[idx, j] <= threshold
                        lg += gradients[idx]; lh += hessians[idx]; lc += 1
                    end
                end
                rc = length(group) - lc
                (lc < min_data_in_leaf || rc < min_data_in_leaf) && continue
                rg, rh = tg - lg, th - lh
                gain += lg^2 / (lh + l2_leaf_reg) + rg^2 / (rh + l2_leaf_reg) -
                        tg^2 / (th + l2_leaf_reg)
            end
            gain > best.gain && (best = SplitCandidate(j, :categorical, threshold, gain))
        end
    end

    return best
end

function _find_best_split_across_leaves_mc(gradients, hessians, num_bins, cat_encoded,
                                            leaf_groups, n_num, n_cat, qf, n_classes;
                                            l2_leaf_reg=3.0, min_data_in_leaf=1)
    best = SplitCandidate(0, :numerical, 0.0, -Inf)

    for j in 1:n_num
        nb = qf.n_bins[j]
        nb <= 1 && continue

        n_leaves = length(leaf_groups)
        leaf_hists_g = [zeros(Float64, nb + 1, n_classes) for _ in 1:n_leaves]
        leaf_hists_h = [zeros(Float64, nb + 1, n_classes) for _ in 1:n_leaves]
        leaf_hists_c = [zeros(Int, nb + 1) for _ in 1:n_leaves]
        leaf_total_g = [zeros(Float64, n_classes) for _ in 1:n_leaves]
        leaf_total_h = [zeros(Float64, n_classes) for _ in 1:n_leaves]
        leaf_total_n = zeros(Int, n_leaves)

        for (li, group) in enumerate(leaf_groups)
            for idx in group
                b = Int(num_bins[idx, j]) + 1
                leaf_hists_c[li][b] += 1
                leaf_total_n[li] += 1
                for c in 1:n_classes
                    leaf_hists_g[li][b, c] += gradients[idx, c]
                    leaf_hists_h[li][b, c] += hessians[idx, c]
                    leaf_total_g[li][c] += gradients[idx, c]
                    leaf_total_h[li][c] += hessians[idx, c]
                end
            end
        end

        left_g = [leaf_hists_g[li][1, :] for li in 1:n_leaves]
        left_h = [leaf_hists_h[li][1, :] for li in 1:n_leaves]
        left_c = [leaf_hists_c[li][1] for li in 1:n_leaves]

        for b in 2:(nb + 1)
            for li in 1:n_leaves
                left_c[li] += leaf_hists_c[li][b]
                for c in 1:n_classes
                    left_g[li][c] += leaf_hists_g[li][b, c]
                    left_h[li][c] += leaf_hists_h[li][b, c]
                end
            end

            gain = 0.0
            for li in 1:n_leaves
                leaf_total_n[li] == 0 && continue
                lc = left_c[li]; rc = leaf_total_n[li] - lc
                (lc < min_data_in_leaf || rc < min_data_in_leaf) && continue
                for c in 1:n_classes
                    rg = leaf_total_g[li][c] - left_g[li][c]
                    rh = leaf_total_h[li][c] - left_h[li][c]
                    gain += left_g[li][c]^2 / (left_h[li][c] + l2_leaf_reg) +
                            rg^2 / (rh + l2_leaf_reg) -
                            leaf_total_g[li][c]^2 / (leaf_total_h[li][c] + l2_leaf_reg)
                end
            end
            gain > best.gain && (best = SplitCandidate(j, :numerical, Float64(b - 1), gain))
        end
    end

    # Categorical features
    for j in 1:n_cat
        all_vals = Set{Float64}()
        for group in leaf_groups, idx in group; push!(all_vals, cat_encoded[idx, j]); end
        sorted_vals = sort(collect(all_vals))
        length(sorted_vals) <= 1 && continue

        for s in 1:(length(sorted_vals) - 1)
            threshold = (sorted_vals[s] + sorted_vals[s + 1]) / 2.0
            gain = 0.0
            for group in leaf_groups
                isempty(group) && continue
                lg = zeros(Float64, n_classes); lh = zeros(Float64, n_classes); lc = 0
                tg = zeros(Float64, n_classes); th = zeros(Float64, n_classes)
                for idx in group
                    for c in 1:n_classes; tg[c] += gradients[idx, c]; th[c] += hessians[idx, c]; end
                    if cat_encoded[idx, j] <= threshold
                        lc += 1
                        for c in 1:n_classes; lg[c] += gradients[idx, c]; lh[c] += hessians[idx, c]; end
                    end
                end
                rc = length(group) - lc
                (lc < min_data_in_leaf || rc < min_data_in_leaf) && continue
                for c in 1:n_classes
                    rg, rh = tg[c] - lg[c], th[c] - lh[c]
                    gain += lg[c]^2 / (lh[c] + l2_leaf_reg) + rg^2 / (rh + l2_leaf_reg) -
                            tg[c]^2 / (th[c] + l2_leaf_reg)
                end
            end
            gain > best.gain && (best = SplitCandidate(j, :categorical, threshold, gain))
        end
    end

    return best
end
