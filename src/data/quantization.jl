function quantize_features(numerical_data::Matrix{Float64}; border_count::Int=254)
    n_samples, n_features = size(numerical_data)
    if n_features == 0
        return QuantizedFeatures(
            Matrix{UInt16}(undef, n_samples, 0),
            Vector{Float64}[],
            Int[],
        )
    end

    borders = Vector{Vector{Float64}}(undef, n_features)
    bins = Matrix{UInt16}(undef, n_samples, n_features)
    n_bins_vec = Vector{Int}(undef, n_features)

    for j in 1:n_features
        col = view(numerical_data, :, j)
        borders[j] = _compute_borders(col, border_count)
        n_bins_vec[j] = length(borders[j]) + 1
        for i in 1:n_samples
            bins[i, j] = _assign_bin(col[i], borders[j])
        end
    end

    return QuantizedFeatures(bins, borders, n_bins_vec)
end

function apply_borders(numerical_data::Matrix{Float64}, borders::Vector{Vector{Float64}})
    n_samples, n_features = size(numerical_data)
    bins = Matrix{UInt16}(undef, n_samples, n_features)
    # For tiny inputs fork-join would cost more than the work itself.  The
    # threshold is empirical: fewer total cells than ~4k puts the overhead
    # above the savings.
    if n_samples * n_features < 4096 || Threads.nthreads() == 1
        for j in 1:n_features
            _apply_borders_column!(bins, numerical_data, borders[j], j, n_samples)
        end
    else
        # Parallelise over features — each column writes to a disjoint region of
        # `bins`, so no synchronisation is needed.  Columnar access also matches
        # Julia's column-major layout for cache locality.
        Threads.@threads :static for j in 1:n_features
            _apply_borders_column!(bins, numerical_data, borders[j], j, n_samples)
        end
    end
    return bins
end

# Function-barrier helper.  Split out so the threaded body gets compiled
# against concrete types and doesn't box `lo` / `v` through the closure.
@inline function _apply_borders_column!(
    bins::Matrix{UInt16},
    data::Matrix{Float64},
    b::Vector{Float64},
    j::Int,
    n_samples::Int,
)
    nb = length(b)
    if nb <= 32
        # Linear scan with early exit — faster than binary search for small
        # border counts due to branch prediction and no function call overhead.
        @inbounds for i in 1:n_samples
            v = data[i, j]
            lo = 1
            while lo <= nb && v > b[lo]
                lo += 1
            end
            bins[i, j] = UInt16(lo)
        end
    else
        @inbounds for i in 1:n_samples
            bins[i, j] = UInt16(searchsortedfirst(b, data[i, j]))
        end
    end
    return nothing
end

"""
    _quantile_cut_points(col, max_points)

Return up to `max_points` sorted values from `col`, evenly spaced through its
unique-sorted values.  Used to cap the bin count of target-encoded categorical
columns — at low raw cardinality ordered target statistics produce up to
`n_samples` distinct encoded values, which would make the per-iteration
histogram zeroing and split sweep O(n_samples) rather than O(max_points).
"""
function _quantile_cut_points(col::AbstractVector{Float64}, max_points::Int)
    unique_vals = sort(unique(col))
    n = length(unique_vals)
    n <= max_points && return unique_vals
    out = Vector{Float64}(undef, max_points)
    @inbounds for k in 1:max_points
        out[k] = unique_vals[1 + round(Int, (k - 1) * (n - 1) / (max_points - 1))]
    end
    return out
end

function _compute_borders(values::AbstractVector{Float64}, border_count::Int)
    valid = filter(!isnan, values)
    isempty(valid) && return Float64[]
    sorted = sort(valid)
    n = length(sorted)
    if n <= 1
        return Float64[]
    end

    n_borders = min(border_count, n - 1)
    borders = Float64[]
    for i in 1:n_borders
        idx = clamp(round(Int, i / (n_borders + 1) * n), 1, n)
        border = sorted[idx]
        if isempty(borders) || border > last(borders) + eps(border)
            push!(borders, border)
        end
    end
    return borders
end

function _assign_bin(value::Float64, borders::Vector{Float64})
    if isnan(value)
        return UInt16(0)
    end
    lo, hi = 1, length(borders)
    while lo <= hi
        mid = (lo + hi) >> 1
        if value <= borders[mid]
            hi = mid - 1
        else
            lo = mid + 1
        end
    end
    return UInt16(lo)
end
