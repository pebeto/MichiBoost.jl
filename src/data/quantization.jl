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
    for j in 1:n_features, i in 1:n_samples
        bins[i, j] = _assign_bin(numerical_data[i, j], borders[j])
    end
    return bins
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
