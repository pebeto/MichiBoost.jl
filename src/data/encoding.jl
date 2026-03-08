function compute_ordered_target_stats(cat_features::Vector{Vector{UInt32}},
                                      label::Vector{Float64},
                                      permutation::Vector{Int};
                                      alpha::Float64=1.0,
                                      prior::Union{Float64,Nothing}=nothing)
    n_samples = length(label)
    n_cat = length(cat_features)
    n_cat == 0 && return (Matrix{Float64}(undef, n_samples, 0),
                          OrderedTargetEncoder(0.0, alpha, Dict{UInt32,Tuple{Float64,Int}}[]))

    p = prior !== nothing ? prior : mean(label)
    encoded = Matrix{Float64}(undef, n_samples, n_cat)
    accumulators = [Dict{UInt32,Tuple{Float64,Int}}() for _ in 1:n_cat]

    for pos in 1:n_samples
        i = permutation[pos]
        for f in 1:n_cat
            cat_id = cat_features[f][i]
            sum_target, count = get(accumulators[f], cat_id, (0.0, 0))
            encoded[i, f] = (sum_target + alpha * p) / (count + alpha)
            accumulators[f][cat_id] = (sum_target + label[i], count + 1)
        end
    end

    encoder = OrderedTargetEncoder(p, alpha, [copy(a) for a in accumulators])
    return encoded, encoder
end

function plain_target_encode(cat_features::Vector{Vector{UInt32}}, y::Vector{Float64})
    n_samples = length(y)
    n_cat = length(cat_features)
    p = mean(y)
    alpha = 1.0

    stats = [Dict{UInt32,Tuple{Float64,Int}}() for _ in 1:n_cat]
    for f in 1:n_cat, i in 1:n_samples
        cat_id = cat_features[f][i]
        sum_t, cnt = get(stats[f], cat_id, (0.0, 0))
        stats[f][cat_id] = (sum_t + y[i], cnt + 1)
    end

    encoded = Matrix{Float64}(undef, n_samples, n_cat)
    for f in 1:n_cat, i in 1:n_samples
        cat_id = cat_features[f][i]
        sum_t, cnt = stats[f][cat_id]
        encoded[i, f] = (sum_t + alpha * p) / (cnt + alpha)
    end

    return encoded, OrderedTargetEncoder(p, alpha, stats)
end

function encode_categorical(encoder::OrderedTargetEncoder, cat_features::Vector{Vector{UInt32}})
    n_cat = length(cat_features)
    n_cat == 0 && return Matrix{Float64}(undef, length(first(cat_features)), 0)

    n_samples = length(first(cat_features))
    encoded = Matrix{Float64}(undef, n_samples, n_cat)
    for f in 1:n_cat, i in 1:n_samples
        cat_id = cat_features[f][i]
        sum_target, count = get(encoder.category_stats[f], cat_id, (0.0, 0))
        encoded[i, f] = (sum_target + encoder.alpha * encoder.prior) / (count + encoder.alpha)
    end
    return encoded
end
