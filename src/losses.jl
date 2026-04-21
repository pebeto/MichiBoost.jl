@inline _sigmoid(x::Real) = one(x) / (one(x) + exp(-x))

function _softmax(logits::AbstractVector)
    m = maximum(logits)
    e = exp.(logits .- m)
    return e ./ sum(e)
end

function _softmax_matrix(logits::AbstractMatrix)
    result = similar(logits)
    for i in axes(logits, 1)
        result[i, :] = _softmax(view(logits, i, :))
    end
    return result
end

# In-place softmax — writes softmax of each row of `logits` into `out`.
# Avoids the per-row allocations that _softmax / _softmax_matrix do.
function _softmax_matrix!(out::AbstractMatrix, logits::AbstractMatrix)
    n_rows, n_cols = size(logits)
    @inbounds for i in 1:n_rows
        m = logits[i, 1]
        for j in 2:n_cols
            x = logits[i, j]
            x > m && (m = x)
        end
        s = 0.0
        for j in 1:n_cols
            e = exp(logits[i, j] - m)
            out[i, j] = e
            s += e
        end
        inv_s = 1.0 / s
        for j in 1:n_cols
            out[i, j] *= inv_s
        end
    end
    return out
end

loss(::RMSELoss, y, pred) = sqrt(mean((y .- pred) .^ 2))
negative_gradient(::RMSELoss, y, pred) = y .- pred
hessian(::RMSELoss, y, pred) = ones(Float64, length(y))
initial_prediction(::RMSELoss, y) = mean(y)

loss(::MAELoss, y, pred) = mean(abs.(y .- pred))
negative_gradient(::MAELoss, y, pred) = sign.(y .- pred)
hessian(::MAELoss, y, pred) = ones(Float64, length(y))
initial_prediction(::MAELoss, y) = median(y)

# Weighted median over a sequence of values and non-negative weights.  Returns
# the smallest `vals[i]` such that cumulative weight crosses half the total.
function weighted_median(vals::AbstractVector{Float64}, w::AbstractVector{Float64})
    n = length(vals)
    n == 0 && return 0.0
    n == 1 && return vals[1]
    perm = sortperm(vals)
    total = 0.0
    @inbounds for i in 1:n
        total += w[i]
    end
    half = total / 2
    cum = 0.0
    @inbounds for j in 1:n
        cum += w[perm[j]]
        cum >= half && return vals[perm[j]]
    end
    @inbounds return vals[perm[end]]
end

function loss(::LoglossLoss, y, pred)
    p = clamp.(_sigmoid.(pred), 1e-15, 1.0 - 1e-15)
    return -mean(y .* log.(p) .+ (1.0 .- y) .* log.(1.0 .- p))
end

function negative_gradient(::LoglossLoss, y, pred)
    return y .- _sigmoid.(pred)
end

function hessian(::LoglossLoss, y, pred)
    p = _sigmoid.(pred)
    return p .* (1.0 .- p)
end

function initial_prediction(::LoglossLoss, y)
    p = clamp(mean(y), 1e-7, 1.0 - 1e-7)
    return log(p / (1.0 - p))
end

function loss(::MultiClassLoss, y_onehot::AbstractMatrix, pred::AbstractMatrix)
    probs = clamp.(_softmax_matrix(pred), 1e-15, 1.0)
    return -mean(sum(y_onehot .* log.(probs); dims=2))
end

function negative_gradient(::MultiClassLoss, y_onehot::AbstractMatrix, pred::AbstractMatrix)
    return y_onehot .- _softmax_matrix(pred)
end

function hessian(::MultiClassLoss, ::AbstractMatrix, pred::AbstractMatrix)
    probs = _softmax_matrix(pred)
    return probs .* (1.0 .- probs)
end

function initial_prediction(::MultiClassLoss, y_onehot::AbstractMatrix)
    class_probs = clamp.(vec(mean(y_onehot; dims=1)), 1e-7, 1.0 - 1e-7)
    return log.(class_probs)
end

# Fused in-place gradient + hessian.  Writes both outputs in a single pass
# and, for losses that need an intermediate (softmax/sigmoid), uses the
# caller's `scratch` buffer so nothing is allocated per boosting round.
# RMSE/MAE ignore `scratch`.

function gradient_hessian!(
    g::AbstractVector, h::AbstractVector, ::RMSELoss,
    y::AbstractVector, pred::AbstractVector, _scratch,
)
    @. g = y - pred
    fill!(h, 1.0)
    return nothing
end

function gradient_hessian!(
    g::AbstractVector, h::AbstractVector, ::MAELoss,
    y::AbstractVector, pred::AbstractVector, _scratch,
)
    @. g = sign(y - pred)
    fill!(h, 1.0)
    return nothing
end

function gradient_hessian!(
    g::AbstractVector, h::AbstractVector, ::LoglossLoss,
    y::AbstractVector, pred::AbstractVector, scratch::AbstractVector,
)
    @. scratch = 1.0 / (1.0 + exp(-pred))
    @. g = y - scratch
    @. h = scratch * (1.0 - scratch)
    return nothing
end

function gradient_hessian!(
    g::AbstractMatrix, h::AbstractMatrix, ::MultiClassLoss,
    y_onehot::AbstractMatrix, pred::AbstractMatrix, scratch::AbstractMatrix,
)
    _softmax_matrix!(scratch, pred)
    @. g = y_onehot - scratch
    @. h = scratch * (1.0 - scratch)
    return nothing
end

function make_loss(name::AbstractString; n_classes::Int=2)
    upper = uppercase(name)
    upper == "RMSE"       && return RMSELoss()
    upper == "MAE"        && return MAELoss()
    upper in ("LOGLOSS", "CROSSENTROPY") && return LoglossLoss()
    upper in ("MULTICLASS", "MULTILOGLOSS") && return MultiClassLoss(n_classes)
    error("Unknown loss function: $name. Supported: RMSE, MAE, Logloss, MultiClass")
end
