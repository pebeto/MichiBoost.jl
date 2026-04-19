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

loss(::RMSELoss, y, pred) = sqrt(mean((y .- pred) .^ 2))
negative_gradient(::RMSELoss, y, pred) = y .- pred
hessian(::RMSELoss, y, pred) = ones(Float64, length(y))
initial_prediction(::RMSELoss, y) = mean(y)

loss(::MAELoss, y, pred) = mean(abs.(y .- pred))
negative_gradient(::MAELoss, y, pred) = sign.(y .- pred)
hessian(::MAELoss, y, pred) = ones(Float64, length(y))
initial_prediction(::MAELoss, y) = median(y)

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

function make_loss(name::AbstractString; n_classes::Int=2)
    upper = uppercase(name)
    upper == "RMSE"       && return RMSELoss()
    upper == "MAE"        && return MAELoss()
    upper in ("LOGLOSS", "CROSSENTROPY") && return LoglossLoss()
    upper in ("MULTICLASS", "MULTILOGLOSS") && return MultiClassLoss(n_classes)
    error("Unknown loss function: $name. Supported: RMSE, MAE, Logloss, MultiClass")
end
