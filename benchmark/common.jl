# Shared helpers for all benchmark scripts: data generators, train wrappers,
# metrics, and tunable constants.  Include with `include("common.jl")`.

using CatBoost
using DataFrames
using MichiBoost
using PythonCall
using Random
using Statistics

const np = pyimport("numpy")
const pd = pyimport("pandas")

const ITERS = 100
const LR = 0.03
const DEPTH = 6
const SEED = 9
const N_THREADS = Threads.nthreads()

# Data generators

function regression_data(; n=200, p=10, seed=SEED)
    rng = MersenneTwister(seed)
    X = randn(rng, n, p)
    y = X * randn(rng, p) .+ 0.5 .* randn(rng, n)
    return X, y
end

function binary_data(; n=200, p=10, seed=SEED)
    rng = MersenneTwister(seed)
    X = randn(rng, n, p)
    y = Float64.(rand(rng, n) .< (1.0 ./ (1.0 .+ exp.(-(X * randn(rng, p))))))
    return X, y
end

function multiclass_data(; n=300, p=10, k=3, seed=SEED)
    rng = MersenneTwister(seed)
    X = randn(rng, n, p)
    y = Float64.([mod(round(Int, X[i, 1] + X[i, 2] + 10), k) for i in 1:n])
    return X, y
end

function categorical_data(; n=1000, p_num=5, p_cat=5, k=20, seed=SEED)
    rng = MersenneTwister(seed)
    X_num = randn(rng, n, p_num)
    X_cat = string.(rand(rng, 0:(k - 1), n, p_cat))
    y = Float64.(rand(rng, n) .< 0.5)
    return X_num, X_cat, y
end

function train_test_split(X::AbstractMatrix, y::AbstractVector; ratio=0.8, seed=SEED)
    rng = MersenneTwister(seed)
    idx = randperm(rng, length(y))
    n_tr = round(Int, ratio * length(y))
    tr, te = idx[1:n_tr], idx[n_tr+1:end]
    return X[tr, :], y[tr], X[te, :], y[te]
end

# CatBoost / MichiBoost training wrappers

function cb_train(X, y; loss="RMSE", cb_pool=nothing, iterations=ITERS)
    pool = cb_pool !== nothing ? cb_pool : CatBoost.Pool(data=np.array(X), label=np.array(y))
    is_reg = loss in ("RMSE", "MAE")
    m = is_reg ?
        CatBoost.CatBoostRegressor(
            iterations=iterations, learning_rate=LR, depth=DEPTH,
            random_seed=SEED, verbose=false, loss_function=loss,
            thread_count=N_THREADS) :
        CatBoost.CatBoostClassifier(
            iterations=iterations, learning_rate=LR, depth=DEPTH,
            random_seed=SEED, verbose=false, loss_function=loss,
            thread_count=N_THREADS)
    CatBoost.fit!(m, pool)
    return m
end

function mb_train(data, y; loss="RMSE", iterations=ITERS)
    m = loss in ("RMSE", "MAE") ?
        MichiBoostRegressor(; iterations=iterations, learning_rate=LR, depth=DEPTH,
            loss_function=loss, random_seed=SEED, verbose=false) :
        MichiBoostClassifier(; iterations=iterations, learning_rate=LR, depth=DEPTH,
            loss_function=loss, random_seed=SEED, verbose=false)
    MichiBoost.fit!(m, data, y)
    return m
end

function make_cat_frames(X_num, X_cat, y)
    cat_names = ["cat$i" for i in 1:size(X_cat, 2)]
    num_names = ["num$i" for i in 1:size(X_num, 2)]
    py_df = pd.DataFrame(pydict(merge(
        Dict(cat_names[i] => X_cat[:, i] for i in eachindex(cat_names)),
        Dict(num_names[i] => X_num[:, i] for i in eachindex(num_names)),
    )))
    jl_df = hcat(
        DataFrame(X_cat, cat_names),
        DataFrame(X_num, num_names),
    )
    cb_pool = CatBoost.Pool(data=py_df, label=np.array(y), cat_features=cat_names)
    jl_pool = MichiBoost.Pool(jl_df; label=y)
    return py_df, jl_df, cb_pool, jl_pool, cat_names
end

# Metrics

rmse(yhat, y) = sqrt(mean((yhat .- y) .^ 2))
mae(yhat, y) = mean(abs.(yhat .- y))

function r2(yhat, y)
    ss_res = sum((y .- yhat) .^ 2)
    ss_tot = sum((y .- mean(y)) .^ 2)
    return 1 - ss_res / ss_tot
end

function binary_logloss(p, y; eps=1e-12)
    pc = clamp.(p, eps, 1 - eps)
    return -mean(y .* log.(pc) .+ (1 .- y) .* log.(1 .- pc))
end

function multiclass_logloss(P, y_int; eps=1e-12)
    # y_int is 1-based class indices
    Pc = clamp.(P, eps, 1 - eps)
    return -mean(log(Pc[i, y_int[i]]) for i in eachindex(y_int))
end

# AUC via Mann-Whitney U (exact, O(n log n))
function auc(score, y)
    n = length(y)
    pos = findall(==(1.0), y)
    neg = findall(==(0.0), y)
    isempty(pos) || isempty(neg) && return NaN
    order = sortperm(score)
    ranks = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && score[order[j + 1]] == score[order[i]]
            j += 1
        end
        avg = (i + j) / 2
        for k in i:j
            ranks[order[k]] = avg
        end
        i = j + 1
    end
    sum_ranks_pos = sum(ranks[pos])
    n_pos, n_neg = length(pos), length(neg)
    U = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return U / (n_pos * n_neg)
end

# Formatting helpers

bench_ms(b) = median(b).time / 1e6
speedup(t_ref, t_cmp) = t_ref / t_cmp  # >1 means cmp is faster
