using BenchmarkTools
using CatBoost
using DataFrames
using MichiBoost
using MLDatasets
using PythonCall
using Random
using Statistics

const np = pyimport("numpy")

const ITERS = 100
const LR = 0.03
const DEPTH = 6
const SEED = 9
const N_THREADS = Threads.nthreads()

function train_test_split(X::Matrix, y::Vector; ratio=0.8, seed=SEED)
    rng = MersenneTwister(seed)
    idx = randperm(rng, length(y))
    n_tr = round(Int, ratio * length(y))
    tr, te = idx[1:n_tr], idx[n_tr+1:end]
    return X[tr, :], y[tr], X[te, :], y[te]
end

println("Loading Boston Housing dataset from MLDatasets.jl...")
dataset = BostonHousing()
df = dataset.dataframe
X = Matrix{Float64}(df[:, 1:end-1])
y = Vector{Float64}(df[:, end])
X_tr, y_tr, X_te, y_te = train_test_split(X, y)

println("Dataset size: $(size(X_tr)) training, $(size(X_te)) test")
println("Using $N_THREADS threads")

# Pre-convert test set for CatBoost inference — avoids timing Python array construction
X_te_np = np.array(X_te)

# ============================================================
# MichiBoost Benchmark
# ============================================================
println("\n" * "="^60)
println("MichiBoost Benchmark")
println("="^60)

jl_pool_tr = MichiBoost.Pool(X_tr; label=y_tr)

println("Benchmarking training (5 samples)...")
b_mb = @benchmark MichiBoost.fit!(
    MichiBoostRegressor(; iterations=$ITERS, learning_rate=$LR, depth=$DEPTH,
        random_seed=$SEED, verbose=false),
    $jl_pool_tr) samples=5 evals=1

t_mb = median(b_mb).time / 1e6
println("Training: $(round(t_mb; digits=1)) ms")

# Train final model
mb_model = MichiBoostRegressor(; iterations=ITERS, learning_rate=LR, depth=DEPTH,
    random_seed=SEED, verbose=false)
MichiBoost.fit!(mb_model, jl_pool_tr)

mb_pred = MichiBoost.predict(mb_model, X_te)
mb_rmse = sqrt(mean((mb_pred .- y_te).^2))
println("RMSE (test): $(round(mb_rmse; digits=4))")

println("Benchmarking inference...")
t_mb_pred = median(@benchmark(MichiBoost.predict($mb_model, $X_te))).time / 1e6
println("Inference: $(round(t_mb_pred; digits=3)) ms")

# ============================================================
# CatBoost.jl Benchmark
# ============================================================
println("\n" * "="^60)
println("CatBoost.jl Benchmark")
println("="^60)

cb_pool_tr = CatBoost.Pool(data=np.array(X_tr), label=np.array(y_tr))

println("Benchmarking training (5 samples)...")
b_cb = @benchmark begin
    model = CatBoost.CatBoostRegressor(
        iterations=$ITERS,
        learning_rate=$LR,
        depth=$DEPTH,
        random_seed=$SEED,
        thread_count=$N_THREADS,
        verbose=false
    )
    CatBoost.fit!(model, $cb_pool_tr)
end samples=5 evals=1

t_cb = median(b_cb).time / 1e6
println("Training: $(round(t_cb; digits=1)) ms")

# Train final model
cb_model = CatBoost.CatBoostRegressor(
    iterations=ITERS,
    learning_rate=LR,
    depth=DEPTH,
    random_seed=SEED,
    thread_count=N_THREADS,
    verbose=false
)
CatBoost.fit!(cb_model, cb_pool_tr)

cb_pred = CatBoost.predict(cb_model, X_te_np)
cb_rmse = sqrt(mean((pyconvert(Vector{Float64}, cb_pred) .- y_te).^2))
println("RMSE (test): $(round(cb_rmse; digits=4))")

println("Benchmarking inference...")
t_cb_pred = median(@benchmark(CatBoost.predict($cb_model, $X_te_np))).time / 1e6
println("Inference: $(round(t_cb_pred; digits=3)) ms")

# ============================================================
# Summary
# ============================================================
println("\n" * "="^60)
println("Summary")
println("="^60)
println("Training:  CatBoost $(round(t_cb; digits=1)) ms  |  MichiBoost $(round(t_mb; digits=1)) ms")
println("Inference: CatBoost $(round(t_cb_pred; digits=3)) ms  |  MichiBoost $(round(t_mb_pred; digits=3)) ms")
println("RMSE:      CatBoost $(round(cb_rmse; digits=4))  |  MichiBoost $(round(mb_rmse; digits=4))")
speedup_train = t_cb / t_mb
speedup_infer = t_cb_pred / t_mb_pred
println("\nMichiBoost speedup: Training $(round(speedup_train; digits=2))x faster  |  Inference $(round(speedup_infer; digits=2))x faster")
println("="^60)