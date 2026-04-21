# Scaling behaviour on a real dataset (Covertype, 581k rows × 54 features)
# plus a threading curve.  Synthetic data is easy to ace; real data is the
# honesty check.  The threading curve is produced by spawning one child
# Julia process per thread count so every measurement starts from a clean
# process.
#
#   # single run at the current thread count
#   julia --project=benchmark -t 4 benchmark/scaling.jl
#
#   # threading curve: spawns 1/2/4/8-thread children and aggregates output
#   julia --project=benchmark benchmark/scaling.jl --sweep

include("common.jl")

using BenchmarkTools
using CSV
using Downloads

const CHILD = haskey(ENV, "MB_SCALING_CHILD")
const SWEEP = "--sweep" in ARGS
const SUBSET_N = let s = findfirst(a -> startswith(a, "--n="), ARGS)
    s === nothing ? 50_000 : parse(Int, split(ARGS[s], "=")[2])
end
const THREAD_COUNTS = [1, 2, 4, 8]

function download_covertype()
    url = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
    data_dir = joinpath(@__DIR__, "data")
    zip_path = joinpath(data_dir, "covertype.zip")
    gz_path = joinpath(data_dir, "covtype.data.gz")
    csv_path = joinpath(data_dir, "covtype.data")

    isdir(data_dir) || mkdir(data_dir)

    if !isfile(zip_path) && !isfile(gz_path) && !isfile(csv_path)
        println("Downloading Covertype dataset (~10.7 MB)...")
        Downloads.download(url, zip_path)
    end
    if isfile(zip_path) && !isfile(gz_path)
        run(`unzip -o $zip_path -d $data_dir`)
    end
    if isfile(gz_path) && !isfile(csv_path)
        run(`gunzip -k $gz_path`)
    end
    isfile(csv_path) || error("Could not find covtype.data after extraction.")
    return csv_path
end

function load_covertype(n_samples::Int)
    csv_path = download_covertype()
    df = CSV.read(csv_path, DataFrame; header=false)
    X = Matrix{Float64}(df[:, 1:end-1])
    y = Vector{Float64}(df[:, end])
    n = min(n_samples, size(X, 1))
    idx = shuffle(MersenneTwister(SEED), 1:size(X, 1))[1:n]
    return X[idx, :], y[idx]
end

function run_once(X_tr, y_tr, X_te, y_te)
    jl_pool_tr = MichiBoost.Pool(X_tr; label=y_tr)
    cb_pool_tr = CatBoost.Pool(data=np.array(X_tr), label=np.array(y_tr))
    X_te_np = np.array(X_te)

    mb_fn = () -> MichiBoostClassifier(; iterations=ITERS, learning_rate=LR, depth=DEPTH,
                      random_seed=SEED, verbose=false)
    cb_fn = () -> CatBoost.CatBoostClassifier(iterations=ITERS, learning_rate=LR, depth=DEPTH,
                      random_seed=SEED, thread_count=N_THREADS, verbose=false)

    t_mb = median(@benchmark(MichiBoost.fit!($mb_fn(), $jl_pool_tr), samples=3, evals=1)).time / 1e6
    t_cb = median(@benchmark(CatBoost.fit!($cb_fn(), $cb_pool_tr), samples=3, evals=1)).time / 1e6

    mb = mb_fn(); MichiBoost.fit!(mb, jl_pool_tr)
    cb = cb_fn(); CatBoost.fit!(cb, cb_pool_tr)

    mb_acc = mean(MichiBoost.predict(mb, X_te) .== y_te)
    cb_acc = mean(pyconvert(Vector{Float64}, CatBoost.predict(cb, X_te_np)) .== y_te)

    t_mb_pred = median(@benchmark(MichiBoost.predict($mb, $X_te))).time / 1e6
    t_cb_pred = median(@benchmark(CatBoost.predict($cb, $X_te_np))).time / 1e6

    return (; t_mb, t_cb, t_mb_pred, t_cb_pred, mb_acc, cb_acc)
end

if SWEEP && !CHILD
    # Parent mode: spawn a child per thread count, parse its RESULT line.
    results = Dict{Int, NamedTuple}()
    script = @__FILE__
    for t in THREAD_COUNTS
        println("\n>>> Spawning child with -t $t ...")
        cmd = Cmd(`julia --project=benchmark -t $t $script --n=$SUBSET_N`;
                  env=merge(ENV, Dict("MB_SCALING_CHILD" => "1")))
        output = read(cmd, String)
        print(output)
        for line in split(output, '\n')
            if startswith(line, "RESULT ")
                kv = Dict{String,Float64}()
                for pair in split(strip(line[length("RESULT ")+1:end]))
                    k, v = split(pair, "=")
                    kv[k] = parse(Float64, v)
                end
                results[t] = (; t_mb=kv["t_mb"], t_cb=kv["t_cb"],
                              t_mb_pred=kv["t_mb_pred"], t_cb_pred=kv["t_cb_pred"],
                              mb_acc=kv["mb_acc"], cb_acc=kv["cb_acc"])
            end
        end
    end

    println("\n", "="^72)
    println("THREADING CURVE  (Covertype, n=$SUBSET_N, 7 classes)")
    println("="^72)
    println("  threads     MichiBoost train   CatBoost train   MB speedup vs 1-thread")
    base_mb = get(results, 1, nothing)
    for t in THREAD_COUNTS
        haskey(results, t) || continue
        r = results[t]
        rel = base_mb === nothing ? "-" : "$(round(base_mb.t_mb / r.t_mb; digits=2))x"
        println("  $(lpad(t, 3))         $(rpad(round(r.t_mb; digits=1), 10))ms      $(rpad(round(r.t_cb; digits=1), 10))ms    $rel")
    end
    println()
    println("  Inference (median ms)")
    for t in THREAD_COUNTS
        haskey(results, t) || continue
        r = results[t]
        println("  threads=$t   MichiBoost $(round(r.t_mb_pred; digits=3))ms   CatBoost $(round(r.t_cb_pred; digits=3))ms")
    end
    exit(0)
end

# Single-run mode (also the child's path when MB_SCALING_CHILD is set).
println("Loading Covertype (n=$SUBSET_N)...")
X, y = load_covertype(SUBSET_N)
X_tr, y_tr, X_te, y_te = train_test_split(X, y)

println("Training: $(size(X_tr))  |  Test: $(size(X_te))  |  Threads: $N_THREADS")
r = run_once(X_tr, y_tr, X_te, y_te)

if CHILD
    # Emit one parseable line for the parent; keep human-readable output too.
    println("RESULT threads=$N_THREADS t_mb=$(r.t_mb) t_cb=$(r.t_cb) " *
            "t_mb_pred=$(r.t_mb_pred) t_cb_pred=$(r.t_cb_pred) " *
            "mb_acc=$(r.mb_acc) cb_acc=$(r.cb_acc)")
else
    println("\n", "="^72)
    println("SUMMARY  (Covertype, n=$SUBSET_N, 7 classes, $N_THREADS threads)")
    println("="^72)
    println("  Training:  CatBoost $(round(r.t_cb; digits=1)) ms   MichiBoost $(round(r.t_mb; digits=1)) ms")
    println("  Inference: CatBoost $(round(r.t_cb_pred; digits=3)) ms   MichiBoost $(round(r.t_mb_pred; digits=3)) ms")
    println("  Accuracy:  CatBoost $(round(100*r.cb_acc; digits=2))%   MichiBoost $(round(100*r.mb_acc; digits=2))%")
    println()
    println("  Training speedup:  MB $(round(r.t_cb/r.t_mb; digits=2))x")
    println("  Inference speedup: MB $(round(r.t_cb_pred/r.t_mb_pred; digits=2))x")
end

println()
