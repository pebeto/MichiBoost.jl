# Single-run profiler for a MichiBoost training.  Produces:
#   1. Wall-time, allocation, and GC totals from @timed.
#   2. CPU sampling profile across all threads (top self-time functions).
#   3. CPU sampling profile filtered to main thread only — anything that
#      shows up here is running serially and is what's capping the
#      threading curve.
#   4. Allocation profile by type (which objects drive GC pressure).
#
# Writes the full flat and tree views to benchmark/profile_flat.txt and
# benchmark/profile_tree.txt so you can open them after the run.
#
#   julia --project=benchmark -t 4 benchmark/profile.jl
#   julia --project=benchmark -t 4 benchmark/profile.jl --n=50000
#   julia --project=benchmark -t 4 benchmark/profile.jl --task=regression

include("common.jl")

using Profile
using Printf

const N = let s = findfirst(a -> startswith(a, "--n="), ARGS)
    s === nothing ? 20_000 : parse(Int, split(ARGS[s], "=")[2])
end
const TASK = let s = findfirst(a -> startswith(a, "--task="), ARGS)
    s === nothing ? "multiclass" : split(ARGS[s], "=")[2]
end
const ITERS_PROFILE = let s = findfirst(a -> startswith(a, "--iters="), ARGS)
    s === nothing ? ITERS : parse(Int, split(ARGS[s], "=")[2])
end

n_total = round(Int, N / 0.8)

X, y, loss, make_model = if TASK == "regression"
    X, y = regression_data(n=n_total, p=54)
    X, y, "RMSE",
    () -> MichiBoostRegressor(; iterations=ITERS_PROFILE, learning_rate=LR, depth=DEPTH,
                              loss_function="RMSE", random_seed=SEED, verbose=false)
elseif TASK == "binary"
    X, y = binary_data(n=n_total, p=54)
    X, y, "Logloss",
    () -> MichiBoostClassifier(; iterations=ITERS_PROFILE, learning_rate=LR, depth=DEPTH,
                               loss_function="Logloss", random_seed=SEED, verbose=false)
else
    X, y = multiclass_data(n=n_total, p=54, k=7)
    X, y, "MultiClass",
    () -> MichiBoostClassifier(; iterations=ITERS_PROFILE, learning_rate=LR, depth=DEPTH,
                               loss_function="MultiClass", random_seed=SEED, verbose=false)
end

X_tr, y_tr, _, _ = train_test_split(X, y)
pool_tr = MichiBoost.Pool(X_tr; label=y_tr)

println("Task:     $TASK ($loss)")
println("Data:     $(size(X_tr, 1)) rows × $(size(X_tr, 2)) features")
println("Threads:  $N_THREADS")
println("Iters:    $ITERS_PROFILE")

println("\nWarmup...")
MichiBoost.fit!(make_model(), pool_tr; iterations=3)

println("\n", "="^72)
println("WALL TIME + ALLOCATION")
println("="^72)
GC.gc()
stats = @timed MichiBoost.fit!(make_model(), pool_tr)
@printf "  wall       %8.3f s\n" stats.time
@printf "  allocated  %8.1f MB   (%d allocations)\n" (stats.bytes/1024^2) stats.gcstats.poolalloc
@printf "  gc         %8.1f %% of wall  (%.3f s)\n" (100*stats.gctime/max(stats.time, 1e-9)) stats.gctime
@printf "  per iter   %8.1f ms/iter  %8.1f MB/iter  %d allocs/iter\n" (1000*stats.time/ITERS_PROFILE) (stats.bytes/1024^2/ITERS_PROFILE) (stats.gcstats.poolalloc ÷ ITERS_PROFILE)

println("\n", "="^72)
println("CPU PROFILE (all threads, top self-time)")
println("="^72)
Profile.clear()
Profile.init(; n=10_000_000, delay=0.001)
Profile.@profile MichiBoost.fit!(make_model(), pool_tr)

Profile.print(
    IOContext(stdout, :displaysize => (50, 200));
    format=:flat, sortedby=:count, mincount=30, noisefloor=2.0,
)

println("\n", "="^72)
println("CPU PROFILE (main thread only — serial regions)")
println("="^72)
Profile.print(
    IOContext(stdout, :displaysize => (40, 200));
    format=:flat, sortedby=:count, threads=[1], mincount=10, noisefloor=2.0,
)

flat_path = joinpath(@__DIR__, "profile_flat.txt")
tree_path = joinpath(@__DIR__, "profile_tree.txt")
open(flat_path, "w") do io
    Profile.print(io; format=:flat, sortedby=:count, mincount=3)
end
open(tree_path, "w") do io
    Profile.print(io; format=:tree, mincount=20)
end
println("\n  full flat view → $flat_path")
println("  full tree view → $tree_path")

if "--no-allocs" in ARGS
    println("\n(skipping allocation profile — --no-allocs passed)")
else
    println("\n", "="^72)
    println("ALLOCATION PROFILE (sample_rate=0.001, extrapolate ×1000)")
    println("="^72)
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=0.001 MichiBoost.fit!(make_model(), pool_tr)
    results = Profile.Allocs.fetch()

    by_type = Dict{String,Tuple{Int,Int}}()
    for a in results.allocs
        t = string(a.type)
        c, b = get(by_type, t, (0, 0))
        by_type[t] = (c + 1, b + a.size)
    end
    sorted = sort(collect(by_type); by = x -> -x[2][2])

    @printf "  %10s  %12s  %s\n" "sampled MB" "sampled #" "type"
    @printf "  %s\n" repeat('-', 90)
    for (t, (c, b)) in first(sorted, 20)
        @printf "  %10.2f  %12d  %s\n" (b/1024^2) c t
    end
    @printf "\n  (multiply by ~1000 to estimate totals at full sample rate)\n"
end

println()
