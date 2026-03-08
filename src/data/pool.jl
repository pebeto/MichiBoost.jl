"""
    Pool(data; label=nothing, cat_features=nothing, text_features=nothing,
         feature_names=nothing, weight=nothing, group_id=nothing)

Create a `Pool` from tabular data.

# Arguments
- `data`: any `Tables.jl`-compatible table (e.g. `DataFrame`, `NamedTuple` of
  vectors) or an `AbstractMatrix`.
- `label`: target vector.  Numeric values are kept as-is; string / categorical
  values are automatically encoded to `Float64`.
- `cat_features`: indices (0-based `Int`) or column names (`Symbol` / `String`)
  of columns that should be treated as categorical.  String columns are
  auto-detected even without this argument.
- `text_features`: same format as `cat_features`; treated identically
  (as categorical).
- `feature_names`: optional vector of column names.
- `weight`: optional per-sample weights (`Vector{<:Real}`).
- `group_id`: optional group identifiers (for ranking-style tasks).

# Examples
```julia
# From a matrix
pool = Pool([1.0 2.0; 3.0 4.0]; label=[0.0, 1.0])

# From a table with auto-detected categorical columns
pool = Pool((color=["red","blue","red"], size=[1.0, 2.0, 3.0]);
            label=[0.0, 1.0, 0.0])
```
"""
function Pool(data; label=nothing, cat_features=nothing, text_features=nothing,
              feature_names=nothing, weight=nothing, group_id=nothing)
    return _build_pool(data; label, cat_features, text_features, feature_names, weight,
                       group_id)
end

function Pool(; data, label=nothing, cat_features=nothing, text_features=nothing,
              feature_names=nothing, weight=nothing, group_id=nothing)
    return _build_pool(data; label, cat_features, text_features, feature_names, weight,
                       group_id)
end

function _build_pool(data; label=nothing, cat_features=nothing, text_features=nothing,
                     feature_names=nothing, weight=nothing, group_id=nothing)
    if Tables.istable(data)
        ct = Tables.columntable(data)
        col_names = collect(Tables.columnnames(ct))
        columns = [collect(ct[name]) for name in col_names]
    elseif data isa AbstractMatrix
        n_cols = size(data, 2)
        col_names = [Symbol("x$i") for i in 1:n_cols]
        columns = [data[:, i] for i in 1:n_cols]
    else
        error("Unsupported data type: $(typeof(data)). Pass a table or matrix.")
    end

    n_samples = length(first(columns))
    n_total_features = length(columns)
    fnames = feature_names !== nothing ? Symbol.(feature_names) : Symbol.(col_names)

    cat_indices_set = Set{Int}()
    _collect_cat_indices!(cat_indices_set, cat_features, fnames)
    _collect_cat_indices!(cat_indices_set, text_features, fnames)
    for (i, col) in enumerate(columns)
        if eltype(col) <: AbstractString || eltype(col) <: CategoricalValue
            push!(cat_indices_set, i)
        end
    end

    cat_indices = sort(collect(cat_indices_set))
    num_indices = sort(setdiff(1:n_total_features, cat_indices_set))

    features_numerical = Matrix{Float64}(undef, n_samples, length(num_indices))
    for (j, idx) in enumerate(num_indices)
        for i in 1:n_samples
            features_numerical[i, j] = _to_float(columns[idx][i])
        end
    end

    features_categorical = Vector{Vector{UInt32}}(undef, length(cat_indices))
    cat_mapping = Vector{Dict{Any,UInt32}}(undef, length(cat_indices))
    for (j, idx) in enumerate(cat_indices)
        col = columns[idx]
        mapping = Dict{Any,UInt32}()
        encoded = Vector{UInt32}(undef, n_samples)
        next_id = UInt32(1)
        for i in 1:n_samples
            v = _unwrap_cat(col[i])
            if !haskey(mapping, v)
                mapping[v] = next_id
                next_id += UInt32(1)
            end
            encoded[i] = mapping[v]
        end
        features_categorical[j] = encoded
        cat_mapping[j] = mapping
    end

    processed_label, label_mapping, label_classes = if label !== nothing
        _process_label_full(label)
    else
        nothing, nothing, nothing
    end

    processed_weight = weight !== nothing ? Float64.(weight) : nothing

    return Pool(features_numerical, features_categorical, cat_mapping,
                processed_label, label_mapping, label_classes, fnames,
                num_indices, cat_indices, n_samples, n_total_features,
                processed_weight, group_id)
end

function _collect_cat_indices!(set, features, fnames)
    features === nothing && return
    for cf in features
        if cf isa Integer
            push!(set, cf + 1)
        elseif cf isa Symbol || cf isa AbstractString
            idx = findfirst(==(Symbol(cf)), fnames)
            idx !== nothing && push!(set, idx)
        end
    end
end

_to_float(x::Real) = Float64(x)
_to_float(x::CategoricalValue) = Float64(unwrap(x))
_to_float(::Missing) = NaN
_to_float(x) = parse(Float64, string(x))

_unwrap_cat(x::CategoricalValue) = unwrap(x)
_unwrap_cat(x) = x

function _process_label_full(label)
    unwrapped = _unwrap_cat.(label)
    if all(x -> x isa Real, unwrapped)
        return Float64.(unwrapped), nothing, nothing
    end
    unique_vals = sort(unique(unwrapped))
    label_map = Dict(v => Float64(i - 1) for (i, v) in enumerate(unique_vals))
    return [label_map[v] for v in unwrapped], label_map, unique_vals
end

n_numerical(pool::Pool)  = size(pool.features_numerical, 2)
n_categorical(pool::Pool) = length(pool.features_categorical)

function get_label(pool::Pool)
    pool.label === nothing && error("Pool has no label")
    return pool.label
end

"""
    slice(pool::Pool, indices) -> Pool

Return a new `Pool` containing only the rows at `indices`.

```julia
subset = slice(pool, 1:100)
```
"""
function slice(pool::Pool, indices::AbstractVector{<:Integer})
    return Pool(
        pool.features_numerical[indices, :],
        [col[indices] for col in pool.features_categorical],
        pool.cat_mapping,
        pool.label !== nothing ? pool.label[indices] : nothing,
        pool.label_mapping,
        pool.label_classes,
        pool.feature_names,
        pool.numerical_feature_indices,
        pool.categorical_feature_indices,
        length(indices),
        pool.n_features,
        pool.weight !== nothing ? pool.weight[indices] : nothing,
        pool.group_id !== nothing ? pool.group_id[indices] : nothing,
    )
end
