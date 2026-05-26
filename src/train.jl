function train(
    pool::Pool;
    iterations::Int=1000,
    learning_rate::Float64=0.03,
    depth::Int=6,
    l2_leaf_reg::Float64=3.0,
    loss_function::Union{String,Symbol,LossFunction}="RMSE",
    border_count::Int=254,
    min_data_in_leaf::Int=1,
    random_seed::Union{Int,Nothing}=nothing,
    verbose::Bool=true,
    rsm::Float64=1.0,
    early_stopping_rounds::Union{Int,Nothing}=nothing,
    eval_pool::Union{Pool,Nothing}=nothing,
    eval_metric::Union{Type{<:Metric},AbstractString,Symbol,Nothing}=nothing,
    boosting_type::Union{String,Symbol}="Ordered",
    init_model::Union{MichiBoostModel,Nothing}=nothing,
    snapshot_path::Union{AbstractString,Nothing}=nothing,
    snapshot_interval::Int=100,
    kwargs...,
)
    if snapshot_path !== nothing && snapshot_interval < 1
        error("`snapshot_interval` must be >= 1; got $snapshot_interval.")
    end
    # Normalise Symbol forms once at entry so the downstream `uppercase` / `==`
    # comparisons keep operating on Strings.
    loss_function isa Symbol && (loss_function = String(loss_function))
    boosting_type isa Symbol && (boosting_type = String(boosting_type))
    if eval_metric isa AbstractString || eval_metric isa Symbol
        eval_metric = _resolve_metric(eval_metric)
    end

    rng = random_seed !== nothing ? MersenneTwister(random_seed) : MersenneTwister()

    label = get_label(pool)
    n_samples = pool.n_samples
    n_num = n_numerical(pool)
    n_cat = n_categorical(pool)

    # When continuing from `init_model`, reuse its borders so the new bin
    # indices align with the existing trees. Recomputing borders on the
    # continuation pool would break every split in the inherited trees.
    qf = if init_model !== nothing
        length(init_model.borders) == n_num || error(
            "`init_model` has $(length(init_model.borders)) numerical features but " *
            "pool has $n_num.",
        )
        bins = if n_num > 0
            apply_borders(pool.features_numerical, init_model.borders)
        else
            Matrix{UInt16}(undef, n_samples, 0)
        end
        n_bins_vec = [length(b) + 1 for b in init_model.borders]
        QuantizedFeatures(bins, init_model.borders, n_bins_vec)
    else
        quantize_features(pool.features_numerical; border_count)
    end

    task = _resolve_task(loss_function, label, pool.label_classes, n_samples)
    loss_function = task.loss_function
    is_custom_loss = task.is_custom_loss
    is_multiclass = task.is_multiclass
    n_classes = task.n_classes
    y = task.y
    y_onehot = task.y_onehot
    class_labels_final = task.class_labels_final

    if init_model !== nothing
        init_model.is_multiclass == is_multiclass || error(
            "`init_model.is_multiclass = $(init_model.is_multiclass)` does not match " *
            "the current task ($(is_multiclass ? "multiclass" : "binary/regression")).",
        )
        init_model.n_classes == n_classes || error(
            "`init_model.n_classes = $(init_model.n_classes)` does not match the " *
            "current task ($n_classes).",
        )
    end

    # Categorical encoding. Reuse init_model's encoder when continuing so the
    # encoded values feeding into inherited trees match what those trees saw
    # during their original training.
    permutation = randperm(rng, n_samples)
    cat_encoded, encoder = if init_model !== nothing
        if n_cat > 0
            init_model.encoder !== nothing || error(
                "`init_model` has no categorical encoder but the pool has $n_cat " *
                "categorical features.",
            )
            (
                encode_categorical(init_model.encoder, pool.features_categorical),
                init_model.encoder,
            )
        else
            (Matrix{Float64}(undef, n_samples, 0), init_model.encoder)
        end
    elseif n_cat > 0
        if boosting_type == "Ordered"
            compute_ordered_target_stats(pool.features_categorical, y, permutation; alpha=1.0)
        else
            plain_target_encode(pool.features_categorical, y)
        end
    else
        (
            Matrix{Float64}(undef, n_samples, 0),
            OrderedTargetEncoder(mean(y), 1.0, Dict{UInt32,Tuple{Float64,Int}}[]),
        )
    end

    lf = is_custom_loss ? loss_function : make_loss(loss_function; n_classes)

    # Initial predictions. Both `initial_pred` (multiclass vector) and
    # `initial_pred_val` (binary/regression scalar) are bound so `_setup_eval`
    # can take both as arguments without UndefVarError on the unused branch.
    initial_pred = Float64[]
    initial_pred_val = 0.0
    if init_model !== nothing
        if is_multiclass
            initial_pred = init_model.initial_pred::Vector{Float64}
        else
            initial_pred_val = init_model.initial_pred::Float64
        end
        predictions = _predict_raw_for_init(
            init_model, qf.bins, cat_encoded, learning_rate
        )
    elseif is_multiclass
        initial_pred = initial_prediction(lf, y_onehot)
        predictions = repeat(initial_pred', n_samples, 1)
    else
        initial_pred_val = initial_prediction(lf, y)
        predictions = fill(initial_pred_val, n_samples)
    end

    trees = if init_model !== nothing
        if is_multiclass
            copy(init_model.trees::Vector{SymmetricTreeMultiClass})
        else
            copy(init_model.trees::Vector{SymmetricTree})
        end
    else
        is_multiclass ? SymmetricTreeMultiClass[] : SymmetricTree[]
    end

    es_orientation, es_metric_fn = if eval_metric === nothing
        # Default: use the training loss for the eval signal (lower is better).
        :minimize, (y, p) -> loss(lf, y, p)
    else
        _eval_metric(eval_metric, is_multiclass, n_classes)
    end
    best_eval_value = es_orientation == :minimize ? Inf : -Inf
    rounds_no_improve, best_iter = 0, 0
    sample_indices = collect(1:n_samples)

    weights = pool.weight !== nothing ? pool.weight : ones(Float64, n_samples)

    max_leaves = 1 << depth
    # Categorical encoded values can have up to n_samples distinct values,
    # so the buffer must accommodate both numerical bins and categorical ranks.
    max_bins = max(border_count + 2, n_samples + 1)
    nt = Threads.maxthreadid()
    buffers = if is_multiclass
        [SplitBuffersMC(max_leaves, max_bins, n_classes, n_samples) for _ in 1:nt]
    else
        [SplitBuffers(max_leaves, max_bins, n_samples) for _ in 1:nt]
    end

    # Cut points per categorical feature. At low raw cardinality ordered
    # target statistics produce up to n_samples distinct encoded values, which
    # would make per-iteration histogram work O(n_samples).  Capping at
    # border_count+1 matches how numerical features are handled and keeps the
    # per-feature bin count bounded regardless of raw cardinality.
    cat_sorted_vals = [
        _quantile_cut_points(view(cat_encoded, :, j), border_count + 1) for j in 1:n_cat
    ]

    hist_cache = if is_multiclass
        HistCacheMC(max_leaves, qf.n_bins, cat_sorted_vals, n_classes)
    else
        HistCache(max_leaves, qf.n_bins, cat_sorted_vals)
    end

    leaf_indices = Vector{Int}(undef, n_samples)

    # Early-stopping eval state: built once, updated incrementally each round
    # so the ES check is O(1) per iteration instead of O(T). `_setup_eval`
    # returns the same NamedTuple shape in both the active and inactive cases
    # (empty matrices when inactive) so the loop can read `eval_state.*`
    # without nullability checks.
    es_active = early_stopping_rounds !== nothing && eval_pool !== nothing
    eval_state = _setup_eval(
        es_active,
        eval_pool,
        qf.borders,
        encoder,
        is_multiclass,
        n_classes,
        initial_pred,
        initial_pred_val,
        init_model,
        learning_rate,
    )

    # Buffers reused across every boosting round. Without this the loop
    # allocates an O(n × n_classes) matrix (or O(n) vector) 4-6 times per
    # iteration for gradient / hessian / softmax temporaries, which on
    # n=40k × k=7 adds up to ~100 MB / iter.
    bufs = _alloc_grad_buffers(is_multiclass, n_samples, n_classes)
    grads_buf = bufs.g
    hess_buf = bufs.h
    scratch_buf = bufs.scratch
    use_refine = lf isa MAELoss
    refine_buf = use_refine ? Vector{Float64}(undef, n_samples) : Float64[]

    for iter in 1:iterations
        if is_multiclass
            gradient_hessian!(grads_buf, hess_buf, lf, y_onehot, predictions, scratch_buf)
            grads_buf .*= weights
            hess_buf .*= weights
            tree = build_symmetric_tree(
                grads_buf,
                hess_buf,
                qf.bins,
                cat_encoded,
                sample_indices,
                depth,
                n_num,
                n_cat,
                qf,
                n_classes;
                l2_leaf_reg,
                min_data_in_leaf,
                rsm,
                rng,
                buffers=buffers::Vector{SplitBuffersMC},
                cat_sorted_vals,
                hist_cache=hist_cache::HistCacheMC,
            )
            push!(trees, tree)
            predict_tree!(
                predictions, tree, qf.bins, cat_encoded, learning_rate, leaf_indices
            )
        else
            gradient_hessian!(grads_buf, hess_buf, lf, y, predictions, scratch_buf)
            grads_buf .*= weights
            hess_buf .*= weights
            # MAE is non-smooth: surrogate gradients (±1) drive split-finding,
            # but leaf values must come from the residual weighted median.
            # Without this, each round shifts predictions by at most
            # learning_rate × 1, causing severe underfitting.
            if use_refine
                refine_buf .= y .- predictions
            end
            tree = build_symmetric_tree(
                grads_buf,
                hess_buf,
                qf.bins,
                cat_encoded,
                sample_indices,
                depth,
                n_num,
                n_cat,
                qf;
                l2_leaf_reg,
                min_data_in_leaf,
                rsm,
                rng,
                buffers=buffers::Vector{SplitBuffers},
                cat_sorted_vals,
                hist_cache=hist_cache::HistCache,
                leaf_refine_values=use_refine ? refine_buf : nothing,
                leaf_refine_weights=use_refine ? weights : nothing,
            )
            push!(trees, tree)
            predict_tree!(
                predictions, tree, qf.bins, cat_encoded, learning_rate, leaf_indices
            )
        end

        if verbose &&
            (iter % max(1, iterations ÷ 10) == 0 || iter == 1 || iter == iterations)
            train_loss = if is_multiclass
                loss(lf, y_onehot, predictions)
            else
                loss(lf, y, predictions)
            end
            println(
                "Iteration $iter/$iterations, train loss: $(round(train_loss; digits=6))"
            )
        end

        if snapshot_path !== nothing &&
            iter != iterations &&
            iter % snapshot_interval == 0
            JLD2.save_object(snapshot_path, _build_model(
                trees, learning_rate, initial_pred, initial_pred_val, encoder,
                qf.borders, pool.feature_names, n_classes, class_labels_final,
                is_multiclass, pool.numerical_feature_indices,
                pool.categorical_feature_indices, length(trees), nothing,
                is_custom_loss ? loss_function : nothing,
            ))
            if verbose
                println("Snapshot written to $snapshot_path (iteration $iter)")
            end
        end

        if es_active
            # Extend the running eval predictions with the just-built tree
            # so we only pay O(1) tree-prediction per ES check.
            eval_value = if is_multiclass
                predict_tree!(
                    eval_state.preds_mat,
                    tree,
                    eval_state.num_bins,
                    eval_state.cat_enc,
                    learning_rate,
                    eval_state.leaf_indices,
                )
                es_metric_fn(eval_state.y_onehot, eval_state.preds_mat)
            else
                predict_tree!(
                    eval_state.preds_vec,
                    tree,
                    eval_state.num_bins,
                    eval_state.cat_enc,
                    learning_rate,
                    eval_state.leaf_indices,
                )
                es_metric_fn(eval_state.y_vec, eval_state.preds_vec)
            end
            improved = if es_orientation == :minimize
                eval_value < best_eval_value
            else
                eval_value > best_eval_value
            end
            if improved
                best_eval_value = eval_value
                best_iter = iter
                rounds_no_improve = 0
            else
                rounds_no_improve += 1
                if rounds_no_improve >= early_stopping_rounds
                    if verbose
                        println("Early stopping at iteration $iter (best: $best_iter)")
                    end
                    trees = trees[1:best_iter]
                    break
                end
            end
        end
    end

    final_best_iteration = es_active ? best_iter : length(trees)
    final_best_score = es_active ? best_eval_value : nothing

    final_model = _build_model(
        trees, learning_rate, initial_pred, initial_pred_val, encoder,
        qf.borders, pool.feature_names, n_classes, class_labels_final,
        is_multiclass, pool.numerical_feature_indices,
        pool.categorical_feature_indices, final_best_iteration, final_best_score,
        is_custom_loss ? loss_function : nothing,
    )

    if snapshot_path !== nothing
        JLD2.save_object(snapshot_path, final_model)
        if verbose
            println("Snapshot written to $snapshot_path (final model)")
        end
    end

    return final_model
end

# Helpers extracted from `train` for readability. Pure functions, no shared
# state with the caller — return NamedTuples so the caller can read fields
# directly without nullability checks.
# Decide the task (regression / binary / multiclass), encode labels, and build
# the one-hot target matrix for multiclass. Mirrors what was previously inlined
# at the top of `train` — including the auto-promotion from binary string-loss
# to "MultiClass" when the training labels have > 2 unique values, and the
# error path for a custom `:binary` loss meeting that same data.
function _resolve_task(loss_function, label, original_class_labels, n_samples::Int)
    is_custom_loss = loss_function isa LossFunction
    custom_task = is_custom_loss ? task_type(loss_function) : nothing
    is_classification = if is_custom_loss
        custom_task === :binary || custom_task === :multiclass
    else
        uppercase(loss_function) in ("LOGLOSS", "CROSSENTROPY", "MULTICLASS", "MULTILOGLOSS")
    end
    is_multiclass = if is_custom_loss
        custom_task === :multiclass
    else
        uppercase(loss_function) in ("MULTICLASS", "MULTILOGLOSS")
    end

    class_labels = Float64[]
    y = copy(label)

    if is_classification && !is_multiclass
        class_labels = sort(unique(label))
        nc = length(class_labels)
        if nc > 2
            is_custom_loss && error(
                "Custom binary `LossFunction` was given but training labels have " *
                "$nc unique values; declare `task_type(::MyLoss) = :multiclass` " *
                "and implement the matrix-shaped contract instead.",
            )
            is_multiclass = true
            loss_function = "MultiClass"
        else
            label_map = Dict(
                class_labels[i] => Float64(i - 1) for i in eachindex(class_labels)
            )
            y = [label_map[v] for v in label]
        end
    end

    y_onehot = Matrix{Float64}(undef, 0, 0)
    n_classes = if is_multiclass
        class_labels = sort(unique(label))
        nc = length(class_labels)
        label_map = Dict(class_labels[i] => i for i in eachindex(class_labels))
        y_onehot = zeros(Float64, n_samples, nc)
        @inbounds for i in 1:n_samples
            y_onehot[i, label_map[label[i]]] = 1.0
        end
        nc
    else
        is_classification ? 2 : 0
    end

    class_labels_final =
        original_class_labels !== nothing ? original_class_labels : class_labels

    return (;
        loss_function,
        is_custom_loss,
        is_classification,
        is_multiclass,
        n_classes,
        y,
        y_onehot,
        class_labels_final,
    )
end

# Allocate the per-iteration gradient / hessian / scratch buffers. Multiclass
# variants are matrices `(n_samples, n_classes)`; the rest are vectors.
function _alloc_grad_buffers(is_multiclass::Bool, n_samples::Int, n_classes::Int)
    if is_multiclass
        return (
            g=Matrix{Float64}(undef, n_samples, n_classes),
            h=Matrix{Float64}(undef, n_samples, n_classes),
            scratch=Matrix{Float64}(undef, n_samples, n_classes),
        )
    else
        return (
            g=Vector{Float64}(undef, n_samples),
            h=Vector{Float64}(undef, n_samples),
            scratch=Vector{Float64}(undef, n_samples),
        )
    end
end

# Build the eval-pool state used by early stopping. Returns the same NamedTuple
# shape in both cases (empty matrices when inactive) so the boosting loop can
# read `eval_state.*` without nullability handling.
function _setup_eval(
    es_active::Bool,
    eval_pool,
    borders,
    encoder,
    is_multiclass::Bool,
    n_classes::Int,
    initial_pred::Vector{Float64},
    initial_pred_val::Float64,
    init_model::Union{MichiBoostModel,Nothing},
    learning_rate::Float64,
)
    if !es_active
        return (
            num_bins=Matrix{UInt16}(undef, 0, 0),
            cat_enc=Matrix{Float64}(undef, 0, 0),
            leaf_indices=Int[],
            preds_vec=Float64[],
            preds_mat=Matrix{Float64}(undef, 0, 0),
            y_vec=Float64[],
            y_onehot=Matrix{Float64}(undef, 0, 0),
        )
    end

    n_eval = eval_pool.n_samples
    num_bins = if n_numerical(eval_pool) > 0
        apply_borders(eval_pool.features_numerical, borders)
    else
        Matrix{UInt16}(undef, n_eval, 0)
    end
    cat_enc = if n_categorical(eval_pool) > 0 && encoder !== nothing
        encode_categorical(encoder, eval_pool.features_categorical)
    else
        Matrix{Float64}(undef, n_eval, 0)
    end
    leaf_indices = Vector{Int}(undef, n_eval)
    y_raw = get_label(eval_pool)

    if is_multiclass
        preds_mat = if init_model !== nothing
            _predict_raw_for_init(init_model, num_bins, cat_enc, learning_rate)
        else
            repeat(initial_pred', n_eval, 1)
        end
        class_labels = sort(unique(y_raw))
        label_map = Dict(class_labels[i] => i for i in eachindex(class_labels))
        y_onehot = zeros(Float64, n_eval, n_classes)
        @inbounds for i in 1:n_eval
            y_onehot[i, label_map[y_raw[i]]] = 1.0
        end
        return (
            num_bins=num_bins,
            cat_enc=cat_enc,
            leaf_indices=leaf_indices,
            preds_vec=Float64[],
            preds_mat=preds_mat,
            y_vec=Float64[],
            y_onehot=y_onehot,
        )
    else
        preds_vec = if init_model !== nothing
            _predict_raw_for_init(init_model, num_bins, cat_enc, learning_rate)
        else
            fill(initial_pred_val, n_eval)
        end
        return (
            num_bins=num_bins,
            cat_enc=cat_enc,
            leaf_indices=leaf_indices,
            preds_vec=preds_vec,
            preds_mat=Matrix{Float64}(undef, 0, 0),
            y_vec=y_raw,
            y_onehot=Matrix{Float64}(undef, 0, 0),
        )
    end
end

# Assemble a `MichiBoostModel` from the loop-local state. Called three times
# from `train`: each periodic snapshot during the loop, an optional final
# snapshot after the loop (covers ES truncation and non-multiple iteration
# counts), and the returned model. Centralising the 14-field constructor here
# keeps those three sites in sync.
function _build_model(
    trees,
    learning_rate::Float64,
    initial_pred::Vector{Float64},
    initial_pred_val::Float64,
    encoder,
    borders,
    feature_names,
    n_classes::Int,
    class_labels,
    is_multiclass::Bool,
    numerical_feature_indices,
    categorical_feature_indices,
    best_iteration::Int,
    best_score::Union{Float64,Nothing},
    custom_loss::Union{LossFunction,Nothing},
)
    return MichiBoostModel(
        trees,
        learning_rate,
        is_multiclass ? initial_pred : initial_pred_val,
        encoder,
        borders,
        feature_names,
        n_classes,
        class_labels,
        is_multiclass,
        numerical_feature_indices,
        categorical_feature_indices,
        best_iteration,
        best_score,
        custom_loss,
    )
end

# Compute raw (pre-link) predictions from `model.trees` on the already-quantized
# features. Used to seed the running prediction vector/matrix when continuing
# training from `init_model`. The new `learning_rate` is applied to inherited
# trees, matching how `predict(final_model, ...)` will scale them after fit
# finishes.
function _predict_raw_for_init(
    model::MichiBoostModel,
    num_bins::AbstractMatrix{UInt16},
    cat_encoded::AbstractMatrix{Float64},
    learning_rate::Float64,
)
    n = size(num_bins, 1)
    if model.is_multiclass
        ip = model.initial_pred::Vector{Float64}
        preds = repeat(ip', n, 1)
        leaf_buf = Vector{Int}(undef, n)
        @inbounds for tree in model.trees
            predict_tree!(preds, tree, num_bins, cat_encoded, learning_rate, leaf_buf)
        end
        return preds
    else
        preds = fill(model.initial_pred::Float64, n)
        leaf_buf = Vector{Int}(undef, n)
        @inbounds for tree in model.trees
            predict_tree!(preds, tree, num_bins, cat_encoded, learning_rate, leaf_buf)
        end
        return preds
    end
end
