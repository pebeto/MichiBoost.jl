function train(
    pool::Pool;
    iterations::Int=1000,
    learning_rate::Float64=0.03,
    depth::Int=6,
    l2_leaf_reg::Float64=3.0,
    loss_function::String="RMSE",
    border_count::Int=254,
    min_data_in_leaf::Int=1,
    random_seed::Union{Int,Nothing}=nothing,
    verbose::Bool=true,
    rsm::Float64=1.0,
    early_stopping_rounds::Union{Int,Nothing}=nothing,
    eval_pool::Union{Pool,Nothing}=nothing,
    boosting_type::String="Ordered",
    kwargs...,
)
    rng = random_seed !== nothing ? MersenneTwister(random_seed) : MersenneTwister()

    label = get_label(pool)
    n_samples = pool.n_samples
    n_num = n_numerical(pool)
    n_cat = n_categorical(pool)

    qf = quantize_features(pool.features_numerical; border_count)

    is_classification = uppercase(loss_function) in
                        ("LOGLOSS", "CROSSENTROPY", "MULTICLASS", "MULTILOGLOSS")
    is_multiclass = uppercase(loss_function) in ("MULTICLASS", "MULTILOGLOSS")

    class_labels = Float64[]
    y = copy(label)
    original_class_labels = pool.label_classes

    if is_classification && !is_multiclass
        class_labels = sort(unique(label))
        n_classes = length(class_labels)
        if n_classes > 2
            is_multiclass = true
            loss_function = "MultiClass"
        else
            label_map = Dict(
                class_labels[i] => Float64(i - 1) for i in eachindex(class_labels)
            )
            y = [label_map[v] for v in label]
        end
    end

    if is_multiclass
        class_labels = sort(unique(label))
        n_classes = length(class_labels)
        label_map = Dict(class_labels[i] => i for i in eachindex(class_labels))
        y_indices = [label_map[v] for v in label]
        y_onehot = zeros(Float64, n_samples, n_classes)
        for i in 1:n_samples
            y_onehot[i, y_indices[i]] = 1.0
        end
    else
        n_classes = is_classification ? 2 : 0
    end

    class_labels_final = if original_class_labels !== nothing
        original_class_labels
    else
        class_labels
    end

    # Categorical encoding
    permutation = randperm(rng, n_samples)
    cat_encoded, encoder = if n_cat > 0
        if boosting_type == "Ordered"
            compute_ordered_target_stats(
                pool.features_categorical,
                y,
                permutation;
                alpha=1.0,
            )
        else
            plain_target_encode(pool.features_categorical, y)
        end
    else
        (
            Matrix{Float64}(undef, n_samples, 0),
            OrderedTargetEncoder(mean(y), 1.0, Dict{UInt32,Tuple{Float64,Int}}[]),
        )
    end

    lf = make_loss(loss_function; n_classes)

    # Initial predictions
    if is_multiclass
        initial_pred = initial_prediction(lf, y_onehot)
        predictions = repeat(initial_pred', n_samples, 1)
    else
        initial_pred_val = initial_prediction(lf, y)
        predictions = fill(initial_pred_val, n_samples)
    end

    trees = is_multiclass ? SymmetricTreeMultiClass[] : SymmetricTree[]
    best_eval_loss, rounds_no_improve, best_iter = Inf, 0, 0
    sample_indices = collect(1:n_samples)

    weights = pool.weight !== nothing ? pool.weight : ones(Float64, n_samples)

    max_leaves = 1 << depth
    # Categorical encoded values can have up to n_samples distinct values,
    # so the buffer must accommodate both numerical bins and categorical ranks.
    max_bins = max(border_count + 2, n_samples + 1)
    nt = Threads.maxthreadid()
    buffers = is_multiclass ?
        [SplitBuffersMC(max_leaves, max_bins, n_classes, n_samples) for _ in 1:nt] :
        [SplitBuffers(max_leaves, max_bins, n_samples) for _ in 1:nt]

    # Pre-compute sorted unique encoded values per categorical feature — fixed for the run.
    cat_sorted_vals = [sort(unique(cat_encoded[:, j])) for j in 1:n_cat]

    hist_cache = is_multiclass ?
        HistCacheMC(max_leaves, qf.n_bins, cat_sorted_vals, n_classes) :
        HistCache(max_leaves, qf.n_bins, cat_sorted_vals)

    leaf_indices = Vector{Int}(undef, n_samples)

    # Early-stopping eval state — set up once, updated incrementally each
    # round so the ES check is O(1) per iteration instead of O(T).  The
    # original _evaluate_loss re-predicted all trees on every call, which
    # made ES O(T²) across a run.
    es_active = early_stopping_rounds !== nothing && eval_pool !== nothing
    eval_num_bins = Matrix{UInt16}(undef, 0, 0)
    eval_cat_enc  = Matrix{Float64}(undef, 0, 0)
    eval_preds_vec = Float64[]
    eval_preds_mat = Matrix{Float64}(undef, 0, 0)
    eval_y_vec = Float64[]
    eval_y_onehot = Matrix{Float64}(undef, 0, 0)
    eval_leaf_indices = Int[]
    if es_active
        n_eval = eval_pool.n_samples
        eval_num_bins = n_numerical(eval_pool) > 0 ?
            apply_borders(eval_pool.features_numerical, qf.borders) :
            Matrix{UInt16}(undef, n_eval, 0)
        eval_cat_enc = n_categorical(eval_pool) > 0 && encoder !== nothing ?
            encode_categorical(encoder, eval_pool.features_categorical) :
            Matrix{Float64}(undef, n_eval, 0)
        eval_leaf_indices = Vector{Int}(undef, n_eval)
        eval_y_raw = get_label(eval_pool)
        if is_multiclass
            eval_preds_mat = repeat(initial_pred', n_eval, 1)
            es_class_labels = sort(unique(eval_y_raw))
            es_label_map = Dict(es_class_labels[i] => i for i in eachindex(es_class_labels))
            eval_y_onehot = zeros(Float64, n_eval, n_classes)
            for i in 1:n_eval
                eval_y_onehot[i, es_label_map[eval_y_raw[i]]] = 1.0
            end
        else
            eval_preds_vec = fill(initial_pred_val, n_eval)
            eval_y_vec = eval_y_raw
        end
    end

    # Buffers reused across every boosting round — without this the loop
    # allocates an O(n × n_classes) matrix (or O(n) vector) 4-6 times per
    # iteration for gradient / hessian / softmax temporaries, which on
    # n=40k × k=7 adds up to ~100 MB / iter.
    grads_buf = is_multiclass ?
        Matrix{Float64}(undef, n_samples, n_classes) :
        Vector{Float64}(undef, n_samples)
    hess_buf = is_multiclass ?
        Matrix{Float64}(undef, n_samples, n_classes) :
        Vector{Float64}(undef, n_samples)
    scratch_buf = is_multiclass ?
        Matrix{Float64}(undef, n_samples, n_classes) :
        Vector{Float64}(undef, n_samples)
    use_refine = lf isa MAELoss
    refine_buf = use_refine ? Vector{Float64}(undef, n_samples) : Float64[]

    for iter in 1:iterations
        if is_multiclass
            gradient_hessian!(grads_buf, hess_buf, lf, y_onehot, predictions, scratch_buf)
            @. grads_buf *= weights
            @. hess_buf  *= weights
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
            predict_tree!(predictions, tree, qf.bins, cat_encoded, learning_rate, leaf_indices)
        else
            gradient_hessian!(grads_buf, hess_buf, lf, y, predictions, scratch_buf)
            @. grads_buf *= weights
            @. hess_buf  *= weights
            # MAE is non-smooth: surrogate gradients (±1) drive split-finding,
            # but leaf values must come from the residual weighted median —
            # otherwise each round can shift predictions by at most
            # learning_rate × 1, causing severe underfitting.
            if use_refine
                @. refine_buf = y - predictions
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
                leaf_refine_values  = use_refine ? refine_buf : nothing,
                leaf_refine_weights = use_refine ? weights    : nothing,
            )
            push!(trees, tree)
            predict_tree!(predictions, tree, qf.bins, cat_encoded, learning_rate, leaf_indices)
        end

        if verbose && (iter % max(1, iterations ÷ 10) == 0 || iter == 1 || iter == iterations)
            train_loss = if is_multiclass
                loss(lf, y_onehot, predictions)
            else
                loss(lf, y, predictions)
            end
            println("Iteration $iter/$iterations, train loss: $(round(train_loss; digits=6))")
        end

        if es_active
            # Extend the running eval predictions with the just-built tree
            # so we only pay O(1) tree-prediction per ES check.
            if is_multiclass
                predict_tree!(eval_preds_mat, tree, eval_num_bins, eval_cat_enc,
                              learning_rate, eval_leaf_indices)
                eval_loss = loss(lf, eval_y_onehot, eval_preds_mat)
            else
                predict_tree!(eval_preds_vec, tree, eval_num_bins, eval_cat_enc,
                              learning_rate, eval_leaf_indices)
                eval_loss = loss(lf, eval_y_vec, eval_preds_vec)
            end
            if eval_loss < best_eval_loss
                best_eval_loss = eval_loss
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

    return MichiBoostModel(
        trees,
        learning_rate,
        is_multiclass ? initial_pred : initial_pred_val,
        loss_function,
        encoder,
        qf.borders,
        pool.feature_names,
        n_classes,
        class_labels_final,
        is_multiclass,
        pool.numerical_feature_indices,
        pool.categorical_feature_indices,
    )
end

