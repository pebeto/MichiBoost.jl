@testset "MichiBoost Core" begin
    using MichiBoost: MichiBoost, Pool, MichiBoostRegressor, MichiBoostClassifier,
                      predict_proba, feature_importance, save_model, cv
    using Tables

    @testset "Pool construction from table" begin
        tbl = (floats=collect(0.5:0.5:3.0), ints=collect(1:6))
        pool = Pool(tbl)
        @test pool.n_samples == 6
        @test pool.n_features == 2
        @test MichiBoost.n_numerical(pool) == 2
        @test MichiBoost.n_categorical(pool) == 0
    end

    @testset "Pool construction with categorical features" begin
        tbl = (cat=["a", "b", "a", "c"], num=[1.0, 2.0, 3.0, 4.0])
        pool = Pool(tbl; label=[1.0, 0.0, 1.0, 0.0])
        @test pool.n_samples == 4
        @test MichiBoost.n_categorical(pool) == 1
        @test MichiBoost.n_numerical(pool) == 1
        @test pool.label == [1.0, 0.0, 1.0, 0.0]
    end

    @testset "Pool construction from matrix" begin
        X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        pool = Pool(X; label=[10.0, 20.0, 30.0])
        @test pool.n_samples == 3
        @test MichiBoost.n_numerical(pool) == 2
    end

    @testset "Pool slicing" begin
        tbl = (a=[1.0, 2.0, 3.0, 4.0], b=[5.0, 6.0, 7.0, 8.0])
        pool = Pool(tbl; label=[10.0, 20.0, 30.0, 40.0])
        sliced = MichiBoost.slice(pool, [1, 3])
        @test sliced.n_samples == 2
        @test sliced.label == [10.0, 30.0]
    end

    @testset "Regression training and prediction" begin
        train_data = [1.0 4.0 5.0 6.0; 4.0 5.0 6.0 7.0; 30.0 40.0 50.0 60.0]
        train_labels = [10.0, 20.0, 30.0]

        model = MichiBoostRegressor(; iterations=10, learning_rate=0.5, depth=2, verbose=false)
        MichiBoost.fit!(model, train_data, train_labels)

        preds = MichiBoost.predict(model, train_data)
        @test length(preds) == 3
        @test all(isfinite, preds)
    end

    @testset "Binary classification" begin
        train_data = [0.0 3.0; 4.0 1.0; 8.0 1.0; 9.0 1.0]
        train_labels = [0.0, 0.0, 1.0, 1.0]

        model = MichiBoostClassifier(; iterations=10, learning_rate=0.5, depth=2, verbose=false)
        MichiBoost.fit!(model, train_data, train_labels)

        preds = MichiBoost.predict(model, train_data)
        @test length(preds) == 4

        probs = predict_proba(model, train_data)
        @test length(probs) == 4
        @test all(0.0 .<= probs .<= 1.0)
    end

    @testset "Feature importance" begin
        train_data = [1.0 4.0 5.0 6.0; 4.0 5.0 6.0 7.0; 30.0 40.0 50.0 60.0]
        train_labels = [10.0, 20.0, 30.0]
        model = MichiBoostRegressor(; iterations=5, learning_rate=0.5, depth=2, verbose=false)
        MichiBoost.fit!(model, train_data, train_labels)
        feat_importance = feature_importance(model)
        @test length(feat_importance) > 0
        @test all(p -> p isa Pair{Symbol,Float64}, feat_importance)
    end

    @testset "Categorical features" begin
        using DataFrames
        df = DataFrame(cat1=["a", "b", "a", "b", "c", "c"],
                       num1=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                       num2=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        labels = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

        model = MichiBoostClassifier(; iterations=5, learning_rate=0.5, depth=2, verbose=false)
        MichiBoost.fit!(model, df, labels)

        preds = MichiBoost.predict(model, df)
        @test length(preds) == 6
    end

    @testset "Model save/load" begin
        train_data = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
        train_labels = [1.0, 2.0, 3.0, 4.0]

        model = MichiBoostRegressor(; iterations=5, learning_rate=0.5, depth=2, verbose=false)
        MichiBoost.fit!(model, train_data, train_labels)

        preds_before = MichiBoost.predict(model, train_data)

        tmpfile = tempname()
        save_model(model, tmpfile)
        loaded = MichiBoost.load_model(tmpfile)

        preds_after = MichiBoost.predict(loaded, Pool(train_data))
        @test preds_before ≈ preds_after
        rm(tmpfile; force=true)
    end

    @testset "Cross-validation" begin
        data = (x1=randn(20), x2=randn(20))
        labels = randn(20)
        pool = Pool(data; label=labels)

        scores = cv(pool; fold_count=2, verbose=false,
                    params=Dict("iterations" => 10, "depth" => 2, "loss_function" => "RMSE"))
        @test haskey(scores, :mean_test_loss)
        @test length(scores.test_loss) == 2
    end
end
