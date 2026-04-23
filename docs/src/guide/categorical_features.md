# Categorical Features

One of MichiBoost.jl's key features is native support for categorical variables using ordered target statistics, eliminating the need for manual one-hot encoding or label encoding.

## Automatic Detection

String columns are automatically detected as categorical:

```julia
using MichiBoost, DataFrames

df = DataFrame(
    color=["red", "blue", "red", "green"],
    size=[1.0, 2.0, 3.0, 4.0],
    material=["wood", "metal", "wood", "plastic"]
)
y = [0.0, 1.0, 0.0, 1.0]

model = MichiBoostClassifier(; iterations=50)
fit!(model, df, y)
```

In this example, `color` and `material` are automatically treated as categorical features, while `size` is treated as numerical.

## Manual Specification

You can explicitly specify which features should be treated as categorical:

```julia
using MichiBoost

# Using column indices (0-based)
pool = Pool(X; label=y, cat_features=[0, 2])

# Using column names
pool = Pool(df; label=y, cat_features=[:color, :material])

# Mix of indices and names
pool = Pool(df; label=y, cat_features=[0, :material])
```

## How It Works: Ordered Target Statistics

MichiBoost.jl uses **ordered target statistics** (also called target encoding) to handle categorical features:

1. **During Training:**
   - For each categorical value, compute statistics based on the target variable
   - Use a random permutation to prevent target leakage
   - Apply smoothing with a prior (global mean, smoothing weight fixed at `1.0`) to handle rare categories

2. **During Prediction:**
   - Use the learned statistics to encode categorical values
   - Unknown categories fall back to the prior (global mean)

This approach:

- Preserves the relationship between categories and the target
- Reduces target leakage through permutation-based encoding
- Handles high-cardinality features efficiently
- Works with missing/unknown categories

The smoothing weight is not currently exposed as a hyperparameter.

## Encoding Modes

### Ordered Target Encoding (Default)

Uses a random permutation so that each sample's encoding is computed from
preceding samples only, reducing target leakage in the categorical statistics:

```julia
model = MichiBoostRegressor(; boosting_type="Ordered")
```

### Plain Target Encoding

Computes target statistics on the entire training set:

```julia
model = MichiBoostRegressor(; boosting_type="Plain")
```

> **Note:** `boosting_type` only controls how categorical features are encoded.
> Gradient computation uses standard (plain) gradient boosting in both modes.

## Example: High-Cardinality Categorical

```julia
using MichiBoost, DataFrames, Random

Random.seed!(42)

# Create dataset with high-cardinality categorical feature
n = 1000
df = DataFrame(
    user_id=string.("user_", rand(1:200, n)),  # 200 unique users
    product_id=string.("prod_", rand(1:50, n)),  # 50 unique products
    price=rand(10.0:100.0, n),
    quantity=rand(1:10, n)
)

# Target: total purchase amount with some noise
y = df.price .* df.quantity .+ randn(n) .* 10

# Train model (user_id and product_id auto-detected as categorical)
model = MichiBoostRegressor(;
    iterations=200,
    learning_rate=0.05,
    depth=4
)

fit!(model, df, y)

# Make predictions
predictions = predict(model, df)
```

## Handling Unknown Categories

When predicting on new data with previously unseen categorical values:

```julia
# Training data
train_df = DataFrame(
    color=["red", "blue", "green"],
    value=[1.0, 2.0, 3.0]
)
y_train = [10.0, 20.0, 30.0]

model = MichiBoostRegressor(; iterations=50)
fit!(model, train_df, y_train)

# Test data with unknown category "yellow"
test_df = DataFrame(
    color=["red", "yellow", "blue"],
    value=[1.5, 2.5, 3.5]
)

# Unknown categories use the global prior (mean target value)
predictions = predict(model, test_df)
```

