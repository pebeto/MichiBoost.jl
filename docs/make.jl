push!(LOAD_PATH, "../src/")
using Documenter
using MichiBoost

makedocs(;
    modules=[MichiBoost],
    authors="Jose Esparza <joseesparzadc@gmail.com>",
    sitename="MichiBoost.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pebeto.github.io/MichiBoost.jl",
        edit_link="main",
        repolink="https://github.com/pebeto/MichiBoost.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "guide/regression.md",
            "guide/classification.md",
            "guide/categorical_features.md",
            "guide/hyperparameters.md",
            "guide/advanced.md",
        ],
        "API Reference" => [
            "api/models.md",
            "api/data.md",
            "api/training.md",
            "api/prediction.md",
            "api/utilities.md",
        ],
    ],
    checkdocs=:none,
    warnonly=true,
    remotes=nothing,
)

deploydocs(;
    repo="github.com/pebeto/MichiBoost.jl",
    devbranch="main",
)
