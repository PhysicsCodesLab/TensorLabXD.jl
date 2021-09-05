using TensorXD
using Documenter

makedocs(;
    modules=[TensorXD],
    authors="PhysicsCodesLab",
    sitename="TensorXD.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        mathengine = MathJax()
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => ["man/intro.md", "man/tutorial.md", "man/categories.md",
                    "man/spaces.md", "man/sectors.md", "man/tensors.md"],
        "Library" => [],
        "Index" => ["index/index.md"]
    ],
)

deploydocs(;
    repo="github.com/PhysicsCodesLab/TensorXD.jl.git",
    devbranch="master",
)
