using TensorXD
using Documenter

DocMeta.setdocmeta!(TensorXD, :DocTestSetup, :(using TensorXD); recursive=true)

makedocs(;
    modules=[TensorXD],
    authors="PhysicsCodesLab",
    repo="https://github.com/PhysicsCodesLab/TensorXD.jl/blob/{commit}{path}#{line}",
    sitename="TensorXD.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://PhysicsCodesLab.github.io/TensorXD.jl",
        assets=String[], mathengine = MathJax()
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
    repo="github.com/PhysicsCodesLab/TensorXD.jl",
    devbranch="master",
)
