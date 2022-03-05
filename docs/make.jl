using SCS
using Documenter

DocMeta.setdocmeta!(SCS, :DocTestSetup, :(using SCS); recursive=true)

makedocs(;
    modules=[SCS],
    authors="Emmanuel Rialland <Emmanuel.Rialland@gmail.com>, Alba Intelligence <Alba.Intelligence@gmail.com>",
    repo="https://github.com/Emmanuel-R8/SCS.jl/blob/{commit}{path}#{line}",
    sitename="SCS.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
