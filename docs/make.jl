using Documenter, DynamicPricingExamples

makedocs(
    modules = [DynamicPricingExamples],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Rob J Goedman",
    sitename = "DynamicPricingExamples.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/goedman/DynamicPricingExamples.jl.git",
    push_preview = true
)
