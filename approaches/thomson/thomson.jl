using DataFrames, Statistics, LinearAlgebra
using Random, Distributions, Plots, StatsBase

ProjDir = @__DIR__
#Random.seed!(12591)

include(joinpath(ProjDir,
    "./optimal_price_probabilities/optimal_price_probabilities.jl"))

# parameters
prices = [1.99, 2.49, 2.99, 3.49, 3.99, 4.49]

# parameters of the true (unknown) demand model
true_intc = 50
true_slope = 7

# prior distribution for each price
θ = [DataFrame() for i in 1:length(prices)]
for (i,p) in enumerate(prices)
    append!(θ[i],
        DataFrame(
            :price => p,
            :α => 30.0, 
            :β => 1.0,
            :mean => 30.0
        )
    )
end

demands = true_intc .- true_slope * prices
probs = optimal_price_probabilities(prices, demands, 60)

α_demand = 50
β_demand = 7

# distribution for each price
θ = DataFrame()
for p in prices
    append!(θ,
        DataFrame(
            :price => p,
            :α => 30.0, 
            :β => 1.0,
            :mean => 30.0
        )
    )
end

function sample_demands_from_model(theta)
    [rand(Gamma(v[:α], 1/v[:β]), 1)[1] for v in eachrow(θ)]
end

function sample_demand(price, demand_a=α_demand, demand_b=β_demand)
    demand = demand_a - demand_b * price
    rand(Poisson(demand), 1)[1]
end

function simulate(;T=T, verbose=false, report_final=true)
    # history for each price
    history = DataFrame()

    for t in 1:T
        demands = sample_demands_from_model(θ)
        price_probs = optimal_price_probabilities(prices, demands, 60)
        price_index_t = sample(1:length(prices), price_probs)
        price_t = prices[price_index_t]
        demand_t = sample_demand(price_t)
        revenue = round(demand_t * price_t, digits=2)

        θ[price_index_t, :α] += demand_t
        θ[price_index_t, :β] += 1
        θ[price_index_t, :mean] = θ[price_index_t, :α] / θ[price_index_t, :β]

        if verbose || (report_final && t == T)
            println("\nPrices = $prices")
            println("Demands (step=$t) = $(round.(demands, digits=2))")
            println("Price probs (step=$t) = $(price_probs)")
            println("Price index (step=$t) = $(price_index_t)")
            print("Selected price (step=$t) = $price_t =>")
            println(" demand = $demand_t, revenue = $revenue")
        end

        v = θ[price_index_t, :]
        append!(history,
            DataFrame(
                :step => t,
                :index => price_index_t,
                :price => v[:price],
                :α => v[:α], 
                :β => v[:β]
            )
        )       
    end

    (θ, history)
end

(θ, history) = simulate(T=100)
#history[history.step .== T, :] |> display
println()
θ |> display

x = 0.0:0.01:60.0
figs = Vector{Plots.Plot{Plots.GRBackend}}(undef, 1)
figs[1] = plot(xlabel="Price", ylabel="Demand pdf")
for v in eachrow(θ)
    figs[1] = plot!(figs[1], x, pdf.(Gamma(v[:α], 1/v[:β]), x),
        label="price=$(v[:price])")
end
plot(figs..., layout=(1,1))
savefig(joinpath(@__DIR__, "thomson.png"))

