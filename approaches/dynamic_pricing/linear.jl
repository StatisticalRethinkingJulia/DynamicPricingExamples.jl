using LinearAlgebra, ForwardDiff, Optim
using DataFrames, Plots, Distributions, Statistics

ProjDir = @__DIR__
cd(ProjDir)

include("./dynamic_pricing/price_demand_models.jl")
include("./demand_curve/select_optimal_demand_curve.jl")
include("./intervals/intervals.jl")
include("./utils/logx.jl")
include("./actual_demand/actual_demand_curve.jl")

# Generate demand hypothesis

a_range = range(-1.2, -0.6, length=4)
b_range = range(60, 70, length=4)
h_vec = generate_demand_hypothesis(a_range, b_range)
println("Price demand hypotheses (see also fig_01.png):")
display(h_vec)
println()

plot(xlim=(0, 60), ylim=(0, 80),
    xlabel="Price", ylabel="Demand",
    title="Price Demand hypotheses with optimal price points."
)
x = range(0, 60, length=101)
for i in 1:size(h_vec, 1)
	plot!(x, h_vec[i, :d].(x), color=:red, leg=false)
end
scatter!(h_vec[:, :p_opt], h_vec[:, :d_opt], m=(:cross))
savefig(joinpath(ProjDir, "fig_01.png"))

# Emperical demand curve

println()
curve = emperical_demand_curve(30, 60, 20)
println("Emperical price demand curve:")
display(curve)

# Intervals for price updates

T = 24 * 1                       # time step is one hour, flash offering for 1 day 
m = 4                            # not more than 4 price updates

t_mask = intervals(m, T, 2)      # intervals for price lvels
p = h_vec[1, :p_opt]             # initial price

println("\nSelected intervals:")
display(t_mask')
println("\nInitial price = $p\n")

# Do a first iteration in selecting demand hypothesis

(price, row, hist_d) = select_optimal_demand_curve(t_mask, p)
println()
display((price, row))
println()
display((hist_d))

plot(xlim=(0, 60), ylim=(0, 80), xlabel="Price", ylabel="Demand")
plot!(curve[:, :prices], curve[:, :demands],
    xlabel="price", ylabel="demand", line=(:dot, :gray))
x = range(0, 60, length=101)
plot!(x, h_vec[row, :d].(x), color=:red, leg=false)
savefig(joinpath(ProjDir, "fig_02.png"))

#=
colors = ['#8EA604', '#F5BB00', '#EC9F05', '#D76A03']

def visualize_snapshot(t):
    fig.clear()
    
    plt.subplot(2, 1, 1)
    plt.xlabel('Price')
    plt.ylabel('Demand')
    plt.title('Price-demand curve fitting')
    plt.plot(curve[0], curve[1], 'k:') 
    for i in range(0, t):
        plt.plot(prices, list(map(np.array(history)[i, 2], prices)), 'k-') 
    plt.plot(np.array(history)[0:t,0], np.array(history)[0:t, 1], 'rx') 
    
    plt.subplot(2, 1, 2)
    plt.xlabel('Time')
    plt.ylabel('Demand/price')
    plt.title('Realized demand and price')
    plt.plot(range(0, t+1), np.array(history)[0:t+1,0], 'r-') 
    bars = plt.bar(range(0, T-1), np.pad(np.array(history)[0:t+1, 1], (0, T-2-t), 'constant'), 0.35)
    for i in range(0, t+1):
        bars[i].set_color(colors[t_mask[i]])
    
fig = plt.figure(figsize=(10, 8))
plt.subplots_adjust(hspace=0.5)
visualize_snapshot(T-2)               # visualize the state in the end of the simulation
plt.show()
=#

