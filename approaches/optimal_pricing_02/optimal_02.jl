using CSV, DataFrames, StatsPlots, StanSample
using Distributions, Statistics, Random

ProjDir = @__DIR__

# Simulate the data

Random.seed!(123)

N = 15
obspairs = ((30.0, 53), (35.0, 45), (40.0, 28), (45.0, 26), (50.0, 25))

ppu = []; quantity = []
for i in 1:N in 
	obs = rand(obspairs, 1)
	append!(ppu, [rand(Normal(obs[1][1], 1.0))])
	append!(quantity, [rand(Normal(obs[1][2], 1.0))])
end

df = DataFrame()
df[!, :ppu] = ppu
df[!, :quantity] = quantity
df[!, :ppu_l] = log.(ppu)
df[!, :quantity_l] = log.(quantity)

CSV.write(joinpath(ProjDir, "data", "data.csv"), df)

opt_01 = "
data {
	int<lower=0> N;
	vector[N] q; // Outcome
	vector[N] p; // Predictor
}
parameters {
	real a;
	real b;
	real<lower=0> sigma;
}
model {
	vector[N] mu;            		// mu is a vector
	a ~ cauchy(0.0, 5.0);			// intercept positive
	b ~ cauchy(0.0, 5.0);			// demand drops as price increases
	sigma ~ uniform(0, 15);
	mu = a + b * (p - mean(p));
	q ~ normal(mu, sigma);
}
";

mean_q_l = mean(df[:, :quantity_l])
mean_p_l = mean(df[:, :ppu_l])
data = Dict(
  :N => size(df, 1),
  :q => df[:, :quantity_l], # .- mean_q_l,
  :p => df[:, :ppu_l]		# .- mean_p_l
)

p1 = scatter(df[:, :ppu], df[:, :quantity],
	xlabel="ppu", ylabel="quantity sold", leg=false)
p2 = scatter(data[:p], data[:q],
	xlabel="log(ppu)", ylabel="log(quantity sold)", leg=false)
plot(p1, p2, layout=(2, 1))
savefig(joinpath(ProjDir, "plots", "data.png"))

sm = SampleModel("opt_01", opt_01)

rc = stan_sample(sm, data=data)

if success(rc)
	chns = read_samples(sm; output_format=:mcmcchains)
	show(chns)
	plot(chns)
	savefig(joinpath(ProjDir, "plots", "chains.png"))

	dfs = read_samples(sm; output_format=:dataframe)
	
	dfsa = DataFrame(chns)
	#density(dfsa[:, Symbol("mu_pred.1")], label="mu_pred.1")
	#for i in 2:3:N
	#	density!(dfsa[:, Symbol("mu_pred.$i")], label="mu_pred.$i")
	#end
	#savefig(joinpath(ProjDir, "plots", "density.png"))


	p = 3.4:0.01:4.0
	ps = collect(p)
	p1 = scatter(data[:p], data[:q],
		#xlims=(3.4, 4.0), ylims=(3.0, 4.0),
		xlabel="ppu_l", ylabel="quantity_l sold", leg=false)
	a_mean = mean(dfs[:, :a])
	b_mean = mean(dfs[:, :b])
	mu_q = a_mean .+ b_mean .* (p .- mean(p))
	plot!(p1, p, mu_q)
	savefig(joinpath(ProjDir, "plots", "regression.png"))

	price_points_l = data[:p]
	plot(xlabel="Price", ylabel="Quantity", leg=false)
	selected_rows = rand(1:size(dfs, 1), 75)
	for row in eachrow(dfs[selected_rows,:])
		qs = exp.(row[:a] .+ row[:b] .* (ps .- mean(ps)))
		plot!(exp.(p), qs, color=:gray)
	end
	qqs = exp.(a_mean .+ b_mean .* (price_points_l .- mean(ps)))
	scatter!(exp.(price_points_l), qqs)
	savefig(joinpath(ProjDir, "plots", "sampling.png"))

	scatter(dfs[:, :a], dfs[:, :b],
		#xlims = (-0.1, 1.0), ylims= (-10.0, 10.0), 
		xlabel="a", ylabel="b", leg=false)
	savefig(joinpath(ProjDir, "plots", "correlations.png"))
end
