using CSV, DataFrames, StatsPlots, StanSample, MCMCChains
using Distributions, Statistics, Random

ProjDir = @__DIR__

# Simulate the data

#Random.seed!(123)

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
	real<lower=0> a;
	real b;
	real<lower=0> sigma;
}
model {
	vector[N] mu;            		// mu is a vector
	a ~ normal(2.0, 10.0);			// intercept positive
	b ~ normal(-2.0, 10.0);			// demand drops as price increases
	sigma ~ uniform(0, 50);
	mu = a + b * p;
	q ~ normal(mu, sigma);
}
generated quantities {
	vector[N] mu_pred;
	mu_pred = exp(a + b * p);
}
";

data = Dict(
  :N => size(df, 1),
  :q => df[:, :quantity_l],
  :p => df[:, :ppu_l]
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
	chn = read_samples(sm; output_format=:mcmcchains)

	chns = set_section(chn, 
	    Dict(
	      :parameters => ["a", "b", "sigma"],
	      :mu => ["mu_pred.$i" for i in 1:length(ppu)],
	      :internals => names(chn, [:internals])
	    )
	 )

	plot(chns)
	savefig(joinpath(ProjDir, "plots", "chains.png"))
	dfsa = DataFrame(chn)
	density(dfsa[:, Symbol("mu_pred.1")], label="mu_pred.1")
	for i in 2:3:N
		density!(dfsa[:, Symbol("mu_pred.$i")], label="mu_pred.$i")
	end
	savefig(joinpath(ProjDir, "plots", "density.png"))

	dfs = DataFrame(chns)
	describe(chns, sections=[:parameters, :mu]) |> display

	p = 3.4:0.01:4.0
	ps = collect(p)
	p1 = scatter(data[:p], data[:q],
		xlims=(3.3, 3.95), ylims=(3.1, 4.05),
		xlabel="ppu_l", ylabel="quantity_l sold", leg=false)
	a_mean = mean(dfs[:, :a])
	b_mean = mean(dfs[:, :b])
	mu_q = a_mean .+ b_mean .* p
	plot!(p1, p, mu_q)
	savefig(joinpath(ProjDir, "plots", "regression.png"))

	price_points_l = data[:p]
	plot(xlabel="Price", ylabel="Quantity", leg=false,
		xlims=(0, 100), ylims=(0, 100)
	)
	selected_rows = rand(1:size(dfs, 1), 75)
	for row in eachrow(dfs[selected_rows,:])
		qs = exp.(row[:a] .+ row[:b] .* (ps .- mean(ps)))
		plot!(exp.(p), qs, color=:gray)
	end
	qqs = exp.(a_mean .+ b_mean .* (price_points_l .- mean(ps)))
	scatter!(exp.(price_points_l), qqs)
	savefig(joinpath(ProjDir, "plots", "sampling.png"))

	scatter(dfs[:, :a], dfs[:, :b],
		xlabel="a", ylabel="b", leg=false)
	savefig(joinpath(ProjDir, "plots", "correlations.png"))



end
