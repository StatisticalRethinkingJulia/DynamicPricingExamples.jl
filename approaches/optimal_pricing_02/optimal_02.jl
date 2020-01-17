using CSV, DataFrames, StatsPlots, StanSample
using Distributions, Statistics, Random

ProjDir = @__DIR__

# Simulate the data

#Random.seed!(123)


ppu = [30.0, 35.0, 40.0, 45.0, 50.0]
quantity = [53, 45, 25, 26, 25]

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
	mu = a + b * p - mean(p);
	q ~ normal(mu, sigma);
}
";

mean_q_l = mean(df[:, :quantity_l])
mean_p_l = mean(df[:, :ppu_l])
data = Dict(
  :N => size(df, 1),
  :q => df[:, :quantity_l], #.- mean_q_l,
  :p => df[:, :ppu_l] # .- mean_p_l
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
	chns = read_samples(sm)
	plot(chns)
	savefig(joinpath(ProjDir, "plots", "chains.png"))

	dfs = DataFrame(chns)
	show(chns)

	p = -0.4:0.01:0.5
	p1 = scatter(data[:p], data[:q],
		xlims=(-0.4, 0.5), ylims=(-0.4, 0.6),
		xlabel="ppu_l", ylabel="quantity_l sold", leg=false)
	a_mean = mean(dfs[:, :a])
	b_mean = mean(dfs[:, :b])
	mu_q = a_mean .+ b_mean .* p
	plot!(p1, p, mu_q)
	savefig(joinpath(ProjDir, "plots", "regression.png"))

	price_points_l = data[:p]
	scatter(price_points_l, a_mean .+ b_mean .* price_points_l,
		xlims = (-0.3, 0.3), ylims= (-0.3, 0.5), 
		leg=false, xlabel="log(ppu)", ylabel="log(quantity sold)")
	selected_rows = rand(1:size(dfs, 1), 75)
	for row in eachrow(dfs[selected_rows,:])
		plot!(p, row[:a] .+ row[:b] .* p, color=:gray)
	end
	scatter!(price_points_l, a_mean .+ b_mean .* price_points_l)
	savefig(joinpath(ProjDir, "plots", "sampling.png"))

	scatter(dfs[:, :a], dfs[:, :b],
		xlims = (-0.1, 1.0), ylims= (-10.0, 10.0), 
		xlabel="a", ylabel="b", leg=false)
	savefig(joinpath(ProjDir, "plots", "correlations.png"))
end
