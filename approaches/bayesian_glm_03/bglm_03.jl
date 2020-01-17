using CSV, DataFrames, StatsPlots, StanSample
using Distributions, Statistics, Random

ProjDir = @__DIR__

# Simulate the data

#Random.seed!(123)

countries = ["Guatamala", "Panama", "Kenya", "Ethiopia", "Papua New Guinea"]
brands = [4, 3, 4, 5, 1]
intercepts = [500, 300, 600, 700, 900] # Quantity at cheapest price_point
slopes = [-6, -8, -10, -15, -20] # Assuming constant price elasticity
price_points = [12.5, 12.75, 13.02, 14.20, 15.15] # Selected price points
costs = [4.0, 4.2, 4.6, 4.7, 5.2]
channels = ["retail", "wholesale", "shop"]

df = DataFrame()
for n in 1:1500
	yr = 2019
	wk = rand(1:52)
	c_ind = rand(1:length(countries))
	p_ind = rand(1:length(price_points))
	c = countries[c_ind]							# country
	b = rand(1:brands[c_ind])						# brand
	ch = rand(channels)								# channel
	cst = rand(Normal(costs[c_ind], 0.1))			# cost
	p = price_points[p_ind]							# ppu
	if rand(Bernoulli(0.2))
		p = rand(Normal(p, 0.07))
	end
	p = round(p, digits=2)
	q = intercepts[c_ind] + slopes[c_ind] * p_ind	# quantity sold
	if rand(Bernoulli(0.2))
		q = round(rand(Normal(q, 30)))
	end
	append!(df, DataFrame(
		year=yr, week=wk, 
		country=c, brand=b, channel=ch,
		quantity=q, ppu=p)
	)
end

df[!, :ppu_l] = log.(df[:, :ppu])
df[!, :quantity_l] = log.(df[:, :quantity])

# Clean up

df = df[df.ppu_l .> 0.04, :]
df = df[df.quantity_l .> 0.0, :]

CSV.write(joinpath(ProjDir, "data", "data.csv"), df)

bglm = "
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
  a ~ normal(2.0, 10.0);		// intercept positive
  b ~ normal(-2.0, 10.0);		// demand drops as price increases
  sigma ~ uniform(0, 15);
  mu = a + b * p;
  q ~ normal(mu, sigma);
}
// generated quantities {
//    vector[N] mu_pred;
//    mu_pred = exp(p);
//  }
";

mean_q_l = mean(df[:, :quantity_l])
mean_p_l = mean(df[:, :ppu_l])
data = Dict(
  :N => size(df, 1),
  :q => df[:, :quantity_l] .- mean_q_l,
  :p => df[:, :ppu_l] .- mean_p_l
)

p1 = scatter(df[:, :ppu], df[:, :quantity],
	xlabel="ppu", ylabel="quantity sold", leg=false)
p2 = scatter(data[:p], data[:q],
	xlabel="log(ppu)", ylabel="log(quantity sold)", leg=false)
plot(p1, p2, layout=(2, 1))
savefig(joinpath(ProjDir, "plots", "data.png"))

sm = SampleModel("bglm", bglm)

rc = stan_sample(sm, data=data)

if success(rc)
	chns = read_samples(sm)
	plot(chns)
	savefig(joinpath(ProjDir, "plots", "chains.png"))

	dfs = DataFrame(chns)
	show(chns)

	p = -0.1:0.01:0.15
	p1 = scatter(data[:p], data[:q],
		xlims=(-0.1, 0.15), ylims=(-1, 1),
		xlabel="ppu_l", ylabel="quantity_l sold", leg=false)
	a_mean = mean(dfs[:, :a])
	b_mean = mean(dfs[:, :b])
	mu_q = a_mean .+ b_mean .* p
	plot!(p1, p, mu_q)
	savefig(joinpath(ProjDir, "plots", "regression.png"))

	price_points_l = log.(price_points) .- mean(log.(price_points))
	scatter(price_points_l, a_mean .+ b_mean .* price_points_l,
		xlims = (-0.1, 0.15), ylims= (-0.1, 0.1), 
		leg=false, xlabel="log(ppu)", ylabel="log(quantity sold)")
	selected_rows = rand(1:size(dfs, 1), 75)
	for row in eachrow(dfs[selected_rows,:])
		plot!(p, row[:a] .+ row[:b] .* p, color=:gray)
	end
	scatter!(price_points_l, a_mean .+ b_mean .* price_points_l)
	savefig(joinpath(ProjDir, "plots", "sampling.png"))
end
