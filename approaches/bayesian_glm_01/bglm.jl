using CSV, DataFrames, StatsPlots, StanSample, Statistics

ProjDir = @__DIR__

old_df = CSV.read(joinpath(ProjDir, "data", "data_final.csv"), delim=' ')

df = DataFrame()
for row in eachrow(old_df)
	vals = split(row[1], "\t")
	yr = Meta.parse(row[1][1:4])
	wk = Meta.parse(row[1][5:6])
	s = Meta.parse(vals[2])
	p = Meta.parse(vals[3])
	q = Meta.parse(vals[4])
	append!(df, DataFrame(year=yr, week=wk, store=s, quantity=q, price=p))
end

df[!, :ppu] = df[:, :price] ./ df[:, :quantity]
df[!, :ppu_l] = log.(df[:, :ppu])
df[!, :quantity_l] = log.(df[:, :quantity])

# Clean up

df = df[df.ppu_l .> 0.04, :]
df = df[df.quantity_l .> 0.0, :]

CSV.write(joinpath(ProjDir, "data", "data.csv"), df)

p1 = scatter(df[:, :ppu], df[:, :quantity],
	xlabel="ppu", ylabel="quantity sold")
p2 = scatter(df[:, :ppu_l], df[:, :quantity_l],
	xlabel="log(ppu)", ylabel="log(quantity sold)")
plot(p1, p2, layout=(2, 1))
savefig(joinpath(ProjDir, "plots", "data.png"))

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
";

data = Dict(
  :N => size(df, 1),
  :q => df[:, :quantity_l],
  :p => df[:, :ppu_l]
)

sm = SampleModel("bglm", bglm)

rc = stan_sample(sm, data=data)

if success(rc)
	chns = read_samples(sm)
	plot(chns)
	savefig(joinpath(ProjDir, "plots", "chains.png"))

	dfs = DataFrame(chns)
	show(chns)

	p = 0.6:0.01:2.0
	p1 = scatter(df[:, :ppu_l], df[:, :quantity_l],
		xlabel="ppu_l", ylabel="quantity_l sold")
	a_mean = mean(dfs[:, :a])
	b_mean = mean(dfs[:, :b])
	mu_q = a_mean .+ b_mean .* p
	plot!(p1, p, mu_q)
	savefig(joinpath(ProjDir, "plots", "regression.png"))

	price_points_l = [1.05, 1.35, 1.5, 1,6, 1.78]
	scatter(price_points_l, a_mean .+ b_mean .* price_points_l,
		xlims = (0.6, 2.0), ylims= (0, 7), leg=false,
		xlabel="log(ppu)", ylabel="log(quantity sold)")
	for row in eachrow(dfs)
		plot!(p, row[:a] .+ row[:b] .* p, color=:gray)
	end
	scatter!(price_points_l, a_mean .+ b_mean .* price_points_l)
	savefig(joinpath(ProjDir, "plots", "sampling.png"))
end
