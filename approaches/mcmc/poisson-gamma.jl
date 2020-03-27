using StanSample, MCMCChains, StatsPlots

ProjDir = @__DIR__

#=
with pm.Model() as m:
    d = pm.Gamma('theta', 1, 1)          # prior distribution
    pm.Poisson('d0', d, observed = d0)   # likelihood
    samples = pm.sample(10000)           # draw samples from the posterior
=#

demand_model = "
data {
  int<lower=1> N;
  int<lower=0> d[N];
}
parameters {
  real<lower=0> theta;
}
model {
  theta ~ gamma(1,1);
  d ~ poisson(theta);
}
";

d = [20, 28, 24, 20, 23]        # observed demand samples
demand_data = Dict(
  :N => length(d),
  :d => d
)

# Keep tmpdir across multiple runs to prevent re-compilation
tmpdir = joinpath(@__DIR__, "tmp")

sm = SampleModel("demand-price-model", demand_model)

rc = stan_sample(sm; data=demand_data)

if success(rc)
  # Convert to an MCMCChains.Chains object
  chns = read_samples(sm; output_format=:mcmcchains)
  show(chns)
  plot(chns)
  savefig(joinpath(@__DIR__, "fig_01.png"))
end
