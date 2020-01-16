
#=
def sample_actual_demand(price): 
    avg_demand = 65 + (-0.8) * price
    theta = 0.1/4
    k = avg_demand / theta
    return np.random.gamma(k*theta, k*theta**2, 1)[0]
=#

function sample_actual_demand(price)
    avg_demand = 65 + (-0.8) * price
    theta = 0.1/4
    k = avg_demand / theta
    return rand(Gamma(k*theta, k*theta^2), 1)
end

#=
def emperical_mean(sampler, n): 
    mean = 0 
    for i in range(1, n): 
        mean = mean + sampler() 
    return mean/n
=#

function emperical_mean(sampler, n)
    mean = 0
    for i in 1:n-1
        mean += sampler()[1]
    end
    return mean / n
end


#=
def emperical_demand_curve(min_price, max_price, n): 
    prices = np.linspace(min_price, max_price, n) 
    sampling = 5000 
    demands = map(lambda p: emperical_mean(functools.partial(sample_actual_demand, p), sampling), prices) 
    return np.dstack((prices, list(demands)))[0]
=#

function emperical_demand_curve(min_price, max_price, n)
    prices = range(min_price, max_price, length=n) 
    sampling = 5000
    demands = []
    for p in prices
        sampler() = sample_actual_demand(p)
        demands = append!(demands, emperical_mean(sampler, sampling))
    end
    return DataFrame(
        :prices => prices,
        :demands => demands
        )
end

