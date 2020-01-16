
#=
def linear(a, b, x):
    return b + a*x
=#

linear(a, b, x) = b + a * x

#=
# a linear demand function is generated for every 
# pair of coefficients in vectors a_vec and b_vec 
def demand_hypotheses(a_vec, b_vec):
    for a, b in itertools.product(a_vec, b_vec):
        yield {
            'd': functools.partial(linear, a, b),
            'p_opt': -b/(2*a)
        }
=#

function demand_hypothesis(f, a, b)
	f1(x) = f(a, b, x)
	return DataFrame(
		:a => a,
		:b => b,
		:d => f1,
        :d_opt => f1(-b / (2a)),
		:p_opt => -b / (2a)
	)
end

function generate_demand_hypothesis(a_range, b_range)
    h_vec = DataFrame()
    for a in a_range
    	for b in b_range
    		df1 = demand_hypothesis(linear, a, b)
    		push!(h_vec, df1[1, :]) 
    	end
    end
    h_vec
end

