
#=
def rounds(m, T, scale):      # generate a price schedule vector
    mask = []
    for i in range(1, m):
    	print(i)
    	print(int(scale * math.ceil(logx(T, m - i))))
    	print(np.full(int(scale * math.ceil(logx(T, m - i))), i - 1))
        mask.extend( np.full(int(scale * math.ceil(logx(T, m - i))), i - 1) )
        print(mask)
    return np.append(mask, np.full(T - len(mask), m - 1))

h_vec = list(demand_hypotheses(np.linspace(start_a, end_a, 4),
	np.linspace(start_b, end_b, 4)))
=#

function intervals(m, T, scale)  # generate a price schedule vector
    mask = []
    for i in 1:m-1
        append!(mask, repeat([i - 1], Int(scale * ceil(logx(T, m - i)))) )
    end
    return append!(mask, repeat([m - 1], T - length(mask)))
end

