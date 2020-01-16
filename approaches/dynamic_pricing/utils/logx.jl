
#=
def logx(x, n): 
    for i in range(0, n):
        x = math.log(x) if x>0 else 0
    return x
=#

function logx(x, n)             # iterative logarithm function
    for i in 0:n-1
    	x = x > 0.0 ? log(x) : 0.0
    end
    return x
end

