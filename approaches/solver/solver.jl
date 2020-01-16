
#=

# prices - k-dimensional vector of allowed price levels 
# demands - n x k matrix of demand predictions
# c - required sum of prices 
# where k is the number of price levels, n is number of products
def optimal_prices_category(prices, demands, c):
    
    n, k = np.shape(demands)  

    # prepare inputs   
    r = np.multiply(np.tile(prices, n), np.array(demands).reshape(1, k*n))
    A = np.array([[
        1 if j >= k*(i) and j < k*(i+1) else 0
        for j in range(k*n)
    ] for i in range(n)])
    A = np.append(A, np.tile(prices, n), axis=0)
    b = [np.append(np.ones(n), c)]

    # solve the linear program
    res = linprog(-r.flatten(), 
              A_eq = A, 
              b_eq = b,  
              bounds=(0, 1))

    np.array(res.x).reshape(n, k)

# test run
optimal_prices_category(
  np.array([[1.00, 1.50, 2.00, 2.50]]),   # allowed price levels
  np.array([
    [ 28, 23, 20, 13],                    # demands for product 1
    [ 30, 22, 16, 12],                    # demands for product 2
    [ 32, 26, 19, 15]                     # demands for product 3
  ]), 5.50)                               # sum of prices for all three products

# output
[[0 0   1 0  ]                            # price vector for product 1 ($2.00)
 [0 1   0 0  ]                            # price vector for product 2 ($1.50)
 [0 0.5 0 0.5]]                           # price vector for product 3 (
1.50
a
n
d
2.50)

=#
