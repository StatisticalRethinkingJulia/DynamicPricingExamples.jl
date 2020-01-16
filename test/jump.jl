using JuMP, GLPK, Test

function optimal_price_probabilties(
    price, demand, capacity; verbose = true)
    revenue = price .* demand
    model = Model(with_optimizer(GLPK.Optimizer))
    @variable(model, x[1:length(price)], lower_bound=0)
    # Objective: maximize profit
    @objective(model, Max, revenue' * x)
    # Constraint: can carry all
    @constraint(model, sum(x) <= 1)
    #=
    for i in 1:length(prices)
        @constraint(model, x[i] <= demand[i])
    end
    =#
    # Solve problem using MIP solver
    JuMP.optimize!(model)
    if verbose
        println("Objective is: ", JuMP.objective_value(model))
        println("Solution is:")
        for i in 1:length(price)
            print("x[$i] = ", JuMP.value(x[i]))
            println(", revenue = ", price[i] * JuMP.value(x[i]))
        end
    end
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    JuMP.value.(x)
end

price = [1.99, 2.49, 2.99, 3.49, 3.99, 4.49]
demand = 50 .- 7 * price
capacity = 60

prob = optimal_price_probabilties(price, demand, capacity)
prob |> display
println()

demand2 = [33.74, 31.29, 25.81, 34.68, 31.84, 15.64]
prob2 = optimal_price_probabilties(price, demand2, capacity)
prob2 |> display
