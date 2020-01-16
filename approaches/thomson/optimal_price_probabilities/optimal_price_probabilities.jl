using StatsBase, JuMP, GLPK, Test

function optimal_price_probabilities(
    price, demand, capacity; verbose = false)
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
    Weights(JuMP.value.(x))
end
