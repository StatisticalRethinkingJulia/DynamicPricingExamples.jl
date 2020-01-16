
function select_optimal_demand_curve(t_mask, p)
    local row_opt

    tau = 1                      # start time of the current interval
    hist_d = []
    hist_df = DataFrame()
    row_opt = 0
    for t in range(1, stop=T-1)  # simulation loop
        old_price = p
        realized_d = sample_actual_demand(p)
        append!(hist_d, realized_d)
        
        if t_mask[t] !== t_mask[t + 1]  # end of the interval
            interval_mean_d = mean(hist_d[tau : t])
            
            min_dist = Inf
            for i in 1:nrow(h_vec)  # search for the best hypothesis
                dist = abs(interval_mean_d - h_vec[i, :d](p))
                if dist < min_dist
                    min_dist = dist
                    row_opt = i
                end
            end
                    
            p = h_vec[row_opt, :p_opt] # set price for the next interval
            tau = t + 1                # switch to the next interval
        end
        append!(hist_df,
            DataFrame(:price => old_price, :realized_d => realized_d,
                :opt_row => row_opt))
    end
    realized_d = sample_actual_demand(p)
    append!(hist_df,
        DataFrame(:price => p, :realized_d => realized_d,
            :opt_row => row_opt))
    (p, row_opt, hist_df)
end
