using Plots

function plot_state(data; points::Int = 200, legend = true, legend_position = :bottomright)
    x_range = range(data.bounds[1], data.bounds[2]; length = points)
    data_len = length(data.points.x)
    
    plot(; legend = legend ? legend_position : nothing)
    plot!(x_range, x -> data.obj_func(x); color = :blue, label = "objective function")
    plot!(x_range, x -> func_est_μ(x, data); color = :red, label = "obj. func. approximation")
    plot!(x_range, x -> func_est_μ(x, data) + func_est_σ(x, data); color = :orange, label = "obj. func. approx. uncertainty")
    plot!(x_range, x -> func_est_μ(x, data) - func_est_σ(x, data); color = :orange, label = "")
    plot!(x_range, x -> data.acq_func(x, data) + 1; color = :green, label = "acquisition function")
    plot!([data.points.x[i] for i in 1:data_len], data.points.y; seriestype = :scatter, color = :black, label = "data")
    display(plot!())
end
