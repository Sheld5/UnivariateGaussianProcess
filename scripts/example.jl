using Revise
using UnivariateGaussianProcess

function run_gp()
    obj_func = sin
    bounds = (-π, π)
    cov_func = (x1,x2) -> matern1(x1, x2; σ=2)
    acq_func = ucb

    noise = 1e-2
    init_mean = (x) -> x
    x_points = [-π, 0., π]
    y_points = obj_func.(x_points)
    points = DATA_POINTS(x_points, y_points)

    return gaussian_process(
        obj_func,
        bounds,
        cov_func,
        acq_func;
        noise,
        init_mean,
        iters = 6,
        #points,
        plot = true,
        plot_only_final = false,
    )
end
