module UnivariateGaussianProcess

export DATA_POINTS, GAUSS_PROC_DATA, gaussian_process, func_est_μ, func_est_σ

using LinearAlgebra
using Optim

include("covariance_functions.jl")
include("acquisition_functions.jl")
include("plotting.jl")



# DATA STRUCTURES - - - - - - - -

struct DATA_POINTS
    x::AbstractArray{<:Real}
    y::AbstractArray{<:Real}
end

mutable struct GAUSS_PROC_DATA
    obj_func::Function # x::Real -> y::Real
    noise::Real # objective function noise (σ)
    bounds::Tuple{<:Real, <:Real}
    points::DATA_POINTS
    cov_func::Function # x1::Real, x2::Real -> cov::Real
    K::Matrix{<:Real} # K = C + noise^2 * I, where C[i,j] = cov_func(xi, xj)
    inv_K::Matrix{<:Real}
    acq_func::Function # x::Real, data::GAUSS_PROC_DATA -> fitness::Real
    init_mean::Function # x::Real -> y::Real
end

# return new_data::GAUSS_PROC_DATA
function update_data!(
    new_point::Tuple{<:Real, <:Real},
    data::GAUSS_PROC_DATA,
)
    new_points = DATA_POINTS(vcat(data.points.x, new_point[1]), vcat(data.points.y, new_point[2]))
    new_K = update_kernel(data.cov_func, data.K, data.points.x, new_point[1]; data.noise)

    data.points = new_points
    data.K = new_K
    data.inv_K = inv(new_K)
    return data
end



# GAUSSIAN PROCESS - - - - - - - -

# return data::GAUSS_PROC_DATA
function gaussian_process(
    obj_func::Function, # x::Real -> y::Real
    bounds::Tuple{<:Real, <:Real},
    cov_func::Function, # x1::Real, x2::Real -> cov::Real
    acq_func::Function; # x::Real, data::GAUSS_PROC_DATA -> fitness::Real
    noise::Real = 1e-8, # should be > 0
    init_mean::Function = (x) -> 0., # x::Real -> y::Real
    iters::Int = 1,
    points::DATA_POINTS = DATA_POINTS(collect(bounds), obj_func.(collect(bounds))),
    plot::Bool = true,
    plot_only_final::Bool = false,
)
    # init data
    init_K = create_kernel(cov_func, points.x; noise)
    data = GAUSS_PROC_DATA(
        obj_func,
        noise,
        bounds,
        points,
        cov_func,
        init_K,
        inv(init_K),
        acq_func,
        init_mean,
    )

    for _ in 1:iters
        plot && (plot_only_final || plot_state(data))

        # choose new point to evaluate
        new_x = find_acq_opt(data)[1]

        # evaluate objective function
        new_y = data.obj_func(new_x)

        # augment data by the new point
        data = update_data!((new_x, new_y), data)
    end

    plot && plot_state(data)

    return data
end



# OBJECTIVE FUNCTION APPROXIMATION - - - - - - - -

function func_est_μ(x::Real, data::GAUSS_PROC_DATA)
    k_x = data.cov_func.(x, data.points.x)
    return data.init_mean(x) + k_x' * data.inv_K * (data.points.y - data.init_mean.(data.points.x))
end

function func_est_σ(x::Real, data::GAUSS_PROC_DATA)
    K_x_x = data.cov_func(x, x)
    K_x = data.cov_func.(x, data.points.x)

    return sqrt(K_x_x - K_x' * data.inv_K * K_x)
end



# KERNEL COMPUTATION - - - - - - - -

# return K::Matrix{<:Real}
function create_kernel(
    cov_func::Function, # x1::Real, x2::Real -> cov::Real
    data_x::AbstractArray{<:Real};
    noise::Real,
)
    return [cov_func(x1, x2) for x1 in data_x, x2 in data_x] + noise^2 * I
end

# return K::Matrix{<:Real}
function update_kernel(
    cov_func::Function,
    K::Matrix{<:Real},
    old_data_x::AbstractArray{<:Real},
    new_x::Real;
    noise::Real,
)
    k_data_x = cov_func.(old_data_x, new_x)
    k_x_x = cov_func(new_x, new_x) + noise^2

    return vcat(hcat(K, k_data_x), vcat(k_data_x, k_x_x)')
end



# ACQ_FUNC OPTIMIZATION - - - - - - - -

# return opt_x::Real, opt_val::Real
function find_acq_opt(data::GAUSS_PROC_DATA)
    acq_func = x -> - data.acq_func(x[1], data)
    data_len = length(data.points.x)

    local_optima::Vector{Tuple{Float64, Float64}} = []
    sorted = sortperm(data.points.x)
    push!(local_optima, find_local_opt(acq_func, data.bounds[1], data.points.x[sorted[1]]))
    for i in 2:data_len
        push!(local_optima, find_local_opt(acq_func, data.points.x[sorted[i-1]], data.points.x[sorted[i]]))
    end
    push!(local_optima, find_local_opt(acq_func, data.points.x[sorted[data_len]], data.bounds[2]))

    optimum = reduce(local_optima) do a, b
        a[2] <= b[2] ? a : b
    end
    return optimum
end

# return opt_x::Real, opt_val::Real
function find_local_opt(f::Function, min_bound::Real, max_bound::Real)
    res = optimize(f, min_bound, max_bound)
    return Optim.minimizer(res)[1], Optim.minimum(res)
end

end
