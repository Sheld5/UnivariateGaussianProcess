module UnivariateGaussianProcess

using DocStringExtensions
using LinearAlgebra
using Optim

include("covariance_functions.jl")
include("acquisition_functions.jl")
include("plotting.jl")

export DATA_POINTS, GAUSS_PROC_DATA
export gaussian_process, func_est_μ, func_est_σ, get_max



# DATA STRUCTURES - - - - - - - -

"""
Data about previous evaluations of the objective function.

$(TYPEDFIELDS)
"""
struct DATA_POINTS
    "Points of the objective function domain at which the function has been evaluated."
    x::AbstractArray{<:Real}
    "The values of the objective function evaluations at these points of the domain."
    y::AbstractArray{<:Real}
end

"""
All data about the performed Bayesian Optimization.

$(TYPEDFIELDS)
"""
mutable struct GAUSS_PROC_DATA
    "The objective function.\n
    x::Real -> y::Real"
    obj_func::Function
    "The deviation of the objective function value when evaluated at a single point of the domain."
    noise::Real
    "The bounds of the searched interval of the objective function domain."
    bounds::Tuple{<:Real, <:Real}
    "Data about performed objective function evaluations."
    points::DATA_POINTS
    "The covariance function used by the Gaussian Process.\n
    x1::Real, x2::Real -> cov::Real"
    cov_func::Function
    "The kernel matrix of the Gaussian Process.\n
    K = C + noise^2 * I, where C[i,j] = cov_func(xi, xj)"
    K::Matrix{<:Real}
    "The inversed kernel matrix of the Gaussian Process."
    inv_K::Matrix{<:Real}
    "The used acquisition function.\n
    x::Real, data::GAUSS_PROC_DATA -> fitness::Real"
    acq_func::Function
    "The prior belief about the objective function.\n
    x::Real -> y::Real"
    init_mean::Function
end

# return data::GAUSS_PROC_DATA
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

"""
    gaussian_process(
        obj_func::Function, # x::Real -> y::Real
        bounds::Tuple{<:Real, <:Real},
        cov_func::Function, # x1::Real, x2::Real -> cov::Real
        acq_func::Function; # x::Real, data::GAUSS_PROC_DATA -> fitness::Real
        noise::Real = 1e-8, # should be > 0
        init_mean::Function = (x) -> 0., # x::Real -> y::Real
        evals::Int = 1,
        points::Union{DATA_POINTS, Nothing} = nothing,
        plot::Bool = true,
        plot_only_final::Bool = false,
    )

Perform Bayesian Optimization with Gaussian Process surrogate model.

Return a structure containing all information about the performed optimization. (::GAUSS_PROC_DATA)

See also [`GAUSS_PROC_DATA`](@ref).

# Example

```julia-repl
julia> data = gaussian_process(sin, (-π, π), matern1, ucb; iters=20, noise=1e-4, init_mean = (x) -> x);

```

# Troubleshooting

If you get error about the kernel not being positive definite or sqrt being called on a negative value,
increase the objective function noise.
    
The objective function noise should not be set too close to zero
to prevent these issues arising because of numerical errors.
"""
function gaussian_process(
    obj_func::Function, # x::Real -> y::Real
    bounds::Tuple{<:Real, <:Real},
    cov_func::Function, # x1::Real, x2::Real -> cov::Real
    acq_func::Function; # x::Real, data::GAUSS_PROC_DATA -> fitness::Real
    noise::Real = 1e-4, # should be > 0
    init_mean::Function = (x) -> 0., # x::Real -> y::Real
    points::Union{DATA_POINTS, Nothing} = nothing,
    evals::Int = 1, # has to be >= 1
    plot::Bool = true,
    plot_only_final::Bool = false,
)
    evals <= 0 && throw(ArgumentError("`evals` has to be >= 1"))

    if points === nothing
        evals < 2 && throw(ArgumentError("`evals` has to be >= 2 if `points === nothing`"))
        points = DATA_POINTS(collect(bounds), obj_func.(collect(bounds)))
        evals -= 2
    end

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

    return gaussian_process(data; evals, plot, plot_only_final)
end

"""
    gaussian_process(
        data::GAUSS_PROC_DATA;
        evals::Int = 1,
        plot::Bool = true,
        plot_only_final::Bool = false,
    )

Continue in Bayesian Optimization with Gaussian Process surrogate model given the data from the previous run.

Return a structure containing all information about the performed optimization. (::GAUSS_PROC_DATA)

See also [`GAUSS_PROC_DATA`](@ref).

# Example

```julia-repl
julia> data = gaussian_process(sin, (-π, π), matern1, ucb; iters=20, noise=1e-4, init_mean = (x) -> x);

julia> data = gaussian_process(data; iters=20);

```

# Troubleshooting

If you get error about the kernel not being positive definite or sqrt being called on a negative value,
increase the objective function noise.
    
The objective function noise should not be set too close to zero
to prevent these issues arising because of numerical errors.
"""
function gaussian_process(
    data::GAUSS_PROC_DATA;
    evals::Int = 1, # has to be >= 0
    plot::Bool = true,
    plot_only_final::Bool = false,
)
    evals < 0 && throw(ArgumentError("`evals` has to be >= 1"))
    evals == 0 && return data

    for _ in 1:evals
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

"""
    func_est_μ(x::Real, data::GAUSS_PROC_DATA)

The most probable approximation of the objective function based on the Gaussian Process with the gathered data.

Return the mean of the gaussian distribution describing the objective function value estimated by the Gaussian Process at the given point.

See also [`func_est_σ`](@ref), [`gaussian_process`](@ref), [`GAUSS_PROC_DATA`](@ref).

# Examples

```julia-repl
julia> data = gaussian_process(sin, (-π, π), matern1, ucb; iters=20, init_mean = (x) -> x);

julia> func_est_μ(1, data)
0.8406095467907111
```

```julia-repl
julia> data = gaussian_process(sin, (-π, π), matern1, ucb; iters=20, init_mean = (x) -> x);

julia> approx_f = (x) -> func_est_μ(x, data);

julia> approx_f(1)
0.8406095467907111

julia> sin(1)
0.8414709848078965
```

(Note that in this example we use the UCB acquisition function
meaning that our goal is maximizing the objective function
and not finding its best approximation.)
"""
function func_est_μ(x::Real, data::GAUSS_PROC_DATA)
    k_x = data.cov_func.(x, data.points.x)
    return data.init_mean(x) + k_x' * data.inv_K * (data.points.y - data.init_mean.(data.points.x))
end

"""
    func_est_σ(x::Real, data::GAUSS_PROC_DATA)

The uncertainty of the objective function approximation.

Return the deviation of the gaussian distribution describing the objective function value estimated by the Gaussian Process at the given point.

See also [`func_est_μ`](@ref), [`gaussian_process`](@ref), [`GAUSS_PROC_DATA`](@ref).

# Examples

```julia-repl
julia> data = gaussian_process(sin, (-π, π), matern1, ucb; iters=20, init_mean = (x) -> x);

julia> func_est_σ(1, data)
0.20269550183507606
```

```julia-repl
julia> data = gaussian_process(sin, (-π, π), matern1, ucb; iters=20, init_mean = (x) -> x);

julia> uncertainty = (x) -> func_est_σ(x, data);

julia> uncertainty(1)
0.20269550183507606
```
"""
function func_est_σ(x::Real, data::GAUSS_PROC_DATA)
    K_x_x = data.cov_func(x, x)
    K_x = data.cov_func.(x, data.points.x)

    return sqrt(K_x_x - K_x' * data.inv_K * K_x)
end

"""
    get_max(data::GAUSS_PROC_DATA)

Return the best-so-far maximum found by the Bayesian optimization.

See also [`GAUSS_PROC_DATA`](@ref).

# Example

```julia-repl
julia> data = gaussian_process(sin, (-π, π), matern1, ucb; iters=20, init_mean = (x) -> x);

julia> get_max(data)
(1.575360217679531, 0.9999895854680737)
```
"""
function get_max(data::GAUSS_PROC_DATA)
    points = data.points
    max_i = argmax(points.y)
    return points.x[max_i], points.y[max_i]
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
