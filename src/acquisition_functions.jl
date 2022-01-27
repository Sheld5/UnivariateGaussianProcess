
export ucb, poi, ei

"""
    ucb(x::Real, data; β::Real = 1)

The Upper Confidence Bound acquisition function.

The `β` is a hyperparameter.
Set `β` high to make the Bayesian Optimization more explorative
and set it lower to make it more exploitative.

Use this function as parameter for the `gaussian_process` function.

See also [`gaussian_process`](@ref), [`poi`](@ref), [`ei`](@ref).
"""
function ucb(x::Real, data; β::Real = 1)
    return func_est_μ(x, data) + β * func_est_σ(x, data)
end

"""
    poi(x::Real, data; τ::Real = maximum(data.points.y))

The Probability of Improvement acquisition function.

The `τ` is a hyperparameter.
This function describes the probability of improvement upon the function value `τ`.

Use this function as parameter for the `gaussian_process` function.

See also [`gaussian_process`](@ref), [`ucb`](@ref), [`ei`](@ref).
"""
function poi(x::Real, data; τ::Real = maximum(data.points.y))
    σ = func_est_σ(x, data)
    σ == 0 && return 0
    μ = func_est_μ(x, data)

    stdnorm = Normal()
    return cdf(stdnorm, (μ - τ) / σ)
end

"""
    ei(x::Real, data; τ::Real = maximum(data.points.y))

The Expected Improvement acquisition function.

The `τ` is a hyperparameter.
This function describes the expected improvement upon the function value `τ`.

Use this function as parameter for the `gaussian_process` function.

See also [`gaussian_process`](@ref), [`ucb`](@ref), [`poi`](@ref).
"""
function ei(x::Real, data; τ::Real = maximum(data.points.y))
    σ = func_est_σ(x, data)
    σ == 0 && return 0
    μ = func_est_μ(x, data)

    stdnorm = Normal()
    return (μ - τ) * cdf(stdnorm, (μ - τ) / σ) + σ * pdf(stdnorm, (μ - τ) / σ)
end
