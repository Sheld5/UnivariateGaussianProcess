
export ucb, poi, ei

# Upper Confidence Bound
function ucb(x::Real, data; β::Real = 1)
    return func_est_μ(x, data) + β * func_est_σ(x, data)
end

# Probability of Improvement
function poi(x::Real, data; τ::Real = maximum(data.points.y))
    σ = func_est_σ(x, data)
    σ == 0 && return 0
    μ = func_est_μ(x, data)

    stdnorm = Normal()
    return cdf(stdnorm, (μ - τ) / σ)
end

# Expected Improvement
function ei(x::Real, data; τ::Real = maximum(data.points.y))
    σ = func_est_σ(x, data)
    σ == 0 && return 0
    μ = func_est_μ(x, data)

    stdnorm = Normal()
    return (μ - τ) * cdf(stdnorm, (μ - τ) / σ) + σ * pdf(stdnorm, (μ - τ) / σ)
end
