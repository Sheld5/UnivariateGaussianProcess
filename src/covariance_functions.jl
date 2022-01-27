export matern1, matern3, matern5, sq_exp

"""
    matern1(x1, x2; σ=1)

The matern1 kernel function. (Also called covariance function.)

Use this function as a parameter for the `gaussian_process` function.

See also [`gaussian_process`](@ref), [`matern3`](@ref), [`matern5`](@ref), [`sq_exp`](@ref).
"""
function matern1(x1, x2; σ=1)
    r = radius(x1, x2)
    return σ^2 * exp(- r)
end

"""
    matern3(x1, x2; σ=1)

The matern3 kernel function. (Also called covariance function.)

Use this function as a parameter for the `gaussian_process` function.

See also [`gaussian_process`](@ref), [`matern1`](@ref), [`matern5`](@ref), [`sq_exp`](@ref).
"""
function matern3(x1, x2; σ=1)
    r = radius(x1, x2)
    return σ^2 * exp(- sqrt(3) * r) * (1 + sqrt(3) * r) 
end

"""
    matern5(x1, x2; σ=1)

The matern5 kernel function. (Also called covariance function.)

Use this function as a parameter for the `gaussian_process` function.

See also [`gaussian_process`](@ref), [`matern1`](@ref), [`matern3`](@ref), [`sq_exp`](@ref).
"""
function matern5(x1, x2; σ=1)
    r = radius(x1, x2)
    return σ^2 * exp(- sqrt(5) * r) * (1 + sqrt(5) * r + (5/3) * r^2) 
end

"""
    matern1(x1, x2; σ=1)

The "squared exponential" kernel function. (Also called covariance function.)

Use this function as a parameter for the `gaussian_process` function.

See also [`gaussian_process`](@ref), [`matern1`](@ref), [`matern3`](@ref), [`matern5`](@ref).
"""
function sq_exp(x1, x2; σ=1)
    r = radius(x1, x2)
    return σ^2 * exp(- (1/2) * r^2) 
end

# return r::Real
radius(x1, x2) = abs(x1 - x2)
