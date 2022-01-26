
export matern1, matern3, matern5, sq_exp

function matern1(x1, x2; σ=1)
    r = radius(x1, x2)
    return σ^2 * exp(- r)
end

function matern3(x1, x2; σ=1)
    r = radius(x1, x2)
    return σ^2 * exp(- sqrt(3) * r) * (1 + sqrt(3) * r) 
end

function matern5(x1, x2; σ=1)
    r = radius(x1, x2)
    return σ^2 * exp(- sqrt(5) * r) * (1 + sqrt(5) * r + (5/3) * r^2) 
end

function sq_exp(x1, x2; σ=1)
    r = radius(x1, x2)
    return σ^2 * exp(- (1/2) * r^2) 
end

radius(x1, x2) = abs(x1 - x2)
