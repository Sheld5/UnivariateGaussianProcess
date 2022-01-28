using Test
using UnivariateGaussianProcess

global atol = 1e-8

@testset "Acquisition function optimization" begin
    @test UnivariateGaussianProcess.find_local_opt(x -> x^2, -1., 1.)[1] ≈ 0. atol=atol
    @test UnivariateGaussianProcess.find_local_opt(x -> -sin(x), 0., π)[1] ≈ π/2 atol=atol
end

@testset "Kernel computation" begin
    cov_func = (x1,x2) -> abs(x2 - x1)
    data_x = [1., 2., 3.]
    noise = 0.
    @test UnivariateGaussianProcess.create_kernel(cov_func, data_x; noise) ≈ [
        0. 1. 2.
        1. 0. 1.
        2. 1. 0.
    ] atol=atol
    
    cov_func = (x1,x2) -> abs(x2 - x1)
    data_x = [1., 2., 3.]
    noise = 0.1
    @test UnivariateGaussianProcess.create_kernel(cov_func, data_x; noise) ≈ [
        0.01 1. 2.
        1. 0.01 1.
        2. 1. 0.01
    ] atol=atol

    cov_func = (x1,x2) -> abs(x2 - x1)
    K = [0.01 1. 2.
         1. 0.01 1.
         2. 1. 0.01]
    old_data_x = [1., 2., 3.]
    new_x = 5.
    noise = 0.1
    @test UnivariateGaussianProcess.update_kernel(cov_func, K, old_data_x, new_x; noise) ≈ [
        0.01 1. 2. 4.
        1. 0.01 1. 3.
        2. 1. 0.01 2.
        4. 3. 2. 0.01
    ] atol=atol
end
