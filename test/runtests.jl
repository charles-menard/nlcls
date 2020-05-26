include("../src/nlcls_helper.jl")

using Test
@testset "givens_rotation" begin
    c, s, sig = givens_rotation(1., 2.)
    @test c ≈ 0.447213595
    @test s ≈ 0.894427191
    @test sig ≈ 2.23606798
    c, s, sig = givens_rotation(-2., 0.)
    @test c ≈ -1.
    @test s ≈ 0.
    @test sig ≈ 2.
end

@testset "sum_sq_active_constraints" begin
    constraints = [1., 2., 3.]
    active = [1, 3]
    @test sum_sq_active_constraints(constraints, active, 2) ≈ 10.
end

