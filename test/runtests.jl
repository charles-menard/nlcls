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

@testset "assort" begin
    u = [1. 4.; 2. 5.; 3. 6.]
    s = 2
    t = 2
    active_constraints = [1, 3]
    penalty_weights = [1., 2., 3.]
    assort(u, s, t, active_constraints, penalty_weights)
    @test u == [1. 4; 2. 5.; 3. 6]
    penalty_weights[3] = 11.
    assort(u, s, t, active_constraints, penalty_weights)
    @test u == [1. 4.; 2. 5. ; 11. 3.]
end

@testset "r_transpose_times_d_norm" begin
    r = [1. 3.; 2. 4]
    d = [1., 2]
    n = 2
    @test r_transpose_times_d_norm(r, n, d) ≈ 11.045361
end

@testset "equal" begin
    a = [1. 4. 7; 2. 5. 8.; 3. 6. 9.]
    b = [ 1., 2., 3.]
    l = 3
    n = 3
    active_constraints = [2, 3]
    t = 2
    p = 1
    p4 = zeros(Int, 3)
    equal(b, l, a, n, active_constraints, t, p, p4)
    a_true = [2. 5. 8; 3. 6. 9.; 1. 4. 7.]
    b_true = [2., 3., 1.]
    @test a == a_true
    @test b == b_true
end

@testset "equal" begin
    scale = 0
    a = [1. 4. 7; 2. 5. 8.; 3. 6. 9.]
    b = [ 1., 2., 3.]
    diag = zeros(Float64, 2)
    n = 3
    active_constraints = [2, 3]
    t = 2
    scale_system(scale, a, t, n, b, diag)
    @test a == [1. 4. 7.; 2. 5. 8.; 3. 6. 9]
    @test b == [1., 2., 3.]
    @test diag ≈ [8.1240384, 9.64365076]
    scale = 1
    diag = zeros(Float64, 2)
    scale_system(scale, a, t, n, b, diag)
    @test a ≈ [0.123091491 0.49236596 0.861640437;
               0.20739033 0.518475847 0.829561356;
               3. 6. 9]
    @test b ≈ [0.123091491, 0.207390339,  3.]
    @test diag ≈ [0.123091491, 0.103695169]
end

@testset "sum_sq_active_constraints" begin
    constraints = [1., 2., 3.]
    active = [1, 3]
    @test sum_sq_active_constraints(constraints, active, 2) ≈ 10.
end

