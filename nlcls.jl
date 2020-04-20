include("./struct.jl")
include("./dblmod2nls.jl")
"""
Easy to call version of NLSIP
"""
function easy_nlcls(first_approximation::Array{Float64},
                    residuals!::Function, constraints!::Function,
                    penalty_weights::Array{Float64}
                    urrent_residuals::Array{Float64},
                    current_constraints::Array{Float64},
                    active_constraints::Array{Int64})
    number_of_equality_constraints = Number_wrapper{Int64}
    exit = Number_wrapper{Int64}
    #default value
    number_of_parameters = length(first_approximation)
    number_of_constraints = length(current_constraints)
    number_of_residuals = length(current_residuals) - number_of_constraints
    min_l_n = min(number_of_constraints, number_of_parameters)
    #validation

    if (length(penalty_weights) != number_of_constraints
        || length(active_constraints) != min_l_n + number_of_constraint)
        exit.value = -1
        return
    end




    iprint = 1
    nout = 10
    max_iteration = 20 * number_of_parameters
    p_norm = 2
    scale = 0
    no_internal = 0 #??
    hessian = true
    tol = sqrt(eps(Float64))
    eps_relative = sqrt(eps(Float64))
    eps_absolute = eps(Float64)
    eps_point = sqrt(eps(Float64))
    eps_constraint = sqrt(eps(Float64))
    #penalty_weights ??
    function hfunc()


    end
    function ffunc()

    end
    number_of_iterations = Number_wrapper{Int64}
    number_of_eval = Number_wrapper{Int64}
    number_of_jac_eval = Number_wrapper{Int64}
    number_of_hessian_eval = Number_wrapper{Int64}
    number_of_linesearch_eval = Number_wrapper{Int64}
    rank_a = Number_wrapper{Int64}
    rank = Number_wrapper{Int64}
    objective_at_termination = Number_wrapper{Float64}
    convergence_factor = Number_wrapper{Float64}
    ltp = Last_two_points(0, 0, 0, 0, 0, 0, 0, 0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    restart_steps = Restart_steps(0.0, 0.0, 0, 0)
    ifree = Number_wrapper{Int64}
    for i = 1:number_of_constraints
        penalty_weights[i] = 0.0
    end
    mpl = number_of_residuals + number_of_constraints
    p1 = Array{Int64}(undef, min_l_n)
    p2 = Array{Int64}(undef, min_l_n)
    p3 = Array{Int64}(undef, number_of_parameters)
    inactive_constraints = Array{Int64}(undef, number_of_constraints)
    p4 = Array{Int64}(undef, number_of_constraints)
    old_penalty_weights = Array{Int64}(undef, number_of_constraints)
    s = Array{Float64}(undef, max(number_of_constraints, number_of_parameters))
    b = Array{Float64}(undef, min_l_n)
    d1 = Array{Float64}(undef, min_l_n)
    d2 = Array{Float64}(undef, min_l_n)
    lagrange_mult = Array{Float64}(undef, min_l_n) #v
    g = Array{Float64}(undef, number_of_parameters)
    pivot = Array{Float64}(undef, number_of_parameters)
    d3 = Array{Float64}(undef, number_of_parameters)
    scaling_matrix = Array{Float64}(undef, number_of_parameters)
    search_direction = Array{Float64}(undef, number_of_parameters)
    u = Array{Float64}(undef, number_of_parameters)
    d = Array{Float64}(undef, mpl)
    v1 = Array{Float64}(undef, mpl)
    v2 = Array{Float64}(undef, mpl)
    c = Array{Float64, 2}(undef, number_of_residuals, number_of_parameters)
    a = Array{Float64, 2}(undef, number_of_constraints, number_of_parameters)
    fmat = Array{Float64, 2}(undef, number_of_parameters, number_of_parameters)
    gmat = Array{Float64, 2}(undef, number_of_parameters, number_of_parameters)

    nonlinear_least_square(first_approximation, number_of_parameters,
                           number_of_residuals, number_of_constraints,
                           number_of_parameters, number_of_parameters,
                           number_of_residuals, number_of_equality_constraints,
                           number_of_constraints, tol, penalty_weights,
                           eps_relative, eps_absolute, eps_point,
                           eps_constraints, iprint, nout, max_iteration, p_norm,
                           scale, hessian, residuals!, constraints!, exit,
                           objective_at_termination, number_of_iterations,
                           number_of_eval, number_of_jac_eval,
                           number_of_hessian_eval, number_of_linesearch_eval,
                           rank_a, rank, p4, b, d1, d2, d3, scaling_matrix,
                           g, pivot, search_direction, lagrange_mult, u, s,
                           old_penalty_weights, a, d, v1, v2, c, fmat, gmat,
                           ltp, restart_steps, ifree)
    scalar_results = Nlcls_scalars(number_of_active_constraints.value,
                                   exit.value, number_of_iterations.value,
                                   number_of_eval.value,
                                   number_of_jac_eval.value,
                                   number_of_hessian_eval.value,
                                   number_of_linesearch_eval.value,
                                   rank_a.value, rank_ac.value,
                                   objective_at_termination.value,
                                   convergence_factor.value)
    return scalar_results

end
