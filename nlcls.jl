include("./struct.jl")
include("./dblmod2nls.jl")
"""
Easy to call version of NLSIP
"""
function easy_nlcls(first_approximation::Array{Float64},
                    number_of_active_constraints::Number_wrapper{Int64},
                    residuals!::Function, constraints!::Function,
                    exit::Number_wrapper{Int64},
                    current_residuals::Array{Float64},
                    current_constraints::Array{Float64},
                    active_constraints::Array{Int64})
    #default value
    number_of_parameters = length(first_approximation)
    number_of_constraints = length(current_constraints)
    number_of_residuals = length(current_residuals) - number_of_constraints
    iprint = 1
    max_iteration = 20 * number_of_parameters
    p_norm = 2
    no_internal = 0 #??
    hessian = true
    tol = sqrt(eps(Float64))
    eps_relative = eps(Float64)
    eps_absolute = eps(Float64)
    eps_point = sqrt(eps(Float64))
    eps_constraint = sqrt(eps(Float64))
    #penalty_weights ??
    function hfunc()


    end
    function ffunc()

    end
    number_of_iteration = Number_wrapper{Int64}
    number_of_eval = Number_wrapper{Int64}
    number_of_jac_eval = Number_wrapper{Int64}
    number_of_hessian_eval = Number_wrapper{Int64}
    number_of_linesearch_eval = Number_wrapper{Int64}
    rank_a = Number_wrapper{Int64}
    objective_at_termination = Number_wrapper{Float64}
    convergence_factor = Number_wrapper{Float64}
    ltp = Last_two_points(0, 0, 0, 0, 0, 0, 0, 0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    restart_steps = Restart_steps(0.0, 0.0, 0, 0)
    exit.value = 0
    min_l_n = min(number_of_constraints, number_of_parameters)
    if number_of_constraints


end
