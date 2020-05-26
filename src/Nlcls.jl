module Nlcls

include("nlcls_helper.jl")
export nonlinear_least_square, easy_nlcls!

"""
Replaces the subroutine NLSNIP in dblmod2nls.f
"""
function nonlinear_least_square(current_point::AbstractArray{Float64},
                                number_of_parameters::Int,
                                leading_dim_c::Int, leading_dim_a::Int,
                                leading_dim_g::Int, leading_dim_f::Int,
                                number_of_residuals::Int,
                                (number_of_equality_constraints::
                                 Number_wrapper{Int}),
                                number_of_constraints::Int, tol::Float64,
                                penalty_weights::AbstractArray{Float64},
                                eps_relative::Float64, eps_absolute::Float64,
                                eps_point::Float64, eps_constraints::Float64,
                                verbose::Int, nout::Int,
                                maximum_iteration::Int, p_norm::Int,
                                scale::Int, hessian::Bool,
                                residuals!::Function, constraints!::Function,
                                exit::Number_wrapper{Int},
                                current_objective::Number_wrapper{Float64},
                                iteration_number::Number_wrapper{Int},
                                number_of_eval::Number_wrapper{Int},
                                number_of_jac_eval::Number_wrapper{Int},
                                number_of_hessian_eval::Number_wrapper{Int},
                                number_of_linesearch_eval::Number_wrapper{Int},
                                rank_a::Number_wrapper{Int},
                                rank::Number_wrapper{Int},
                                current_residuals::AbstractArray{Float64},
                                current_constraints::AbstractArray{Float64},
                                active_constraints::AbstractArray{Int},
                                convergence_factor::Number_wrapper{Float64},
                                p1::AbstractArray{Int},
                                p2::AbstractArray{Int},
                                p3::AbstractArray{Int},
                                inactive_constraints::AbstractArray{Int},
                                p4::AbstractArray{Int},
                                b::AbstractArray{Float64},
                                d1::AbstractArray{Float64},
                                d2::AbstractArray{Float64},
                                d3::AbstractArray{Float64},
                                scaling_matrix::AbstractArray{Float64},
                                g::AbstractArray{Float64},
                                pivot::AbstractArray{Float64},
                                search_direction::AbstractArray{Float64},
                                v::AbstractArray{Float64},
                                u::AbstractArray{Float64},
                                s::AbstractArray{Float64},
                                old_penalty_weights::AbstractArray{Float64},
                                a::AbstractArray{Float64, 2},
                                d::AbstractArray{Float64},
                                v1::AbstractArray{Float64},
                                v2::AbstractArray{Float64},
                                c::AbstractArray{Float64, 2},
                                fmat::AbstractArray{Float64, 2},
                                gmat::AbstractArray{Float64, 2},
                                ltp::Last_two_points,
                                restart_steps::Restart_steps,
                                ifree::Number_wrapper{Int})
    d1_norm = Number_wrapper{Float64}(0.)
    d_norm::Number_wrapper{Float64} = Number_wrapper(0.)
    b1_norm::Number_wrapper{Float64} = Number_wrapper(0.)
    beta_k = Number_wrapper(0.)
    rank_c2 = Number_wrapper{Int}(0)
    gn_direction_norm = Number_wrapper(0.)
    gres = Number_wrapper{Float64}(0.)
    deleted_constraints = Number_wrapper{Bool}(false)
    added_constraints = Number_wrapper(false)
    time = Number_wrapper{Int}(0)
    sigmin = Number_wrapper{Float64}(0.)
    absvmx = Number_wrapper{Float64}(0.)
    wh_norm = Number_wrapper(0.0)
    gradient_norm = 0.0
    code = Number_wrapper{Int}(0)
    eval = Number_wrapper{Int}(0)
    dim_a = Number_wrapper{Int}(0)
    dim_c2 = Number_wrapper(0)
    steplength = Number_wrapper{Float64}(0.)
    lower_bound_steplength = Number_wrapper{Float64}(0)
    upper_bound_steplength = Number_wrapper{Float64}(0)
    current_psi = Number_wrapper(0.0)
    index_upper_bound = Number_wrapper{Int}(0)
    number_of_active_constraints = Number_wrapper{Int}(0)
    lmt = Number_wrapper{Int}(0)
    number_of_householder = Number_wrapper{Int}(0)
    #common wsave, check if all sub that use it are called just in enlsip
    wsave = Array{Float64, 2}(undef, 100, 4)
 
    exit.value = 0
    if ((number_of_residuals + number_of_constraints) < number_of_parameters
        || number_of_parameters <= 0
        || number_of_residuals <= 0
        || leading_dim_c < number_of_residuals)
        exit.value = -1
    end
    if (number_of_equality_constraints > number_of_parameters
        || number_of_equality_constraints < 0
        || number_of_constraints < number_of_equality_constraints
        || leading_dim_a < number_of_constraints)
        exit.value = -1
    end
    if (leading_dim_g < number_of_parameters
        || maximum_iteration <= 0
        || p_norm < 0
        || p_norm > 2)
        exit.value = -1
    end
    if (number_of_constraints > number_of_equality_constraints
        && leading_dim_f < number_of_parameters)
        exit.value = -1
    end
    if (tol < 0.0
        || scale < 0
        || eps_relative < 0)
        exit.value = -1
    end
    if (eps_absolute < 0
        || eps_point < 0
        || eps_constraints < 0)
        exit.value = -1
    end
    if eps_relative + eps_absolute + eps_point + eps_constraints <= 0
        exit.value = -1
    end

    epsilon = eps(Float64)
    if number_of_parameters > 100
        p_norm = 0
    end
    iteration_number.value = 0
    ifree.value = 0
    min_l_n = min(number_of_constraints, number_of_parameters)
    error = Number_wrapper(0)
    number_of_eval.value = 0
    number_of_jac_eval.value = 0
    number_of_hessian_eval.value = 0
    number_of_linesearch_eval.value = 0
    restart = Number_wrapper(false)
    ltp.kodkm2 = 1
    ltp.alfkm2 = 1.0
    ltp.kodkm1 = 1
    tau = tol
    x_diff = Number_wrapper(norm(current_point) ^ 2 * eps_point)

    ctrl_residuals = Number_wrapper(1)
    ctrl_constraints = Number_wrapper(1)
    residuals!(current_point, number_of_parameters, current_residuals,
               number_of_residuals, ctrl_residuals, c, leading_dim_c)
    
    constraints!(current_point, number_of_parameters, current_constraints,
                 number_of_constraints, ctrl_constraints, a, leading_dim_a)
    number_of_eval.value += 1
    if ctrl_residuals.value == -1 || ctrl_constraints == -1
        exit.value = -7
    end
    if exit.value < 0
        return
    end
   init_working_set(number_of_equality_constraints.value, number_of_constraints,
                     current_constraints, active_constraints,
                     number_of_active_constraints, min_l_n, p_norm,
                     inactive_constraints, lmt, penalty_weights,
                     old_penalty_weights, exit, wsave)
    if exit.value < 0
        return
    end

    ltp.rkckm1 = number_of_parameters - number_of_active_constraints.value
    ltp.alfkm1 = 1.0
    restart_steps.lattry = number_of_parameters
    restart_steps.bestpg = 0.0
    ltp.rkakm1 = number_of_active_constraints.value
    ltp.tkm1 = number_of_active_constraints.value
    fsum = Number_wrapper(
        BLAS.nrm2(number_of_residuals, current_residuals, 1) ^ 2)
    ltp.fsqkm1 = fsum.value
    ltp.hsqkm1 = sum_sq_active_constraints(
        current_constraints, active_constraints,
        number_of_active_constraints.value)

    #10
    @label iteration_loop
    if error.value == -5 || error.value == -3
        error.value = 0
    end
    if restart.value
        @goto check_termination_criteria
    end
    new_point(current_point, number_of_parameters, current_constraints,
              number_of_constraints, current_residuals, number_of_residuals,
              constraints!, residuals!, leading_dim_a, leading_dim_c,
              number_of_eval, a, c, b, d, error)
    if error.value < -10
        @goto user_stop
    end
    number_of_jac_eval.value += 1

    ath_norm = 0.0
    if number_of_active_constraints.value != 0
        sss = 0.0
        ii = 0
        for j = 1:number_of_parameters
            sss = 0.0
            for i = 1:number_of_active_constraints.value
                ii = active_constraints[i]
                sss += a[ii, j] * current_constraints[ii]
            end
            ath_norm += sss ^ 2
        end
        ath_norm = sqrt(ath_norm)
    end

    equal(b, number_of_constraints, a, leading_dim_a, number_of_parameters,
          active_constraints, number_of_active_constraints.value,
          number_of_equality_constraints.value, p4)
    gradient(c, number_of_residuals, number_of_parameters,
             current_residuals, g)
    gradient_norm = norm(g)
    scale_system(scale, a, leading_dim_a, number_of_active_constraints.value,
                 number_of_parameters, b, scaling_matrix)
    current_point_norm = norm(current_point)
    current_objective.value = 0.5 * fsum.value
    println(number_of_active_constraints)
    update_active_constraints(a, leading_dim_a, number_of_active_constraints,
                              number_of_equality_constraints.value,
                              number_of_parameters, g, b, tau, leading_dim_f,
                              scale, iteration_number.value, scaling_matrix,
                              active_constraints, min_l_n, inactive_constraints,
                              lmt, current_constraints, gn_direction_norm,
                              p4, c, leading_dim_c, number_of_residuals,
                              current_residuals, leading_dim_g, current_point,
                              constraints!, residuals!, number_of_eval,
                              number_of_jac_eval, p2, p3, search_direction,
                              v1, d2, d3, rank_c2, d1_norm, d_norm, b1_norm,
                              d, gmat, p1, v, d1, fmat, rank_a, gres,
                              number_of_householder, time, deleted_constraints,
                              pivot, v2, s, u, ltp, restart_steps)

    hsum = Number_wrapper(
        sum_sq_active_constraints(current_constraints,
                                  active_constraints,
                                  number_of_active_constraints.value))
    gn_direction_norm.value = d_norm.value
    search_direction_norm = Number_wrapper(
        BLAS.nrm2(number_of_parameters, search_direction, 1)
    )
    alfnoi = sqrt(epsilon) / (search_direction_norm.value + epsilon)
    min_max_lagrange_mult(number_of_equality_constraints.value,
                          number_of_active_constraints.value,
                          v, scale, scaling_matrix, sigmin, absvmx)

    if ltp.kodkm1 == 2
        @goto check_termination_criteria
    end
    success = ltp.d1km1 - d1_norm.value
    if ltp.rkckm1 == rank_c2.value
        @goto check_termination_criteria
    end
    if success < 0.0
        ifree.value = 5
        restart_steps.nrrest = 0
        @goto iteration_loop
    end

    @label check_termination_criteria
    #wh_norm not initialised but check for value in termination_criteria??
    termination_criteria(rank_a.value, rank_c2.value, error.value,
                         number_of_equality_constraints.value,
                         number_of_active_constraints.value, number_of_parameters,
                         restart.value, deleted_constraints.value,
                         maximum_iteration, iteration_number.value,
                         fsum.value, d1_norm.value, hsum.value, gres.value,
                         current_point_norm, gradient_norm, alfnoi,
                         current_constraints, inactive_constraints, lmt.value,
                         x_diff.value, eps_absolute, eps_relative, eps_point,
                         eps_constraints, sigmin.value, absvmx.value,
                         old_penalty_weights,
                         current_point, ath_norm, wh_norm.value, exit, ltp,
                         restart_steps, ifree)
    if exit.value != 0
        @goto set_convergence
    end
    check_last_step(iteration_number.value, restart.value, code, fsum.value,
                    d1_norm,
                    d_norm, c, leading_dim_c, number_of_residuals,
                    number_of_parameters, rank_c2.value, d, current_residuals,
                    p3, d3, active_constraints, v, inactive_constraints,
                    lmt.value, p4, time.value, a, leading_dim_a,
                    number_of_equality_constraints.value,
                    number_of_active_constraints.value, rank_a.value, b1_norm,
                    hsum.value, number_of_householder.value, d1, p1, d2, p2, b,
                    current_constraints, number_of_constraints, fmat,
                    leading_dim_f, pivot, gmat, leading_dim_g, residuals!,
                    constraints!, current_point, hessian, added_constraints.value,
                    deleted_constraints.value, scale, scaling_matrix,
                    search_direction, search_direction_norm, v1, eps_relative,
                    error, eval, beta_k, dim_a, dim_c2,
                    v2, u, ltp, restart_steps, ifree)
    if error.value < -10
        @goto user_stop
    end
    number_of_eval.value += eval.value
    number_of_hessian_eval.value += eval.value

    if error.value == -3
        ifree.value = 5
        restart_steps.value = 0
        @goto iteration_loop
    end
    if error.value == -4
        @goto check_termination_criteria
    end

    copyto!(u, 1, current_point, 1, number_of_parameters)
    if code.value == 2 && restart_steps.value == 1
        coptyto!(old_penalty_weights, 1, current_point, 1, number_of_parameters)
    end
    compute_steplength(restart.value, current_point, a, leading_dim_a,
                       search_direction, number_of_parameters,
                       current_residuals, v1,
                       number_of_residuals, fsum.value, residuals!,
                       rank_c2.value, code.value, current_constraints,
                       constraints!, active_constraints,
                       number_of_active_constraints.value, inactive_constraints,
                       p4, lmt.value, number_of_constraints, penalty_weights,
                       old_penalty_weights, dim_a.value, p_norm,
                       number_of_householder.value,
                       current_psi, steplength, lower_bound_steplength,
                       upper_bound_steplength,
                       number_of_eval, x_diff, error, index_upper_bound, wh_norm,
                       d, v2, s, g, ltp, restart_steps, wsave)
    println("steplength ", steplength)
    if error.value < -10
        @goto user_stop
    end
    number_of_eval.value += eval.value
    number_of_linesearch_eval.value += eval.value
    evaluation_restart(current_point, u, number_of_parameters,
                       number_of_residuals, iteration_number, residuals!,
                       number_of_eval, current_residuals, d1_norm.value,
                       d_norm.value, fsum, dim_c2.value, code.value,
                       search_direction_norm.value, beta_k.value,
                       steplength.value, lower_bound_steplength.value,
                       active_constraints,
                       current_constraints,
                       number_of_constraints, number_of_active_constraints.value,
                       constraints!, b1_norm.value, hsum, dim_a.value, error,
                       restart, ltp, restart_steps, ifree)
    if error.value < -10
        @goto user_stop
    end
    added_constraints.value = false
    if restart.value || error.value == -1 || error.value == -5
        @goto iteration_loop
    end
    move_violated_constraints(current_constraints, active_constraints,
                              number_of_active_constraints, min_l_n,
                              number_of_equality_constraints.value,
                              inactive_constraints, lmt, ind,
                              iteration_number.value, added_constraints)
    output(verbose, iteration_number.value, nout, gres.value, penalty_weights,
           active_constraints, convergence_factor, ltp, restart_steps)
    @goto iteration_loop

    #30
    @label set_convergence
    convergence_factor.value = ((d1_norm.value + b1_norm.value)
                                / max(ltp.betkm1, epsilon ^ 2))
    number_of_equality_constraints.value = number_of_active_constraints.value
    rank.value = rank_c2.value + rank_a.value
    return

    @label user_stop
    exit.value = error.value
end
"""

"""
function easy_nlcls!(first_approximation::Array{Float64},
                    number_of_residuals::Int, number_of_constraints::Int,
                    number_of_equality_constraints::Int,
                    residuals!::Function, constraints!::Function,
                    jac_residuals!::Function, jac_constraints!::Function)
    rpc = number_of_residuals + number_of_constraints
    act = min(2 * number_of_constraints,
              number_of_constraints + length(first_approximation))
    penalty_weights = Array{Float64}(undef, number_of_constraints)
    current_residuals = Array{Float64}(undef, rpc)
    current_constraints = Array{Float64}(undef, number_of_constraints)
    active_constraints = Array{Int}(undef, act)
    res = easy_nlcls!(first_approximation, number_of_residuals, residuals!,
                     constraints!, number_of_equality_constraints,
                     penalty_weights, current_residuals, current_constraints,
                     active_constraints, jac_residuals, jac_constraints)
    return res
end
    
"""
Easy to call version of NLSIP
residuals!(x, f) and constraints!(x, h) must compute the
residuals/constraints at x and return the result in f/ h

    THE REQUIRED LENGTH OF THE SUPPLIED ARRAY
    first_approximation is length : the number of parameters
    current_residuals is length : number_of_residuals + number_of_constraints
    penalty_weight is length : number_of_constraints
    current_constraints is length : number_of_constraints
    active_constraints is length : min(2*number_of_constraints,
                                 number_of_constraints + number_of_parameters)
    
"""
function easy_nlcls!(first_approximation::Array{Float64},
                    number_of_residuals::Int,
                    residuals!::Function,
                    constraints!::Function,
                    number_of_equality_constraints::Int,
                    penalty_weights::Array{Float64},
                    current_residuals::Array{Float64},
                    current_constraints::Array{Float64},
                    active_constraints::Array{Int},
                    jac_residuals!::Union{Function, Nothing}=nothing,
                    jac_constraints!::Union{Function, Nothing}=nothing)
    number_of_active_constraints = Number_wrapper(number_of_equality_constraints)
    exit = Number_wrapper(0)
    number_of_parameters = length(first_approximation)
    number_of_constraints = length(current_constraints)
    min_l_n = min(number_of_constraints, number_of_parameters)
    #validation

    if (length(penalty_weights) != number_of_constraints
        || length(active_constraints) != min_l_n + number_of_constraints
        || (length(current_residuals) !=
            number_of_residuals + number_of_constraints))
        exit.value = -1
        return
    end
    #default values
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
    eps_constraints = sqrt(eps(Float64))

    #---------------------
    #we wrap the constraints in residuals function in other function
    # usable by nonlinear_least_squares
    function hfunc!(current_point::Array{Float64}, number_of_parameters::Int,
                   current_constraints::Array{Float64},
                   number_of_constraints::Int, ctrl::Number_wrapper{Int},
                   jacobian::Array{Float64, 2}, leading_dim_c::Int,)
        if ctrl.value == 1
            constraints!(current_point, current_constraints)
            #if uncomputable ctrl == -1
            return
        elseif ctrl.value == -1
            constraints!(current_point, current_constraints)
            #if uncomputable ctrl = -10
            return
        elseif ctrl.value == 2
            if jac_constraints! != nothing
                jac_constraints!(current_point, jacobian)
            else
                #maybe calculate jac right here
                ctrl.value = 0
            end
            return
        end
    end
    function ffunc!(current_point::Array{Float64}, number_of_parameters::Int,
                   current_residuals::Array{Float64},
                   number_of_residuals::Int, ctrl::Number_wrapper{Int},
                   jacobian::Array{Float64, 2}, leading_dim_c::Int,)
        if ctrl.value == 1
            residuals!(current_point, current_residuals)
            #if uncomputable ctrl == -1
            return
        elseif ctrl.value == -1
            residuals!(current_point, current_residuals)
            #if uncomputable ctrl = -10
            return
        elseif ctrl.value == 2
            if jac_residuals! != nothing
                jac_residuals!(current_point, jacobian)
            else
                #maybe calculate jac right here
                ctrl.value = 0
            end
            return
        end
    end
    #---------------

    number_of_iterations = Number_wrapper{Int}(0)
    number_of_eval = Number_wrapper{Int}(0)
    number_of_jac_eval = Number_wrapper{Int}(0)
    number_of_hessian_eval = Number_wrapper{Int}(0)
    number_of_linesearch_eval = Number_wrapper{Int}(0)
    rank_a = Number_wrapper{Int}(0)
    rank_ac = Number_wrapper{Int}(0)
    objective_at_termination = Number_wrapper{Float64}(0.)
    convergence_factor = Number_wrapper{Float64}(0.)
    ltp = Last_two_points(0, 0, 0, 0, 0, 0, 0, 0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    restart_steps = Restart_steps(0.0, 0.0, 0, 0)
    ifree = Number_wrapper{Int}(0)
    for i in 1:number_of_constraints
        penalty_weights[i] = 0.0
    end
    mpl = number_of_residuals + number_of_constraints
    p1 = Array{Int}(undef, min_l_n)
    p2 = Array{Int}(undef, min_l_n)
    p3 = Array{Int}(undef, number_of_parameters)
    inactive_constraints = Array{Int}(undef, number_of_constraints)
    p4 = Array{Int}(undef, number_of_constraints)
    old_penalty_weights = Array{Float64}(undef, number_of_constraints)
    s = Array{Float64}(undef, max(number_of_constraints, number_of_parameters))
    #in the original code b is length = min_l_n but some functions need
    # it to be length =  number_of_constraints
    b = Array{Float64}(undef, number_of_constraints)
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
                           number_of_residuals, number_of_active_constraints,
                           number_of_constraints, tol, penalty_weights,
                           eps_relative, eps_absolute, eps_point,
                           eps_constraints, iprint, nout, max_iteration, p_norm,
                           scale, hessian, ffunc!, hfunc!, exit,
                           objective_at_termination, number_of_iterations,
                           number_of_eval, number_of_jac_eval,
                           number_of_hessian_eval, number_of_linesearch_eval,
                           rank_a, rank_ac, current_residuals,
                           current_constraints, active_constraints,
                           convergence_factor, p1, p2, p3,
                           inactive_constraints,
                           p4, b, d1, d2, d3, scaling_matrix,
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

end
