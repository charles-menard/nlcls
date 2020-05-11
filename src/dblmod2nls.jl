include("./struct.jl")
include("./dblredunls.jl")
"""
Replaces the subroutine GIVEN1
"""
function givens_rotation(z1::Float64, z2::Float64)
    if z2 != 0.0
        gamma = sqrt(z1 ^ 2 + z ^ 2)
        c = z1 / gamma
        s = z2 / gamma
        sig = gamma
        return c, s, sig
    end
    c = copysign(1.0, z1)
    s = 0.0
    sig = abs(z1)
    return c, s, sig
end

"""
Replaces the subroutine NLSNIP
"""
function nonlinear_least_square(current_point::AbstractArray{Float64},
                                number_of_parameters::Int64,
                                leading_dim_c::Int64, leading_dim_a::Int64,
                                leading_dim_g::Int64, leading_dim_f::Int64,
                                number_of_residuals::Int64,
                                (number_of_equality_constraints::
                                 Number_wrapper{Int64}),
                                number_of_constraints::Int64, tol::Float64,
                                penalty_weights::AbstractArray{Float64},
                                eps_relative::Float64, eps_absolute::Float64,
                                eps_point::Float64, eps_constraints::Float64,
                                verbose::Int64, nout::Int64,
                                maximum_iteration::Int64, p_norm::Int64,
                                scale::Int64, hessian::Bool,
                                residuals!::Function, constraints!::Function,
                                exit::Number_wrapper{Int64},
                                current_objective::Number_wrapper{Float64},
                                iteration_number::Number_wrapper{Int64},
                                number_of_eval::Number_wrapper{Int64},
                                number_of_jac_eval::Number_wrapper{Int64},
                                number_of_hessian_eval::Number_wrapper{Int64},
                                number_of_linesearch_eval::Number_wrapper{Int64},
                                rank_a::Number_wrapper{Int64},
                                rank::Number_wrapper{Int64},
                                current_residuals::AbstractArray{Float64},
                                current_constraints::AbstractArray{Float64},
                                active_constraints::AbstractArray{Int64},
                                convergence_factor::Number_wrapper{Float64},
                                p1::AbstractArray{Int64},
                                p2::AbstractArray{Int64},
                                p3::AbstractArray{Int64},
                                inactive_constraints::AbstractArray{Int64},
                                p4::AbstractArray{Int64},
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
                                ifree::Number_wrapper{Int64})
    d1_norm = Number_wrapper{Float64}(0.)
    d_norm::Number_wrapper{Float64} = Number_wrapper(0.)
    b1_norm::Number_wrapper{Float64} = Number_wrapper(0.)
    beta_k = Number_wrapper(0.)
    rank_c2 = Number_wrapper{Int64}(0)
    gn_direction_norm = Number_wrapper(0.)
    gres = Number_wrapper{Float64}(0.)
    deleted_constraints = Number_wrapper{Bool}(false)
    added_constraints = Number_wrapper(false)
    time = Number_wrapper{Int64}(0)
    sigmin = Number_wrapper{Float64}(0.)
    absvmx = Number_wrapper{Float64}(0.)
    wh_norm = Number_wrapper(0.0)
    gradient_norm = 0.0
    code = Number_wrapper{Int64}(0)
    eval = Number_wrapper{Int64}(0)
    dim_a = Number_wrapper{Int64}(0)
    dim_c2 = Number_wrapper(0)
    steplength = Number_wrapper{Float64}(0.)
    lower_bound_steplength = Number_wrapper{Float64}(0)
    upper_bound_steplength = Number_wrapper{Float64}(0)
    current_psi = Number_wrapper(0.0)
    index_upper_bound = Number_wrapper{Int64}(0)
    number_of_active_constraints = Number_wrapper{Int64}(0)
    lmt = Number_wrapper{Int64}(0)
    number_of_householder = Number_wrapper{Int64}(0)
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
Replaces the subroutine TERCRI
"""
function termination_criteria(rank_a::Int64, rank_c2::Int64, error::Int64,
                              number_of_equality_constraints::Int64,
                              nuber_of_active_constraints::Int64,
                              number_of_parameters::Int64, restart::Bool,
                              deleted_constraints::Bool, maximum_iteration::Int64,
                              iteration_number::Int64, sq_sum_residuals::Float64,
                              d1_norm::Float64, sq_sum_constraints::Float64,
                              gres::Float64, current_point_norm::Float64,
                              gradient_norm::Float64, alfnoi::Float64,
                              current_constraints::AbstractArray{Float64},
                              inactive_constraints::AbstractArray{Int64},
                              number_of_inactive_constraints::Int64,
                              x_diff::Float64,
                              eps_absolute::Float64, eps_relative::Float64,
                              eps_current_point::Float64,
                              eps_current_constraints::Float64,
                              smallest_lagrange_mult::Float64,
                              largest_lagrange_mult::Float64,
                              w_old::AbstractArray{Float64}, current_point::AbstractArray{Float64},
                              ath_norm::Float64, wh_norm::Float64,
                              exit::Number_wrapper{Int64},
                              ltp::Last_two_points, restart_steps::Restart_steps,
                              ifree::Number_wrapper{Int64})

    exit.value = 0
    feas = 1

    if ((restart || ifree.value > 3)
        || (ltp.kodkm1 == -1 && alfnoi <= 0.25)
        || (error < 0 && error != -2)
        || deleted_constraints
        || sqrt(sq_sum_constraints) > eps_current_constraints
        || gres > sqrt(eps_relative) * (1.0 + gradient_norm))
        @goto abnormal_termination_criteria
    end

    if number_of_inactive_constraints > 0
        j = 0
        for i = 1:number_of_inactive_constraints
            j = inactive_constraints[i]
            if current_constraints[j] <= 0.0
                feas = -1
            end
        end
    end
    factor = 0.0
    if number_of_active_constraints != number_of_equality_constraints
        if number_of_active_constraints == 1
            factor = 1.0 + sq_sum_residuals
        elseif number_of_active_constraints > 1
            factor = largest_lagrange_mult
        end
        if smallest_lagrange_mult < eps_relative * factor
            @goto abnormal_termination_criteria
        end
    end

    if d1_norm ^ 2 <= eps_relative ^ 2 * sq_sum_residuals
        exit.value += 10000
    end
    if sq_sum_residuals <= eps_absolute ^ 2
        exit.value += 2000
    end
    if x_diff < eps_current_point * current_point_norm
        exit.value += 300
    end
    if alfnoi > 0.25 || error == -2
        exit.value += 40
    end
    if exit.value == 0
        @goto abnormal_termination_criteria
    end

    if ltp.kodkm1 != 1
        exit.value += 2
        exit.value *= feas
        return
    end

    if (rank_a != number_of_active_constraints
        || rank_c2 != (number_of_parameters - number_of_active_constraints))
        exit.value += 1
        exit.value *= feas
        return
    end

    if ltp.kodkm2 != 1 && iteration_number >= 2
        exit.alue += 3
        exit.value *= feas
        return
    end
    if (!(abs(ltp.alfkm2 - 1.0) <= 1.0e-2)
        && abs(ltp.alfkm1 - 1.0) <= 1.0e-2)
        exit.value += 4
    end
    ext.value *= feas
    return

    @label abnormal_termination_criteria

    if iteration_number + restart_steps.nrrest > maximum_iteration
        exit.value = -2
        return
    end
    if ifree.value > 0
        exit.value = 0
        return
    end
    if error == -1
        exit.value = -6
        return
    end
    if (x_diff <= eps_current_point * 10.0
        && ath_norm <= eps_current_constraints * 10.0
        && wh_norm >= 1.0)
        exit.value = -10
        return
    end
    if error >= -5 && error<= -3
        exit.value = error
    end
    if error == -9
        return
    end
    exit.value = error
    copyto!(current_point, 1, w_old, 1, number_of_parameters)
end

"""
Replaces the subroutine STPLNG
"""
function compute_steplength(restart::Bool, current_point::AbstractArray{Float64},
                    inactive_constraints_gradient::AbstractArray{Float64, 2},
                            leading_dim_gradient::Int64,
                            search_direction::AbstractArray{Float64},
                    number_of_parameters::Int64,
                            current_residuals::AbstractArray{Float64},
                            v1::AbstractArray{Float64},
                    number_of_residuals::Int64, sq_sum_residuals::Float64,
                    residuals!::Function, rank_c2::Int64, code::Int64,
                            current_constraints::AbstractArray{Float64},
                            constraints!::Function,
                    active_constraints::AbstractArray{Int64},
                    number_of_active_constraints::Int64,
                            inactive_constraints::AbstractArray{Int64},
                            p4::AbstractArray{Int64},
                    number_of_inactive_constraints::Int64,
                            number_of_constraints::Int64,
                            penalty_weights::AbstractArray{Float64},
                    old_penalty_weights::AbstractArray{Float64}, dim_a::Int64,
                    p_norm::Int64, number_of_householder::Int64,
                    psi::Number_wrapper{Float64},
                    steplength::Number_wrapper{Float64},
                    lower_bound_steplength::Number_wrapper{Float64},
                    upper_bound_steplength::Number_wrapper{Float64},
                    number_of_eval::Number_wrapper{Int64},
                    x_diff::Number_wrapper{Float64}, error::Number_wrapper{Int64},
                    index_upper_bound::Number_wrapper{Int64},
                    wh_norm::Number_wrapper{Float64},
                            next_residuals::AbstractArray{Float64},
                            v2::AbstractArray{Float64},
                            next_constraints::AbstractArray{Float64},
                            g::AbstractArray{Float64},
                    ltp::Last_two_points, restart_steps::Restart_steps,
                    wsave::AbstractArray{Float64, 2})

    c1 = 1.0e-3
    error.value = 0
    old = false
    lower_bound_steplength.value = c1

    if abs(code) != 1
        @goto undamped_step
    end
    if !restart
        copyto!(old_penalty_weights, 1, penalty_weights, 1, number_of_constraints)
    else
        old = true
    end
    psi0 = Number_wrapper{Float64}(0.)
    dpsi0 = Number_wrapper{Float64}(0.)
    atwa = Number_wrapper{Float64}(0.)
    cdx = Number_wrapper{Float64}(0.)
    dxtctf = Number_wrapper{Float64}(0.)
    compute_penalty_weights(penalty_weights, old_penalty_weights, v1,
                            current_residuals,
                            sq_sum_residuals, number_of_residuals,
                            active_constraints, number_of_active_constraints,
                            current_constraints, number_of_constraints, p_norm,
                            dim_a, restart, wh_norm,
                            psi0, dpsi0, atwa, cdx, dxtctf, next_residuals,
                            next_constraints,
                            v2, wsave)
    if dpsi0.value >= 0.0
        error.value = -1
    end
    if error.value < 0
        return
    end

    upper_bound_steplength(inactive_constraints_gradient, leading_dim_gradient,
                           number_of_inactive_constraints,
                           number_of_householder, number_of_parameters,
                           current_constraints, p4, v1, number_of_residuals,
                           search_direction, upper_bound_steplength,
                           index_upper_bound)
    lower_bound_steplength.value = upper_bound_steplength.value / 3000.0
    magfy = 3.0
    if rank_c2 < ltp.rkckm1
        magfy *= 2.0
    end
    steplength.value = 2.0 * min(1.0, magfy * ltp.alfkm1, upper_bound_steplength)
    exit = Number_wrapper{Int64}(0)

    while true
        steplength.value *= 0.5
        linesearch_steplength(current_point, search_direction, current_residuals,
                              v1, number_of_residuals, number_of_parameters,
                              steplength, psi0.value, dpsi0.value,
                              lower_bound_steplength.value, residuals!,
                              constraints!, current_constraints, next_constraints,
                              active_constraints, number_of_active_constraints,
                              inactive_constraints,
                              number_of_inactive_constraints,
                              number_of_constraints, penalty_weights,
                              upper_bound_steplength.value, next_residuals,
                              v2, g, psi, x_diff.value, number_of_eval, exit)
        if exit.value < -10
            error.value = exit.value
            return
        elseif exit.value != -3
            break
        end
    end
    error.value = exit.value
    #50
    upper_bound = min(1.0, upper_bound_steplength)
    ltp.prelin = upper_bound * (-2.0 * dxtctf.value - upper_bound * cdx.value
                                + (2.0 - upper_bound ^ 2) * atwa.value)
    ltp.pgress = (2.0 * psi0.value
                  - BLAS.nrm2(number_of_residuals, current_residuals, 1) ^ 2)
    if number_of_active_constraints <= 0
        @goto check_upper_bound
    end
    whsum = 0.0
    k = 0
    for i = 1:number_of_active_constraints
        k = active_constraints[i]
        whsum += penalty_weight[i] * current_constraints[i] ^ 2
    end
    ltp.pgress -= whsum
    @goto check_upper_bound
    @label undamped_step
    upper_bound_steplength.value = 3.0
    sum = 0.0
    x_i = 0.0
    for i = 1:number_of_parameters
        x_i = current_point[i]
        current_point[i] = x_i + search_direction[i]
        sum += (x_i - current_point[i]) ^ 2
    end
    x_diff.value = sqrt(sum)

    ctrl = Number_wrapper(-1)
    dummy = Array{Float64, 2}
    residuals!(current_point, number_of_parameters, current_residuals,
               number_of_residuals, ctrl, dummy, 1)
    exit.value = ctrl.value
    if exit.value < -10
        error.value = exit.value
        return
    end

    ctrl.value = -1
    constraints!(current_point, number_of_parameters, current_constraints,
                 number_of_constraints, ctrl, dummy, 1)
    steplength.value = 1.0
    number_of_eval.value = 1
    @label check_upper_bound
    if index_upper_bound.value == 0
        return
    end
    if abs(steplength.value - upper_bound_steplength.value) > 0.1
        index_upper_bound.value = 0
    end
    return

end

"""
Replaces the subroutine EUCNRM
"""
function update_penalty_euc_norm(va::AbstractArray{Float64},
                                current_constraints::AbstractArray{Float64},
                                active_constraints::AbstractArray{Int64},
                                number_of_active_constraints::Int64,
                                mu::Float64, va_norm::Float64, b_norm::Float64,
                                 dim_a::Int64,
                                 penalty_weights::AbstractArray{Float64},
                                number_of_constraints::Int64,
                                 pset::AbstractArray{Int64},
                                 y::AbstractArray{Float64},
                                old_penalty_weights::AbstractArray{Float64},
                                wsave::AbstractArray{Float64, 2})

    if number_of_active_constraints <= 0
        return
    end
    ztw = 0.0
    for i = 1:number_of_constraints
        penalty_weights[i] = wsave[i, 4]
    end
    j = 0
    for i = 1:number_of_active_constraints
        j = active_constraints[i]
        ztw += va[i] ^ 2 * penalty_weights[j]
    end
    ztw *= va_norm ^ 2
    if ztw >= mu
        if number_of_active_constraints == dim_a
            assort(wsave, 100, number_of_constraints, 4,
                   number_of_active_constraints,
                   active_constraints, penalty_weights, old_penalty_weights)
            return

        end
        ctrl = 2
        nrp = 0
        gamma = 0.0
        y_el = 0.0
        for i = 1:number_of_active_constraints
            j =active_constraints[i]
            y_el = va[i] * va_norm * (va[i] * va_norm + b_norm
                                     * current_constraints[j])
            if y_el > 0.0
                nrp += 1
                pset[nrp] = j
                y[nrp] = y_el
                continue
            end
            gamma -= y_el * penalty_weights[j]
        end
        minimize_euclidean_norm(ctrl, penalty_weights, number_of_constraints,
                                pset, nrp, y, gamma, old_penalty_weights)
        assort(wsave, 100, number_of_constraints, 4, number_of_active_constraints,
               active_constraints, penalty_weights, old_penalty_weights)
        return
    end

    if number_of_active_constraints != dim_a
        ctrl = 2
        nrp = 0
        tau = mu
        y_el = 0.0
        gamma = 0.0
        for i = 1:number_of_active_constraints
            j = active_constraints[i]
            y_el = -va[i] * current_constraints[j] * va_norm * b_norm
            if y_el > 0.0
                nrp += 1
                pset[nrp] = j
                y[nrp] = y_el
                continue
            end
            tau -= y_el * penalty_weights[j]
        end
        minimize_euclidean_norm(ctrl, penalty_weights, number_of_constraints,
                            pset, nrp, y, tau, old_penalty_weights)
        assort(wsave, 100, number_of_constraints, 4, number_of_active_constraints,
               active_constraints, penalty_weights, old_penalty_weights)
        return
    end

    ctrl = 1
    for i = 1:number_of_active_constraints
        pset[i] = active_constraints[i]
        y[i] = va[i] ^ 2 * va_norm ^ 2
    end
    println("---")
    println("LENGTH PSET, T ", length(pset), " ", number_of_active_constraints)
    minimize_euclidean_norm(ctrl, penalty_weights, number_of_constraints,
                            pset, number_of_active_constraints, y, mu,
                            old_penalty_weights)
    assort(wsave, 100, number_of_constraints, 4, number_of_active_constraints,
           active_constraints, penalty_weights, old_penalty_weights)
    return
end

"""
Replaces the subroutine MAXNRM
"""
function update_penalty_max_norm(ata::Float64, rmy::Float64, alfa::Float64,
                                 delta::Float64,
                                 penalty_weights::AbstractArray{Float64},
                                 active_constraints::AbstractArray{Int64},
                                 number_of_active_constraints::Int64,
                                 wsave::AbstractArray{Float64, 2})
    mu = rmy / ata
    if abs(alfa - 1.0) <= delta
        mu = 0.0
    end
    l = active_constraints[1]
    wkm1 = penalty_weights[l]
    nu = max(mu, wsave[1, 4])
    for i = 1:number_of_active_constraints
        penalty_weights[active_constraints[i]] = nu
    end
    if mu <= wkm1
        return
    end
    for i = 1:4
        if mu <= wsave[1, i]
           continue
        end
        for j = 4:i+1:-1
            wsave[1,j] = wsave[1, j-1]
        end
        wave[1,i] = mu
    end
end

"""
Replaces the subroutine ASSORT
"""
function assort(u::AbstractArray{Float64, 2}, leading_dim_u::Int64, l::Int64, s::Int64,
                t::Int64, active_constraints::AbstractArray{Int64},
                penalty_weights::AbstractArray{Float64},
                old_penalty_weights::AbstractArray{Float64})
    k = 0
    w_k = 0.0
    for x = 1:t
        k = active_constraints[x]
        w_k = penalty_weights[k]
        for y = 1:s
            if w_k <= u[k, y]
                continue
            end
            for z = s:y+1:-1
                u[k, z] = u[k, z-1]
            end
            u[k, y] = w_k
            break
        end
    end
end

"""
Replaces the subroutine INIALC
"""
function init_working_set(number_of_equality_constraints::Int64,
                          number_of_constraints::Int64,
                          current_constraints::AbstractArray{Float64},
                          active_constraints::AbstractArray{Int64},
                          number_of_active_constraints::Number_wrapper{Int64},
                          bnd::Int64, p_norm::Int64,
                          inactive_constraints::AbstractArray{Int64},
                          lmt::Number_wrapper{Int64},
                          penalty_weights::AbstractArray{Float64},
                          old_penalty_weights::AbstractArray{Float64},
                          exit::Number_wrapper{Int64}, wsave::AbstractArray{Float64, 2})

    number_of_active_constraints.value = number_of_equality_constraints
    lmt.value = 0
    if number_of_constraints == 0
        return
    end
    delta = 0.1
    epsilon = 0.01
    lmin = min(number_of_constraints, 100)
    for i = 1:lmin, j = 1:4
        wsave[i, j] = delta
    end

    sum = 0.0
    abs_constraints = 0.0
    pos = 0.0
    for i = 1:number_of_constraints
        abs_constraints = abs(current_constraints[i])
        if current_constraints[i] > 0.0
            pos = current_constraints[i]
        else
            pos = min(abs_constraints + epsilon, delta)
        end
        old_penalty_weights[i] = pos
        penalty_weights = pos
        sum += pos
    end

    if p_norm == 0
        sum /= Float64(number_of_constraints)
        for i = 1:number_of_constraints
            old_penalty_weights[i] = sum
            penalty_weights[i] = sum
        end
    end
    if number_of_equality_constraints != 0
        for i = 1:number_of_equality_constraints
            active_constraints[i] = i
        end
    end
    pp1 = number_of_equality_constraints + 1
    if pp1 > number_of_constraints
        return
    end
    j = 0
    for i = pp1:number_of_constraints
        j = bnd + i - number_of_equality_constraints
        if current_constraints[i] > 0.0
            lmt.value += 1
            inactive_constraints[lmt.value] = i
            active_constraints[j] = 0
            continue
        end
        number_of_active_constraints.value += 1
        active_constraints[number_of_active_constraints.value] = i
        active_constraints[j] = 1
    end
    if number_of_active_constraints > bnd
        exit.value = -8
    end
end

"""
Replaces the subroutine GNDCHK
"""
function choose_search_method(b1_norm::Float64, d1_norm::Float64, d_norm::Float64,
                              sq_sum_constraints::Float64,
                              iteration_number::Int64, restart::Bool,
                              d1apm1::Float64, added_constraints::Bool,
                              number_of_residuals::Int64,
                              number_of_parameters::Int64,
                              constraint_deleted::Bool,
                              active_constraints::AbstractArray{Int64},
                              number_of_equality_constraints::Int64,
                              number_of_active_constraints::Int64,
                              lagrange_mult::AbstractArray{Float64},
                              inactive_constraints::AbstractArray{Int64},
                              number_of_inactive_constraints::Int64,
                              current_constraints::AbstractArray{Float64},
                              eps_relative::Float64, rank_a::Int64, scale::Int64,
                              scaling_matrix::AbstractArray{Float64},
                              beta_k::Number_wrapper{Float64},
                              method::Number_wrapper{Int64},
                              ltp::Last_two_points, restart_steps::Restart_steps,
                              ifree::Number_wrapper{Int64})

    delta = 0.1
    c1, c2, c3, c4, c5 = 0.5, 0.1, 4.0, 10.0, 0.05
    beta_k.value = sqrt(d1_norm ^ 2 + b1_norm ^ 2)
    pp1 = number_of_equality_constraints + 1
    nmt = number_of_parameters - number_of_active_constraints
    qpt = number_of_active_constraints + number_of_inactive_constraints

    if !restart || ltp.kodkm1 != 2
        method.value = 1
        if ifree.value > 0
            ifree.value -= 1
        end
        if (ifree.value == 4
           || iteration_number == 0 && !restart)
            return
        end
        if ltp.kodkm1 != -1
            if (constraint_added || constraint_deleted
                || beta_k.value < c1 * ltp.betkm1
                || ltp.pgress > c2 * ltp.prelin && d_norm <= c3 * beta_k.value)
                return
            end
        end
    end

    method.value = -1
    if ifree.value > 0
        return
    end
    if ltp.kodkm1 == 2 && !constraint_deleted
        method.value = 2
        return
    end
    nlk = sqrt(d1_norm ^ 2 + sq_sum_constraints)
    nlkm1 = sqrt(d1apm1 ^ 2 + sq_sum_constraints)
    neg = false
    pp1 = number_of_equality_constraints + 1
    if pp1 <= number_of_active_constraints
        sq_rel = sqrt(eps(Float64))
        row_i = 0.0
        for i = pp1:number_of_active_constraints
            row_i = scaling_matrix[i]
            if scale != 0
                row_i = 1.0 / scaling_matrix[i]
            end
            if -sq_rel < lagrange_mult[i] * row_i
                continue
            end
            if lagrange_mult[i] < 0.0
                neg = true
            end
        end
    end
    if number_of_inactive_constraints > 0
        j = 0
        for i = 1:number_of_inactive_constraints
            j = active_constraints
            if current_constraints[j] < delta
                neg = true
            end
        end
    end
    #50
    if (sq_sum_constraints > c2
        || constraint_added
        || constraint_deleted
        || neg
        || (number_of_active_constraints == number_of_parameters
            && number_of_active_constraints == rank_a))
        return
    end
    if (qpt != number_of_equality_constraints
        && rank_a >= number_of_active_constraints)
        epsilon= max(1.0e-2, 10.0 * eps_relative)
        if (!beta_k.value < epsilon * d_norm
            || (b1_norm < epsilon && number_of_residuals == nmt))
            return
        end
    end
    #55
    if restart
        if nlkm1 > c2 * nlk
            return
        end
        method.value = 2
        return
    end
    if ((ltp.alfkm1 >= c5 || nlkm1 < c2 * nlk)
        && number_of_residuals != nmt)
        if d_norm <= c4 * beta_k.value
            return
        end
        method.value = 2
        return
    end
end

"""
Replaces the subroutine SUBSPC
"""
function subspace_dimension(restart::Bool, sq_sum_residuals::Float64,
                            c::AbstractArray{Float64, 2}, leading_dim_c::Int64,
                            number_of_residuals::Int64,
                            number_of_parameters::Int64, rank_c2::Int64,
                            current_residuals::AbstractArray{Float64}, p3::AbstractArray{Int64},
                            d3::AbstractArray{Float64}, a::AbstractArray{Float64, 2},
                            leading_dim_a::Int64,
                            number_of_active_constraints::Int64, rank_a::Float64,
                            sq_sum_constraints::Float64, p1::AbstractArray{Int64},
                            d2::AbstractArray{Float64}, p2::AbstractArray{Int64},
                            b::AbstractArray{Float64}, fmat::AbstractArray{Float64, 2},
                            leading_dim_f::Int64, pivot::AbstractArray{Float64},
                            gmat::AbstractArray{Float64, 2}, leading_dim_g::Int64,
                            d::AbstractArray{Float64}, dx::AbstractArray{Float64},
                            work_area::AbstractArray{Float64},
                            dim_a::Number_wrapper{Int64},
                            dim_c2::Number_wrapper{Int64}, ltp::Last_two_points,
                            restart_steps::Restart_steps,)
    #r11td1, r22td2 SAVE ?
    k = 0
    beta1 = 0.1
    beta2 = 0.1
    lower_alpha = 0.2
    dim_a.value = 0
    rtd_norm = Number_wrapper{Float64}(0.)
    d3_i_dummy = Number_wrapper{Float64}(0.)
    c_ik_dummy = Number_wrapper{Float64}(0.)
    for i = 1:number_of_residuals
        d[i] = -current_residuals[i]
    end

    etaa = Number_wrapper(1.0)
    etac = Number_wrapper{Float64}(0.)
    if rank_a <= 0
        #goto 50
    end
    if (number_of_active_constraints <= rank_a
        && restart_steps.nrrest <= 1)
        l_to_upper_triangular(a, leading_dim_a, rank_a,
                              number_of_active_constraints, b,
                              leading_dim_g, p2, gmat, d2)
        v_times_p(c, number_of_residuals, rank_a, p2)
        r11td1 = r_transpose_times_d_norm(a, leading_dim_a, rank_a, b)
        for i = 1:rank_c2
            k = rank_a + i
            d3_i_dummy.value = d3[i]
            c_ik_dummy.value = c[i, k]
            householder_transform(2, i, i+1, number_of_residuals, view(c, 1, k),
                                  1, d3_i_dummy, d, 1, number_of_residuals,
                                  1, c_ik_dummy)
            d3[i] = d3_i_dummy.value
            c[i, k] = c_ik_dummy.value
        end
        r22td2 = r_transpose_times_d_norm(view(c, :, rank_a+1), leading_dim_c,
                                          rank_c2, d)
        for i = 1:number_of_residuals
            d[i] = -current_residuals[i]
        end
    end
    copyto!(dx, 1, b, 1, rank_a)
    #30

    drkm1 = abs(ltp.rkakm1) + number_of_active_constraints - ltp.tkm1
    b1aspr = BLAS.nrm2(drkm1, b, 1)

    hpgrs = ltp.hsqkm1 - sq_sum_constraints
    if !restart && r11td1 < beta1 * r22td2
        dim_a.value = 0
    else
        compute_solving_dim(restart, drkm1, hpgrs, rank_a, gmat,
                            leading_dim_g, dx, work_area,
                           ltp.b1km1, b1aspr, ltp.alfkm1, dim_a, etaa)
    end

    copyto!(dx, 1, b, 1, rank_a)
    upper_trianguar_solve(dim_a, gmat, dx)
    d_minus_c1_times_x1(dim_a, number_of_residuals, d, c, dx)
    #50
    dim_c2.value = rank_c2
    
    if rank_c2 > 0
        for i = 1:rank_c2
            k = rank_a + i
            d3_i_dummy.value = d3[i]
            c_ik_dummy.value = c[i, k]
            householder_transform(2, i, i+1, number_of_residuals, view(c, :, k),
                                  1, d3_i_dummy, d, 1, number_of_residuals, 1,
                                  c_ik_dummy)
            d3[i] = d3_i_dummy.value
            c[i, k] = c_ik_dummy
        end
    end
    dukm1 = abs(ltp.rkckm1) + ltp.tkm1 - number_of_active_constraints
    d1aspr = BLAS.nrm2(dukm1, d, 1)
    fpgrs = ltp.fsqkm1 - sq_sum_residuals

    if !restart && r22td2 < beta2 * r11td1
        dim_c2.value = 0
        etac.value = 1.0
    else
        compute_solving_dim(restart, dukm1, fpgrs, rank_c2,
                            view(c, :, rank_a+1:number_of_parameters),
                            leading_dim_c, d, work_area, ltp.d1km1, d1aspr,
                            ltp.alfkm1, dim_c2, etac)
    end
    if !restart
       @goto check_lower_alpha
    end
    #???
    if (restart_steps.nrrest <= 1
        || ltp.pgress > restart_steps.bestpg)
            restart_steps.bestpg = ltp.pgress
            ltp.rkckm2 = ltp.rkckm1
            ltp.rkakm2 = ltp.rkakm1
    end

    if etaa.value >= etac.value
        if etaa.value == etac.value && rank_a > 1
            ltp.rkakm1 -= 1
            dim_c2.value = max(0, ltp.rkckm - 1)
            dim_a.value = ltp.rkakm1
            if dim_c2.value == 0 && rank_a == 1
                dim_a.value = 0
            end
        end
    else
        dim_c2.value = ltp.rkckm1
        dim_a.value = max(ltp.rkakm1 - 1, 0)
    end
    #110
    if dim_c2.value <= 0 && dim_a.value <= 0
        dim_c2.value = ltp.rkckm2
        dim_a.value = ltp.rkakm2
        restart_steps.lattry = 0
    end
    @label check_lower_alpha
    if restart
        return
    end
    if ltp.alfkm1 >= lower_alpha
        dim_a.value = max(dim_a.value, drkm1)
        dim_c2.value = max(dim_c2.value, dukm1)
    end
end

"""
Replaces the subroutine RTD
"""
function r_transpose_times_d_norm(r::AbstractArray{Float64, 2}, leading_dim_r::Int64,
                                  n::Int64, d::AbstractArray{Float64})
    sum = 0.0
    sum2 = 0.0
    for j = 1:n
        sum2 = 0.0
        for i = 1:j
            sum2 += r[i, j] * d[i]
        end
        sum += sum2 ^ 2
    end
    return sqrt(sum)
end

"""
Replaces the subroutine DIMUPP
"""
function compute_solving_dim(restart::Bool, dim_latest_step::Int64,
                             obj_latest_step::Float64,
                             rank_a::Int64, a::AbstractArray{Float64, 2},
                             leading_dim_a::Int64, b::AbstractArray{Float64},
                             work_area::AbstractArray{Float64},
                             predicted_reduction_latest::Float64,
                             predictd_reduction_current::Float64,
                             steplength_latest_step::Float64,
                             suggested_dim::Number_wrapper{Int64},
                             eta::Number_wrapper{Float64})
    c1 = 0.1
    c2 = 0.01
    stepb = 0.2
    pgb1 = 0.3
    pgb2 = 0.1
    suggested_dim.value = rank_a
    eta.value - 1.0
    if rank_a <= 0
        return
    end

    work_area[1] = abs(b[1])
    b[1] = abs(b[1] / a[1, 1])
    if rank_a != 1
        for i = 2:rank_a
            work_area[i] = b[i]
            b[i] /= a[i, i]
            b[i] = norm(b[i-1:i])
            r[i] = norm(b[i-1:i])
        end
    end
    rn = r[rank_a]
    sn = b[rank_a]

    d_sum = 0.0
    psi_max = 0.0
    mindim = 0
    psi = 0.0
    for i = 1:rank_a
        dsum += b[i] ^ 2
        psi = sqrt(d_sum) * abs(a[i, i])
        if psi > psi_max
            psi_max = psi
            mindim = i
        end
    end

    k = mindim
    if !restart
        ik = Number_wrapper{Int64}(0)
        if dim_latest_step == rank_a || dim_latest_step <= 0
            gn_previous_step(b, sn, r, rn, mindim, rank_a, ik)
        else
            subspace_minimization(b, r, rn, c1, rank_a, dim_latest_step,
                                  obj_latest_step, predicted_reduction_latest,
                                  predicted_reduction_current,
                                  steplength_latest_step, ik)
        end
        suggested_dim.value = max(mindim, ik.value)
        return
    end
    suggested_dim.value = min(rank_a, dim_latest_step)
    suggested_dim.value = max(suggest_dim.value, 0)
    if suggest_dim == 0
        return
    end
    k = max(dim_latest_step - 1, 1)
    if b[suggested_dim.value] != 0.0
        eta.value = b[k] / b[suggested_dim.value]
    end
end

"""
Replaces the subroutine MULEST
"""
function estimate_lagrange_mult(time::Number_wrapper{Int64},
                                a::AbstractArray{Float64, 2},
                                leading_dim_a::Int64,
                                number_of_active_constraints::Int64,
                                number_of_parameters::Int64,
                                gradient::AbstractArray{Float64},
                                bv::AbstractArray{Float64},
                                deleted_column::Int64, tol::Float64,
                                d1::AbstractArray{Float64},
                                h::AbstractArray{Float64, 2},
                                leading_dim_h::Int64,
                                pivot::AbstractArray{Float64},
                                p::AbstractArray{Int64}, scale::Int64,
                                scaling_matrix::AbstractArray{Float64},
                                lagrange_mult::AbstractArray{Float64},
                                rank_a::Number_wrapper{Int64},
                                residu::Number_wrapper{Float64},
                                s::AbstractArray{Float64},
                                u::AbstractArray{Float64},
                                w::AbstractArray{Float64})
    println("number of active constraints begining of mulest, ",
            number_of_active_constraints)

    ctrl = time.value
    time.value += 1
    rank_a.value = number_of_active_constraints
    residu.value = 0
    k1 = 0
    k2 = 0
    #both eta and com1 are used in the original code without being
    #initialized so i don't know what value to start them at
    co = 0.
    si = 0.
    eta = Number_wrapper(0.)
    com1 = Number_wrapper(0.)
    if ctrl >= 2
        @goto compute_givens_rotations
    end
    #100
    if number_of_active_constraints == 0
        residu.value = BLAS.nrm2(number_of_parameters, g, 1)
        return
    end
    a_to_lower_triangular(number_of_active_constraints, number_of_parameters,
                          p, a, leading_dim_a, tol, rank_a, d1, gradient)
    for i = 1:rank_a.value
        pivot[i] = a[i, i]
    end
    p_transpose_times_v(p, number_of_active_constraints, bv)
    solve_t_times_t(a, rank_a.value, gradient, lagrange_mult,
                    number_of_parameters, residu)
    ip1 = rank_a.value + 1
    if rank_a.value != number_of_active_constraints
        for i = ip1:number_of_active_constraints
            lagrange_mult[i] = 0.0
        end
    end
    nonzero_first_order_lm(a, rank_a.value, number_of_active_constraints,
                           bv, lagrange_mult, w, u)
    p_times_v(p, number_of_active_constraints, lagrange_mult)
    @goto back_transform

    @label compute_givens_rotations
    if number_of_active_constraints == 0
        return
    end
    if ctrl < 3
        for i = 1:number_of_parameters
            for k = 1:number_of_parameters
                h[i, k] = 0.0
            end
           h[i, i] = 1.0
        end
    end
    tp1 = number_of_active_constraints + 1
    if deleted_column == tp1
        @goto choose_pseudo_rank
    end
    pi = eps(Float64)
    ist = deleted_column
    for i = deleted_column:number_of_active_constraints
        ip1 = i + 1
        k1 = i
        k2 = i
        if i > deleted_column
            k1 += 1
        end
        if ip1 > deleted_column
            k2 += 1
        end
        co, si, a[i,i] = givens_rotation(a[k1, i], a[k2, ip1])
        if ip1 <= number_of_active_constraints
            for k = ip1:number_of_active_constraints
                k1 = k
                k2 = k
                if i > deleted_column
                    k1 += 1
                end
                if ip1 > deleted_column
                    k2 += 1
                end
                if si != 0.0
                    apply_rotation(co, si, view(a, k1, i),
                                   view(a, k2, ip1))
                end
                a[k, i] = a[k1, i]
            end
        end

        if si == 0.0
            #eta = ???
            u[i] = eta.value
            w[i] = -com1.value / eta.value
            if ctrl == 2
                multiply_matrices_and_transpose(h, leading_dim_h, s, u, w, ist, i)
                ist = i + 1
                continue
            end
        end
        apply_rotation(co, si, view(gradient, i), view(gradient, ip1))

        s[i] = si
        if i <= ist
            u[i] = co / pi
            w[i] = pi
            eta.value = si / pi
            com1.value = co
            continue
        end
        #240
        u[i] = co * eta.value
        w[i] = -com1.value / eta.value
        eta.value *= si
        com1.value = co
    end #250

    u[tp1] = eta.value
    w[tp1] = -com1.value / eta.value
    if ctrl == 2
       multiply_matrices_and_transpose(h, leading_dim_h, s, u, w, ist, tp1)
    end
    @label choose_pseudo_rank
    k = 0
    for i = 1:number_of_active_constraints
        if abs(a[i, i] < tol)
            break
        end
        k += 1
    end
    rank_a.value = k

    solve_t_times_t(a, rank_a.value, gradient, lagrange_mult, number_of_parameters,
                    residu)
    nonzero_first_order_lm(a, rank_a.value, number_of_active_constraints, bv,
                           lagrange_mult, w, u)
    copyto!(w, 1, lagrange_mult, 1, number_of_active_constraints)
    permute(number_of_active_constraints, p, w, lagrange_mult)

    @label back_transform
    if scale == 0
        return
    end
    for i = 1:number_of_active_constraints
        lagrange_mult[i] *= scaling_matrix[i]
    end
end

"""
Replaces the subroutine WEIGHT
"""
function compute_penalty_weights(penalty_weights::AbstractArray{Float64},
                         old_penalty_weights::AbstractArray{Float64},
                                 v1::AbstractArray{Float64},
                                 current_residuals::AbstractArray{Float64},
                         sq_sum_residuals::Float64, number_of_residuals::Int64,
                         active_constraints::AbstractArray{Int64},
                         number_of_active_constraints::Int64,
                         current_constraints::AbstractArray{Float64},
                         number_of_constraints::Int64, p_norm::Int64,
                         dim_a::Int64, restart_in_old::Bool,
                         wh_norm::Number_wrapper{Float64},
                         psi_at_zero::Number_wrapper{Float64},
                         psi_derivative_at_zero::Number_wrapper{Float64},
                         atwa::Number_wrapper{Float64},
                         ctc::Number_wrapper{Float64},
                         ctd::Number_wrapper{Float64},
                         next_residuals::AbstractArray{Float64},
                                 next_constraints::AbstractArray{Float64},
                                 v2::AbstractArray{Float64},
                         wsave::AbstractArray{Float64, 2})
    delta = 0.25
    if restart_in_old
        copyto!(penalty_weights, 1, old_penalty_weights, 1, number_of_constraints)
    end
    mpt = number_of_residuals + number_of_active_constraints
    @views a_norm = BLAS.nrm2(number_of_residuals,
                       v1[number_of_residuals+1:end], 1)
    a_norm_sq = a_norm ^ 2
    scale_vector(v1, a_norm, number_of_residuals+1, dim_a )
    btwa = 0.0
    atwa.value = 0.0
    b_norm = 0.0
    k = 0
    mpi = 0
    if dim_a > 0.0
        for i = 1:dim_a
            k = active_constraints[i]
            mpi = number_of_residuals + i
            atwa.value += penalty_weights[k] * v1[mpi] ^ 2
            b_norm = max(abs(current_constraints[k]), b_norm)
        end
        scale_vector(current_constraints, b_norm, 1, number_of_constraints)
        atwa.value *= a_norm ^ 2
        for i = 1:dim_a
            k = active_constraints[i]
            mpi = number_of_residuals + i
            btwa += penalty_weights[k] * v1[mpi] * current_constraints[k]
        end
        btwa *= b_norm * a_norm
    end

    c_norm = BLAS.nrm2(number_of_residuals, v1, 1)
    c_norm_sq = c_norm ^ 2
    scale_vector(v1, c_norm, 1, number_of_residuals)
    d_norm = sqrt(sq_sum_residuals)
    scale_vector(current_residuals, d_norm, 1, number_of_residuals)
    ctd = 0.0
    for i = 1:number_of_residuals
        ctd += v1[i] * current_residuals[i]
    end
    ctd *= c_norm * d_norm

    if atwa.value + c_norm_sq == 0.0
        alfa = 0.0
    else
        alfa = (-btwa - ctd) / (atwa.value + c_norm_sq)
    end

    psi_at_zero.value = sq_sum_residuals * 0.5
    psi_derivative_at_zero = ctd
    atwa.value = 0.0
    whsum = 0.0
    if number_of_active_constraints <= 0
        @goto scale_all
    end
    scale_vector(v1, 1.0/a_norm, number_of_residuals+1, dim_a)
    scale_vector(v1, a_norm, number_of_residuals+1, number_of_active_constraints)
    rmy = abs(-ctd - c_norm_sq) / delta - c_norm_sq
    pset = Array{Int64}(undef, number_of_active_constraints)
    if p_norm == 0
        update_penalty_max_norm(a_norm_sq, rmy, alfa, delta, penalty_weights,
                                active_constraints, number_of_active_constraints,
                                wsave)
    else
        @views update_penalty_euc_norm(v1[number_of_residuals+1:mpt],
                                current_constraints, active_constraints,
                                number_of_active_constraints, rmy, a_norm,
                                b_norm, dim_a, penalty_weights,
                                number_of_constraints, pset, next_constraints,
                                v2, wsave)

    end

    btwa = 0.0
    for i = 1:number_of_active_constraints
        k = active_constraints[i]
        mpi = number_of_residuals + i
        whsum += penalty_weights[k] * current_constraints[k] ^ 2
        btwa += penalty_weights[k] * v1[mpi] * current_constraints[k]
        atwa.value += penalty_weights[k] * v1[mpi] ^ 2
    end
    whsum *= b_norm ^ 2
    btwa *= a_norm * b_norm
    atwa.value *= a_norm ^ 2
    psi_at_zero.value = 0.5 * (whsum + sq_sum_residuals)
    psi_derivative_at_zero = btwa + ctd

    @label scale_all
    wh_norm.value = whsum
    scale_vector(v1, 1.0/a_norm, number_of_residuals+1,
                 number_of_active_constraints)
    scale_vector(current_constraints, 1/b_norm, 1, number_of_constraints)
    scale_vector(v1, 1.0/c_norm, 1, number_of_residuals)
    scale_vector(current_residuals, 1.0/d_norm, 1, number_of_residuals)
end

"""
Replaces the subroutine WRKSET
in the dblwrkset.f file
"""
function update_active_constraints(a::AbstractArray{Float64, 2},
                                   leading_dim_a::Int64,
                                   (number_of_active_constraints::
                                    Number_wrapper{Int64}),
                                   number_of_equality_constraints::Int64,
                                   number_of_parameters::Int64,
                                   gradient_objective::AbstractArray{Float64},
                                   minus_active_constraints::AbstractArray{Float64},
                                   tau::Float64, leading_dim_fmat::Int64,
                                   scale::Int64, iteration_number::Int64,
                                   scaling_matrix::AbstractArray{Float64},
                                   active_constraints::AbstractArray{Int64},
                                   min_l_n::Int64,
                                   inactive_constraints::AbstractArray{Int64},
                                   (number_of_inactive_constraints::
                                    Number_wrapper{Int64}),
                                   current_constraints::AbstractArray{Float64},
                                   gn_steplength_norm::Number_wrapper{Float64},
                                   p4::AbstractArray{Int64},
                                   jac_residuals::AbstractArray{Float64, 2},
                                   leading_dim_jac_residuals::Int64,
                                   number_of_residuals::Int64,
                                   current_residuals::AbstractArray{Float64},
                                   leading_dim_gmat::Int64,
                                   current_point::AbstractArray{Float64},
                                   constraints!::Function, residuals!::Function,
                                   number_of_eval::Number_wrapper{Int64},
                                   number_of_jac_eval::Number_wrapper{Int64},
                                   p2::AbstractArray{Int64},
                                   p3::AbstractArray{Int64},
                                   gn_search_direction::AbstractArray{Float64},
                                   v1::AbstractArray{Float64},
                                   d2::AbstractArray{Float64},
                                   d3::AbstractArray{Float64},
                                   rank_c2::Number_wrapper{Int64},
                                   d1_norm::Number_wrapper{Float64},
                                   d_norm::Number_wrapper{Float64},
                                   b1_norm::Number_wrapper{Float64},
                                   d::AbstractArray{Float64},
                                   gmat::AbstractArray{Float64, 2},
                                   p1::AbstractArray{Int64},
                                   v::AbstractArray{Float64},
                                   d1::AbstractArray{Float64},
                                   fmat::AbstractArray{Float64},
                                   rank_a::Number_wrapper{Int64},
                                   gres::Number_wrapper{Float64},
                                   number_of_householder::Number_wrapper{Int64},
                                   (deleted_constraints_plus2::
                                    Number_wrapper{Int64}),
                                   deleted_constraints::Number_wrapper{Bool},
                                   pivot::AbstractArray{Float64},
                                   v2::AbstractArray{Float64},
                                   s::AbstractArray{Float64},
                                   u::AbstractArray{Float64},
                                   ltp::Last_two_points,
                                   restart_steps::Restart_steps,)

    j = Number_wrapper(0)
    user_stop = Number_wrapper{Int64}(0)
    number_of_constraints = (number_of_active_constraints.value
                                           + number_of_inactive_constraints.value)
    deleted_constraints_plus2.value = 1
    tol = sqrt(Float64(number_of_active_constraints.value)) * tau
    del = false

    estimate_lagrange_mult(deleted_constraints_plus2, a, leading_dim_a,
                            number_of_active_constraints.value,
                            number_of_parameters, gradient_objective,
                            minus_active_constraints, (j.value), tol, d1, fmat,
                            leading_dim_fmat, pivot, p1, scale, scaling_matrix,
                            v, rank_a, gres, s, u, v2)
    number_of_householder.value = rank_a.value
    noeq = Number_wrapper{Int64}(0)
    if (number_of_residuals - number_of_active_constraints.value
        > number_of_parameters)
        sign_ch(deleted_constraints_plus2.value, p1, v,
                number_of_active_constraints.value, min_l_n, ltp.d1km1,
                gn_steplength_norm, iteration_number, scale, scaling_matrix,
                gres.value, current_constraints, number_of_equality_constraints,
                number_of_constraints, j.value, noeq, u, v2)
    end
    if (noeq.value != 0
        || (number_of_residuals - number_of_active_constraints.value
            > number_of_parameters))
        del = true
        reorder(a, number_of_active_constraints, number_of_parameters,
                minus_active_constraints, j.value, noeq.value, active_constraints,
                inactive_constraints, number_of_inactive_constraints,
                p4, u, scale, scaling_matrix)
        estimate_lagrange_mult(deleted_constraints_plus2, a, leading_dim_a,
                                number_of_active_constraints.value,
                                number_of_parameters, gradient_objective,
                                minus_active_constraints, j.value, tol,
                                d1, fmat, leading_dim_fmat, pivot, p1,
                                scale, scaling_matrix, v, rank_a, gres, s, u, v2)
        gn_search(deleted_constraints_plus2.value, a, leading_dim_a,
                  number_of_active_constraints.value, number_of_parameters,
                  d1, p1, rank_a.value, number_of_householder.value,
                  minus_active_constraints, fmat, leading_dim_fmat,
                  jac_residuals, leading_dim_jac_residuals, number_of_residuals,
                  current_residuals, pivot, tau, leading_dim_gmat, scale,
                  scaling_matrix, inactive_constraints,
                  number_of_inactive_constraits.value, p4, p2, p3,
                  gn_search_direction, v1, d2, d3, rank_c2, d1_norm, d_norm,
                  b1_norm, d, s, u, gmat)
        i2 = inactive_constraints[number_of_inactive_constraints.value]
        i3 = number_of_residuals + number_of_constraints
        if (v1[i3] >= -current_constraints[i2]
            & v1[3] > 0.0)
            return
        end
        del = false
        new_point(current_point, number_of_parameters, current_constraints,
                  number_of_constraints, current_residuals,
                  number_of_residuals, constraints!, residuals!,
                  leading_dim_a, leading_dim_jac_residuals,
                  number_of_eval, a, jac_residuals, minus_active_constraints,
                  d, user_stop)
        number_of_jac_eval.value += 1
        j.value = number_of_inactive_constraints.value
        add_constraints(active_constraints, inactive_constraints,
                        number_of_active_constraints,
                        number_of_inactive_constraints, j.value)
        equal(minus_active_constraints, number_of_constraints, a,
              leading_dim_a, number_of_parameters, active_constraints,
              number_of_active_constraints.value, number_of_equality_constraints,
              p4)
        gradient(jac_residuals, number_of_residuals, number_of_parameters,
                 current_residuals, gradient_objective)
        scale_system(scale, a, leading_dim_a, number_of_active_constraints.value,
                     number_of_parameters, minus_active_constraints,
                     scaing_matrix)
        unscramble_array(active_constraints, min_l_n, number_of_constraints,
                         number_of_equality_constraints)
        j.value = 0
        deleted_constraints_plus2.value = 1
        estimate_lagrange_mult(deleted_constraints_plus2, a, leading_dim_a,
                                number_of_active_constraints.value,
                               number_of_parameters, gradient_objective,
                               minus_current_constraints,
                                j.value, tol, d1, fmat, leading_dim_fmat,
                                pivot, p1, scale, scaling_matrix, v, rank_a,
                                gres, s, u, v2)
    end
    gn_search(deleted_constraints_plus2.value, a, leading_dim_a,
              number_of_active_constraints.value, number_of_parameters, d1, p1,
              rank_a.value, number_of_householder.value, minus_active_constraints,
              fmat, leading_dim_fmat, jac_residuals, leading_dim_jac_residuals,
              number_of_residuals,
              current_residuals, pivot, tau, leading_dim_gmat,
              scale, scaling_matrix, inactive_constraints,
              number_of_inactive_constraints.value, p4, p2, p3,
              gn_search_direction, v1, d2, d3, rank_c2, d1_norm, d_norm, b1_norm,
              d, s, u, gmat)
    #think i've got the condition to replace the goto 80, 100 correct
    if (ltp.kodkm1 != 1
        || number_of_active_constraints.value != rank_a.value
        || rank_c2.value != min(number_of_residuals,
                                number_of_parameters - rank_a.value))
        return
    end
    special_lagrange_mult(a, number_of_active_constraints.value,
                           current_residuals, number_of_residuals, v1,
                           jac_residuals, p1, scale, scaling_matrix, s, v, gres)
    sign_ch(deleted_constraints_plus2.value, p1, v,
            number_of_active_constraints.value, active_constraints,
            min_l_n, d1_norm.value, d_norm.value, iteration_number, scale,
            scaling_matrix, gres.value, current_constraints,
            number_of_equality_constraints, number_of_constraints, j, noeq,
            s, v2)
    if noeq.value == 0
        return
    end
    del = true
    reorder(a, number_of_active_constraints, number_of_parameters,
            minus_active_constraints, j.value, noeq.value, active_constraints,
            inactive_constraints, number_of_inactive_constraints, p4, u, scale,
            scaling_matrix)
    estimate_lagrange_mult(deleted_constraints_plus2, a, leading_dim_a,
                            number_of_active_constraints.value,
                           number_of_parameters, gradient_objective,
                           minus_active_constraints,
                            j.value, tol, d1, fmat, leading_dim_fmat,
                            pivot, p1, scale, scaling_matrix, v, rank_a,
                            gres, s, u, v2)
    user_stop.value = 2
    residuals!(current_point, number_of_parameters, current_residuals,
               number_of_residuals, user_stop, jac_residuals,
               leading_dim_jac_residuals)
    if user_stop.value == 0
        jacobian_forward_diff(current_point, number_of_parameters,
                             current_residuals, residuals!, jac_residuals,
                             leading_dim_jac_residuals, d, user_stop)
        number_of_eval.value += number_of_parameters
    end
    number_of_jac_eval.value += 1

    gn_search(deleted_constraints_plus2.value, a, leading_dim_a,
               number_of_active_constraints.value, number_of_parameters, d1, p1,
               rank_a.value, number_of_householder.value, minus_active_constraints,
               fmat, leading_dim_fmat, jac_residuals, leading_dim_jac_residuals,
              number_of_residuals, current_residuals, pivot, tau,
              leading_dim_gmat, scale, scaling_matrix, inactive_constraints,
               number_of_inactive_constraints.value, p4, p2, p3,
               gn_search_direction, v1, d2, d3, rank_c2, d1_norm, d_norm, b1_norm,
               d, s, u, gmat)
end
