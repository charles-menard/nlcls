using Printf
using LinearAlgebra
include("struct.jl")

"""
Replaces the subroutine NEWPNT
Compute the jacobian of the constraints and the residuals and put them
respectively in `jac_constraints` and `jac_residuals`

"""

function new_point(current_point::Array{Float64}, number_of_parameters::Int,
                   current_constraints::Array{Float64},
                   number_of_constraints::Int, current_residuals::Array{Float64},
                   number_of_residuals::Int, constraints!::Function,
                   residuals!::Function, leading_dim_jac_constraints::Int,
                   leading_dim_jac_residuals::Int,
                   number_of_eval::Number_wrapper{Int},
                   jac_constraints::Array{Float64, 2},
                   jac_residuals::Array{Float64, 2},
                   b::Array{Float64}, d::Array{Float64},
                   user_stop::Number_wrapper{Int})

    ctrlc = Number_wrapper(2)
    residuals!(current_point, number_of_parameters, current_residuals,
               number_of_residuals, ctrlc, jac_residuals,
               leading_dim_jac_residuals)

    if ctrlc.value < -10
        user_stop.value = ctrlc.value
        return
    end
    if ctrlc.value == 0
        jacobian_forward_diff(current_point, number_of_parameters,
                               current_residuals, number_of_residuals,
                               residuals!, jac_residuals,
                               leading_dim_jac_residuals, d, user_stop)
        if user_stop < -10
            return
        end
        number_of_eval.value += number_of_parameters
    end
    if number_of_constraints <= 0
        return
    end
    ctrla = Number_wrapper(2)

    constraints!(current_point, number_of_parameters, current_constraints,
                 number_of_constraints, ctrla, jac_constraints,
                 leading_dim_jac_constraints)

    println("ctrla value ::::", ctrla.value)
    if ctrla.value < -10
        user_stop.value = ctrla.value
        return
    end
    if ctrlc == 0
        jacobian_forward_diff(current_point, number_of_parameters,
                              current_constraints, number_of_constraints,
                              constraints!, jac_constraints,
                              leading_dim_jac_constraints, b, user_stop)
        if user_stop < -10
            return
        end
    end
    for i = 1:number_of_constraints
        b[i] = -current_constraints[i]
    end
end
"""
Replaces the subroutine EQUAL
"""
function equal(b::Array{Float64}, l::Int, a::Array{Float64,2},
                leading_dim_a::Int, n::Int, active_constraints::Array{Int64},
                t::Int, p::Int, p4::Array{Int64})
    if l > 0
        for i=1:l
            p4[i] = i
        end
    end
    if t <= 0 || t == p
        return
    end

    index= 1
    ip = 0
    ik = 0
    for i=1:t
        index = active_constraints[i]
        ip = p4[index]
        if ip == i
            continue
        end
        for j=1:n
            a[i,j], a[ip, j] = a[ip, j], a[i, j]
        end
        println("i, ip")
        println(i, " ", ip)
        b[i], b[ip]= b[ip], b[i]
        for j = 1:t
            if i != p4[j]
                continue
            end
            ik = j
        end
    end

end

"""
Replaces the subroutine EVSCAL
C     SCALE THE SYSTEM  A*DX = B    IF SO INDICATED BY FORMING
C     A@D=DIAG*A      B@D=DIAG*B
"""
function scale_system(scale::Int, jacobian::Array{Float64,2},
                      leading_dim_jacobian::Int,
                      number_of_active_constraints::Int64,
                      number_of_parameters::Int,
                      neg_constraints::Array{Float64},
                      scaling_matrix::Array{Float64})

    if number_of_active_constraints == 0
        return
    end

    current_norm = 0.0
    for i = 1:number_of_active_constraints
        row_norm = norm(jacobian[i, :])
            if row_norm > current_norm
                current_norm = row_norm
        end
        scaling_matrix[i] = row_norm
        if scale == 0
            continue
        end

        if row_norm == 0.0
            row_norm == 1.0
        end
        for j = 1:number_of_parameters
            jacobian[i,j] /= row_norm
        end
        neg_constraints[i] /= row_norm
        scaling_matrix[i] = 1.0 / row_norm
    end
end
"""
Replaces the subroutine GNSRCH
"""
function gn_search(fmat_is_identity::Int, a::Array{Float64,2},
                  leading_dim_a::Int, number_of_active_constraints::Int,
                  number_of_parameters::Int, d1::Array{Float64}, p1::Array{Int},
                  rank_a::Int, number_of_householder::Int, b::Array{Float64},
                  fmat::Array{Float64,2}, leading_dim_f::Int,
                  jac_residuals::Array{Float64,2},
                  leading_dim_jac_residuals::Int,
                  current_residuals::Array{Float64},pivot::Array{Float64},
                  tau::Float64, leading_dim_g::Int, scale::Int,
                  diag::Array{Float64}, inactive_constraints::Array{Int},
                  number_of_inactive_constraints::Int, p4::Array{Float64,2},
                  p2::Array{Int}, p3::Array{Int},
                  gn_direction::Array{Float64}, v1::Array{Float64},
                  d2::Array{Float64}, d3::Array{Float64},
                  rank_c2::Number_wrapper{Int}, d1_norm::Number_wrapper{Float64},
                  d_norm::Number_wrapper{Float64},
                  b1_norm::Number_wrapper{Float64}, d::Array{Float64},
                  work_area_s::Array{Float64}, work_area_u::Array{Float64},
                  gmat::Array{Float64,2})


    code = 1
    if number_of_active_constraints != rank_a
        code = -1
        l_to_upper_triangular(a, leading_dim_a, rank_a, number_of_parameters,
                              b, leading_dim_g, p2, gmat, d2)
    end

    c_q1_h_p2_product(fmat_is_identity, jac_residuals,
                      leading_dim_jac_residuals, number_of_parameters,
                      number_of_residuals, number_of_active_constraints,
                      rank_a, pivot, number_of_householder, a, leading_dim_a, d1,
                      fmat, leading_dim_f, p2, work_area_s)
    kc2 = rank_a + 1
    nmp = number_of_parameters - rank_a
    tol = sqrt(Float64(nmp)) * tau
    c2_to_upper_to_triangular(number_of_residuals, nmp,
                              view(jac_residuals, :, kc2:number_of_parameters),
                              leading_dim_c, tol, p3, rank_c2, d3)
    sub_search_direction(fmat_is_identity, a, leading_dim_a,
                         number_of_active_constraints, number_of_parameters,
                         d1, p1, rank_a, rank_a, number_of_householder,
                         b, fmat, leading_dim_f, jac_residuals,
                         leading_dim_jac_residuals,
                         number_of_residuals, current_residuals, pivot, gmat,
                         leading_dim_g, d2, p2, p3, d3, rank_c2.value,
                         inactive_constraints, number_of_inactives_constraints,
                         p4, rank_c2.value, code, scale, diag, d, gn_direction,
                         v1, d1_norm, d_norm, b1_norm, work_area_u)
end
"""
Replaces the subroutine MINMAX
"""
function min_max_langrange_mult(number_of_equality_constraints::Int64,
                                number_of_active_constraints::Int64,
                                estimated_lagrange_mult::Array{Float64},
                                scale::Int64,
                                scaling_matrix::Array{Float64},
                                smallest_ineq_mult::Number_wrapper{Float64},
                                max_mult::Number_wrapper{Float64} )
    if number_of_equality_constraints == number_of_active_constraints
        return
    end
    tol = sqrt(eps(Float64))
    smallest_ineq_mult = 1e6
    max_mult = 0.0
    current_scale = 0.0
    current_abs = 0.0

    for i = 1:number_of_active_constraints
        current_abs = abs(estimated_lagrange_mult[i])
        if current_abs > max_mult.value
            max_mult.value = current_abs
        end
        if i <= number_of_equality_constraints
            continue
        end
        current_scale = scaling_matrix[i]
        if scale != 0
            current_scale =1.0 / scaling_matrix[i]
        end
        if -tol < estimated_lagrange_mult*current_scale
            continue
        end
        if estimated_lagrange_mult < smallest_ineq_mult.value
            smallest_ineq_mult.value = estimated_lagrange_mult[i]
        end
    end
end
"""
replaces subroutine analys
"""
function check_last_step(iteration_number::Int64, restart::Bool,
                         code::Number_wrapper{Int64}, sq_sum_residuals::Float64,
                         d1_norm::Number_wrapper{Float64},
                         d_norm::Number_wrapper{Float64},
                         c::Array{Float64, 2}, leading_dim_c::Int64,
                         number_of_residuals::Int64, number_of_parameters::Int64,
                         rank_c2::Int64, d::Array{Float64},
                         current_residuals::Array{Float64}, p3::Array{Float64},
                         d3::Array{Float64}, active_constraints::Array{Int64},
                         estimated_lagrange_mult::Array{Float64},
                         inactive_constraints::Array{Int64},
                         number_of_inactive_constraints::Int64,
                         p4::Array{Int64}, deleted_constraints::Int64,
                         a::Array{Float64, 2}, leading_dim_a::Int64,
                         number_of_equality_constraints::Int64,
                         number_of_active_constraints::Int64,
                         rank_a::Int64, b1_norm::Number_wrapper{Float64},
                         sq_sum_constraints::Float64, number_of_householder::Int64,
                         d1::Array{Float64},
                         p1::Array{Int64}, d2::Array{Float64}, p2::Array{Int64},
                         b::Array{Float64}, h::Array{Float64},
                         number_of_constraints::Int64,
                         fmat::Array{Float64, 2}, leading_dim_f::Int64,
                         pivot::Array{Float64}, gmat::Array{Float64, 2},
                         leading_dim_g::Int64, residuals!::Function,
                         constraints!::Function,
                         current_point::Array{Float64}, hessian::Bool,
                         constraint_added::Bool, constraint_deleted::Bool,
                         scale::Int64, scaling_mat::Array{Float64, 2},
                         gn_direction::Array{Float64},
                         gn_direction_norm::Number_wrapper{Float64},
                         v1::Array{Float64}, relative_termination_tol::Float64,
                         error::Number_wrapper{Int64},
                         number_of_eval::Number_wrapper{Int64},
                         d1_plus_b1_norm::Number_wrapper{Float64},
                         dim_a::Number_wrapper{Int64},
                         dim_c2::Number_wrapper{Int64}, v2::Array{Float64},
                         work_area::Array{Float64},
                         ltp::Last_two_points, restart_steps::Restart_steps,
                         ifree::Number_wrapper{Int64}, )

    ind = Number_wrapper(0)
    if !restart
        ind.value = ltp.rkckm1 + ltp.tm1 - number_of_active_constraints
        ind.value -= 1
        @views ltp.d1km2 = norm(d[1:ind.value])
    end
    choose_search_direction(b1_norm.value, d1_norm.value, d_norm.value,
                            sq_sum_constraints, iteration_number, restart,
                            ltp.d1km2, constraint_added, number_of_residuals,
                            number_of_parameters, constraint_deleted,
                            active_constraints, number_of_equality_constraints,
                            number_of_active_constraints,
                            estimated_lagrange_mult, inactive_constraints,
                            number_of_inactive_constraints, current_constraints,
                            eps_relative, rank_a, scale, scaling_mat,
                            d1_plus_b1_norm, ind, ltp, restart_steps, ifree)
    number_of_eval.value = 0
    code.value = ind
    error.value = 0
    if ind == 1
        dim_a.value = rank_a
        dim_c2.value = rank_c2
        return
    end
    if ind != 2
        subspace_dimension(restart, sq_sum_residuals, c, leading_dim_c,
                           number_of_residuals, number_of_parameters,
                           rank_c2, current_residuals, p3, d3, a,
                           leading_dim_a, number_of_active_constraints,
                           rank_a, sq_sum_constraints, p1, d2, p2, b, fmat,
                           leading_dim_f, pivot, gmat, leading_dim_g,
                           d, gn_direction, work_area, dim_a, dim_c2, ltp,
                           restart_steps)

        restart_steps.nrrest += 1
        sub_search_direction(deleted_constraints, a, leading_dim_a,
                             number_of_active_constraints, number_of_parameters,
                             d1, p1, dim_a.value, rank_a, number_of_householder,
                             b, fmat, leading_dim_f, c, leading_dim_c,
                             current_residuals, pivot, gmat, leading_dim_g,
                             d2, p2, p3, d3, dim_c2.value, inactive_constraints,
                             number_of_inactive_constraints, p4, rank_c2,
                             code, scale, scaling_matrix, d, gn_direction,
                             v1, d1_norm, d_norm, b1_norm, work_area)
        if dim_a.value == rank_a && dim_c2.value == rank_c2
            code.value = 1
            gn_direction_norm.value = norm(gn_direction)
        end

    end
    if !hessian
        error.value = -4

    end
    if ltp.kodkm1 != 2
        restart_steps.nrrest = 0
    end
    restart_steps.nrrest += 1
    newton_search_direction(residuals!, constraints!, current_point,
                            number_of_parameters, c, leading_dim_c,
                            number_of_residuals, rank_c2, current_residuals,
                            p3, d3, estimated_lagrange_mult,
                            a, leading_dim_a, active_constraints,
                            number_of_active_constraints, rank_a, d1, p1, p2, d2,
                            b, current_constraints, number_of_constraints,
                            leading_dim_f, pivot, gmat, leading_dim_g,
                            gn_direction, number_of_eval, error, fmat, d, v1, v2)
    dim_a.value = -number_of_active_constraints
    dim_c2.value = -number_of_parameters + number_of_active_constraints
    if restart_steps.nrrest > 5
        error.value = -9
    end
    gn_direction_norm.value = norm(gn_direction)
end
#------
"""
Replaces subroutine SUBDIR
"""
function sub_search_direction(deleted_constraints::Int64, a::Array{Float64, 2},
                              leading_dim_a::Int64,
                              number_of_active_constraints::Int64,
                              number_of_parameters::Int64,
                              d1::Array{Float64}, p1::Array{Int64}, dim_a::Int64,
                              rank_a::Int64, number_of_householder::Int64,
                              b::Array{Float64}, fmat::Array{Float64, 2},
                              leading_dim_f::Int64,
                              c::Array{Float64, 2}, leading_dim_c::Int64,
                              number_of_residuals::Int64,
                              current_residuals::Array{Float64},
                              pivot::Array{Float64}, gmat::Array{Float64,2},
                              leading_dim_g::Int64, d2::Array{Float64},
                              p2::Array{Int64}, p3::Array{Int64},
                              d3::Array{Float64}, dim_c2::Int64,
                              inactive_constraints::Array{Int64},
                              number_of_inactive_constraints::Int64,
                              p4::Array{Int64}, rank_c2::Int64, code::Int64,
                              scale::Int64, scaling_matrix::Array{Float64},
                              d::Array{Float64}, search_direction::Array{Float64},
                              v1::Array{Float64}, d1_norm::Number_wrapper{Float64},
                              d_norm::Number_wrapper{Float64},
                              b1_norm::Number_wrapper{Float64},
                              work_area::Array{Float64},)
    b1_norm.value = 0.0
    if number_of_active_constraints > 0
        copyto!(seach_direction, 1, b, 1,
                number_of_active_constraints)
        @views b1_norm.value = norm(search_direction[1:dim_a])
        z = number_of_residuals + number_of_active_constraints
        search_direction_product(code, gmat, leading_dim_g, rank_a, dim_a,
                                 p1, d2, search_direction,
                                 number_of_active_constraints,
                                 number_of_householder, scale, scaling_matrix,
                                 v1[number_of_residuals+1, z], work_area)
        if code == 1
            lower_triangular_solve(leading_dim_a, a, search_direction)
        else
            upper_triangular_solve(leading_dim_g, gmat, search_direction)
        end
    end
    for i = 1:number_of_residuals
        d[i] = -current_residuals[i]
    end
    d_minus_c1_times_x1(dim_a, number_of_residuals, d, c, leading_dim_c,
                        search_direction)
    d_norm = norm(d)
    k = 0
    if rank_c2 > 0
        for i=1:rank_c2
            k = rank_a + i
            @views householder_transform(2, i, i+1, number_of_residuals,
                                  c[:, k:number_of_parameters], 1,
                                  d3[i], d, 1,
                                  number_of_residuals, 1, c[i, k])
        end
    end
    @views d1_norm = d[1:dim_c2]
    k = rank_a + 1
    nmt = number_of_parameters - rank_a
    orthogonal_projection(c, leading_dim_c, number_of_residuals, dim_a,
                          search_direction, view(c, :, k:number_of_parameters),
                          rank_c2, dim_c2, d3,
                          d, v1)
    if nmt > 0
        copyto!(search_direction, rank_a+1, d, 1, nmt - 1)
        upper_triangular_solve(leading_dim_c, view(c, :, k:number_of_parameters),
                               view(search_direction, k:number_of_parameters))
        if dim_c2 != nmt
            k = dim_c2 + 1
            for i = k:nmt
                j = rank_a + i
                search_direction[i] = 0.0
            end
        end
    end
    #??????
    @views p_times_v(p2, nmt, search_direction[rank_a+1:number_of_parameters], 1)
    if code != 1
        p_times_v(p2, rank_a, search_direction, 1)
    end
    if deleted_constraints > 2 || (deleted_constraints !== 3 || length(b) != 0)
        h_times_x(fmat, leading_dim_f, number_of_parameters, search_direction,
                  work_area)
    end

    no::Int64 = number_of_householder - length(b)
    irow::Int64 = 0
    i2::Int64 = 0
    i3::Int64 = 0
    no_elem::Int64 = 0
    ait_search_direction::Float64 = 0.0
    if no > 0
        irow = length(b) + 1
        i2 = inactive_constraints[number_of_inactive_constraints]
        no_elem = p4[i2]
        for j=1:no_elem
            ait_search_direction += a[irow, j] * search_direction[j]
        end
        i3 = length(current_residuals) + length(inactive_constraints) + length(b)
        v1[i3] = ait_search_direction
        if scale != 0
            v1[i3] /= scaling_matrix[irow]
        end
    end

    if number_of_householder == 0
        return
    end
    d1_k_dummy  = Number_wrapper{Float64}(0.)
    pivot_k_dummy  = Number_wrapper{Float64}(0.)
    for i=1:number_of_householder
        k = number_of_householder - i + 1
        d1_k_dummy.value = d1[k]
        pivot_k_dummy.value = pivot[k]
        @views householder_transform(2, k, k+1, number_of_parameters,
                                     a[k:leading_dim_a, 1:end], leading_dim_a,
                                     d1[k], search_direction, 1,
                              leading_dim_a, 1,
                              pivot[k])
        d1[k], pivot[k] = d1_k_dummy.value, pivot_k_dummy.value
    end
end

"""
Replaces the subroutine UPBND
"""
function upper_bound_steplength(a::Array{Float64, 2},
                                leading_dim_a::Int64,
                                number_of_inactive_constraints::Int64,
                                number_of_householder::Int64,
                                number_of_active_constraints::Int64,
                                number_of_parameters::Int64,
                                current_constraints::Array{Float64},
                                inactive_constraints::Array{Int64},
                                p4::Array{Int64}, v1::Array{Float64},
                                number_of_residuals::Int64,
                                search_direction::Array{Float64},
                                upper_bound::Number_wrapper{Float64},
                                upper_bound_cons_index::Number_wrapper{Int64})

    upper_bound.value = 1.0e6
    upper_bound_cons_index.value = 0
    mpt = number_of_residuals  + number_of_active_constraints
    l = (number_of_inactive_constraints - number_of_householder
    + number_of_active_constraints)
    k = 0
    ip = 0
    ir = 0
    ait_search_direction = 0.0
    alf = 0.0
    change_ait_search_dir = true
    if number_of_inactive_constraints.value > 0
        for i=1:number_of_inactive_constraints.value
            k = inactive_constraints[i]
            ip = mpt + i
            ir = p4[k]
            if i <= l
                ait_search_direction = 0.0
                for j=1:number_of_parameters
                    ait_search_direction += a[ir, j] * search_direction[j]
                end
                v1[ip] = ait_search_direction
                change_ait_search_dir = false
            end
            if change_ait_search_dir
                ait_search_direction = v1[ip]
            end
            if ait_search direction >= 0.0 || -(curent_constraints[k] <= 0.0)
                continue
            end
            alf = -(current_constraints[k]) / ait_search_direction
            if alf > upper_bound
                continue
            end
            upper_bound.value = alf
            upper_bound_cons_index.value = k
        end
    end
    upper_bound.value = min(upper_bound.value, 3.0)
end
"""
Replaces subroutine EVADD
"""
function move_violated_constraints(current_constraints::Array{Float64},
                                   active_constraints::Array{Int64},
                                   number_of_active_constraints::Number_wrapper{Int64},
                                   min_l_n::Int64, number_of_equality::Int64,
                                   inactive_constraints::Array{Int64},
                                   number_of_inactive_constraints::Number_wrapper{Int64},
                                   ind::Int64, iteration_number::Int64,
                                   constraint_added::Number_wrapper{Bool})

    tol = sqrt(eps(Float64))
    delta = 0.1
    i = 1
    k = 0
    kind = 0
    jj = 0
    kk = 0
    max_constraint = 0.0
    while i <= number_of_active_constraints.value
        k = inactive_constraints[i]
        if (inactive_constraints[k] >= tol &&
            (k == ind || current_constraints[k] >= delta))
            i += 1
            continue
        end

        kind = 0
        if t >= min_l_n
            kind = 1
            max_constraint  = 0.0
            kk = 0
            for j = 1:number_of_active_constraints
                jj = active_constraints[j]
                if current_constraints[jj] > max_constraint
                        max_constraint, kk = current_constraint[jj], j
                end
            end
            if kk == 0
                max_constraint = 100.0
                for j=1:number_of_active_constraints
                    jj = active_constraints[j]
                    if abs(current_constraint[jj]) < max_constraint
                        max_constraint = abs(current_constraint)
                        kk = j
                    end
                end
            end
            delete_constraints(active_constraints, inactive_constraints,
                                   number_of_active_constraints,
                                   number_of_inactive_constraints, kk)
        end
        add_constraints(active_constraints, inactive_constraints,
                        number_of_active_constraints,
                        number_of_inactive_constraints, i)
        j = min_l_n + k - number_of_equality
        if active_constraints[j] == -1
            active_constraints[j] = iteration_number
        end
        if active_constraints[j] == 0
            active_constraints[j] = 1
        end
        constraint_added.value = true
        if kind == 1
            i = number_of_inactive_constraints.value + 1
        end
    end
end
    """
Replaces the subroutine SCALV
check if the factor is inf or zero to avoid arith error

    vector ./= factor
"""
function scale_vector(v::Array{Float64}, factor::Float64, start::Int64,
                      length::Int64)
    if factor == 0.0 || isinf(factor)
        return
    end
    for i in 1:length
        v[start + i - 1] /= factor
    end
end

"""
Replaces the subroutine EVREST
"""
function evaluation_restart(current_point::Array{Float64},
                            previous_point::Array{Float64},
                            number_of_parameters::Int64,
                            number_of_residuals::Int64,
                            iteration_number::Number_wrapper{Int64},
                            residuals!::Function,
                            number_of_eval::Number_wrapper{Int64},
                            current_residuals::Array{Float64}, d1_norm::Float64,
                            d_norm::Float64,
                            sq_sum_previous_residuals::Number_wrapper{Float64},
                            dim_c2::Int64, code::Int64,
                            search_direction_norm::Float64,
                            d1_plus_b1_norm::Float64, current_steplength::Float64,
                            lower_bound_steplength::Float64,
                            active_constraints::Array{Int64},
                            current_constraints::Array{Float64},
                            number_of_constraints::Int64,
                            number_of_active_constraints::Int64,
                            constraints!::Function, b1_norm::Float64,
                            sq_sum_constraints::Number_wrapper{Float64},
                            dim_a::Int64, error::Number_wrapper{Int64},
                            restart::Number_wrapper{Bool}, ltp::Last_two_points,
                            restart_steps::Restart_steps,
                            ifree::Number_wrapper{Int64})

    if (restart_steps.lattry != 0 || restart_steps.bestpg <= 0.0 ||
        (-1 != error.value && error.value > -3))
        if current_steplength <= lower_bound_steplength
            restart.value = true
        end
    end
    if restart.value != true
        iteration_number.value += 1
        if code.value != 2
            restart_steps.nrrest = 0
        end
        restart_steps.lattry = max(dim_c2, dim_a)
        ltp.betkm2 = ltp.betkm1
        ltp.d1km2 = ltp.d1km1
        ltp.dkm2 = ltp.dkm1
        ltp.fsqkm2 = ltp.fsqkm1
        ltp.b1km2 = ltp.b1km1
        ltp.hsqkm2 = ltp.hsqkm1
        ltp.dxnkm2 = ltp.dxnkm1
        ltp.alfkm2 = ltp.alfkm1
        ltp.rkakm2 = ltp.rkakm1
        ltp.rkckm2 = ltp.rkckm1
        ltp.tkm2 = ltp.tkm1
        ltp.kodkm2 = ltp.kodkm1
        ltp.betkm1 = d1_plus_b1_norm
        ltp.d1km1 = d1_norm
        ltp.dkm1 = d_norm
        ltp.fsqkm1 = sq_sum_residuals.value
        ltp.b1km1 = b1_norm
        ltp.hsqkm1 = sq_sum_constraints.value
        ltp.dxnkm1 = serch_direction_norm
        ltp.alfkm1 = current_steplength
        ltp.rkakm1 = dim_a
        ltp.rkckm1 = dim_c2
        ltp.tkm1 = number_of_active_constraints
        ltp.kodkm1 = code

        skip_count = false
        if -1 != error.value
            if error.value < 0
                return
            end
            sq_sum_constraints.value = sum_sq_active_constraints(
                current_constraints, active_constraints,
                number_of_active_constraints)
            ltp.hsqkm2 = sq_sum_constraints.value
            sq_sum_residuals.value = norm(current_residuals)^2
        else -1 == error.value
            skip_count = true
        end
    end
    #50
    if !skip_count
        restart_steps.nrrest += 1
        ltp.rkakm1 = dim_a
        rkckm1 = dim_c2
        if iteration_number.value == 0
            ltp.d1km1 = d1_norm
            ltp.b1km1 = b1_norm
        end
    end
    #55
    copyto!(latest_point, 1, previous_point, 1, number_of_parameters)
    if (abs(code)) == 2
        error.value =-5
    end
    ctrl = Number_wrapper(-1)
    dummy = Array{Float64, 2}
    residuals!(current_point, number_of_parameters, current_residuals,
               number_of_residuals, ctrl, dummy, 1)
    if ctrl.value < -10
        error.value = ctrl.value
    end
    ctrl.value = -1
    constraints!(current_point, number_of_parameters, current_constraints,
    number_of_active_constraints, ctrl, dummy, 1)
    if ctrl.value < -10
        error.value = ctrl.value
    end
    current_steplength = ltp.alfkm1
    number_of_eval.value += 1
    return
end
"""
Replaces the subroutine OUTPUT
"""
function output(printing_stepsize::Int64, iteration_number::Int64, unit::Int64,
    gres::Float64, penalty_weights::Array{Float64},
    active_constraints::Array{Int64}, convergence_factor::Number_wrapper{Float64},
    ltp::Last_two_points, restart_steps::Restart_steps, )
    if printing_stepsize <= 0
        return
    end
    itno = iteration_number - 1
    if itno / printing_stepsize * printing_stepsize != itno
        return
    end
    convergence_factor.value = 0.0
    if itno <= 0
        convergence_factor.value = ltp.betkm1 / ltp.betkm2
    end
    wmax = 0.0
    if ltp.tkm1 > 0
        for i = 1:ltp.tkm1
            j = active_constraints[i]
            if penalty_weights[j] > wmax
                wmax = penalty_weights[j]
            end
        end
    end
    #FIX THE ACTIVE SET PRINTING
    if itno == 0
        println(("\n\n\n\n COLLECTED INFORMATION FOR ITERATION STEPS" +
                "K   FSUM(K)   HSUM(K)   LAGRES   DXNORM    KODA   KODC   ALPHA" +
                "CONV.SPEED    MAX(W)    PREDICTED    REDUCTION"))
    end
    if ltp.tkm1 > 0
        @printf("%4d%e11.3%e11.3%e10.3%e10.3%4d%5d%e10.3%e10.3%e10.3%e10.3%e10.3",
        itno, ltp.fsqkm1, ltp.hsqkm1, gres, ltp.dxnkm1, ltp.rkakm1, ltp.rkckm1,
        ltp.alfkm1, convergence_factor.value, wmax, ltp.prelin,
        ltp.prgress)
    else
        @printf("%4d%e11.3%e11.3%e10.3%e10.3%4d%5d%e10.3%e10.3%e10.3%e10.3%e10.3",
        itno, ltp.fsqkm1, ltp.hsqkm1, gres, ltp.dxnkm1, ltp.rkakm1, ltp.rkckm1,
        ltp.alfkm1, convergence_factor.value, wmax, ltp.prelin, ltp.prgress)
    end
end


"""
Replaces the subroutine ADX
"""

function search_direction_product(code::Int64, gmat::Array{Float64, 2},
                                  leading_dim_g::Int64,
                                  rank_a::Int64, number_of_non_zero_in_b::Int64,
                                  p1::Array{Int64}, d2::Array{Float64},
                                  b::Array{Float64},
                                  number_of_active_constraints::Int64,
                                  number_of_householder::Int64,
                                  scale::Int64, scaling_matrix::Array{Float64},
                                  product::Array{Float64},
                                  work_area::Array{Float64})
    k = 0
    if dim_a != length(b)
        k = dim_a + 1
        for e in b[k:end]
            e = 0.0
        end
    end
    copyto!(product, 1, b, 1, number_of_active_constraints)
    if code != 1
        for i = 1:rank_a
            k = rank_a + 1 - i
            @views householder_transform(2, k, k+1, number_of_active_constraints,
                                  gmat[:, k:number_of_parameters], 1,
                                  d2[k], product,
                                  1, number_of_active_constraints, 1,
                                  gmat[k, k])
        end
    end
    if number_of_active_constraints < number_of_householder
        permute(number_of_active_constraints, p1, product, work_area)
        copyto!(product, 1, work_area, 1, number_of_active_constraints)
    else
        p_times_v(p1, number_of_active_constraints, product,
                  number_of_active_constraints, 1)
    end
    if scale <= 0
        return
    end
    product ./= scaling_matrix
    return
end

"""
Replaces the subroutine LTOUP

"""
function l_to_upper_triangular(a::Array{Float64, 2}, leading_dim_a::Int64,
                               rank_a::Int64, number_of_parameters::Int64,
                               b::Array{Float64}, leading_dim_g::Int64,
                               p2::Array{Int64},
                               gmat::Array{Float64,2}, d2::Array{Float64})

    p2[1:rank_a] = [1:rank_a]
    for i=1:number_of_parameters
        for j=1:rank_a
            if i < j
                gmat[i, j] = 0.0
            else
                gmat[i, j] = a[i, j]
            end
        end
    end
    cmax = Number_wrapper{Int64}(0)
    collng  = Number_wrapper{Float64}(0.)
    for i=1:rank_a
        max_partial_row_norm(number_of_parameters, rank_a, gmat, leading_dim_g,
                             i, i, cmax, collng)
        p2[i] = cmax
        permute_columns(gmat, leading_dim_g, number_of_parameters, i, cmax.value)
        @views householder_transform(1, i, i+1, number_of_parameters,
                              gmat[:, i:number_of_parameters],
                              1, d2[i], gmat[:, i+1:end], 1,
                              leading_dim_g, rank_a-i, gmat[i, i])
        @views householder_transform(2, i, i+1, number_of_parameters,
                              gmat[ :, i:number_of_parameters],
                              1, d2[i], b, 1, number_of_parameters, 1,
                              gmat[i, i])
    end
    return

end

"""
Replaces the subroutine ATOLOW
"""
function a_to_lower_triangular(number_of_rows_a::Int64, length_g::Int64,
                               p1::Array{Int64},
                               a::Array{Float64, 2}, leading_dim_a::Int64,
                               tol::Float64, pseudo_rank_a::Number_wrapper{Int64},
                               d1::Array{Float64},
                               current_gradient::Array{Float64})
    
    pseudo_rank_a.value = number_of_rows_a
    if number_of_rows_a == 0
        return
    end
    ldiag = min(number_of_rows_a, length_g)
    p1[1:ldiag] = [1:ldiag]
    krank = 0
    imax = Number_wrapper{Int64}(0)
    rmax  = Number_wrapper{Float64}(0.)
    for i = 1:ldiag
        krank = i
        max_partial_row_norm(number_of_rows_a, length_g, a, leading_dim_a,
                             i, i, imax, rmax)
        if rmax.value >= tol
            break
        end
        p1[i] = imax.value
        permute_row(a, leading_dim_a, length_g, i, imax.value)
        @views householder_transform(1, i, i+1, length_g, a[i:end, :],
                                     leading_dim_a,
                                     d1[i], a[i+1:end, :], leading_dim_a, 1,
                                     t-i, a[i, i])
        @views householder_transform(2, i, i+1, length_g_, a[i:end, :],
                                     leading_dim_a,
                                     d1[i], g, 1, length_g, 1, a[i, i])
        krank = i + 1
    end
    pseudo_rank_a = krank - 1
    return
end
"""
check how to pass view
Replaces subroutine C2TOUP
"""

function c2_to_upper_trianguar(number_of_rows_c2::Int64, number_of_col_c2::Int64,
                               c2::Array{Float64, 2}, leading_dim_c2::Int64,
                               tol::Float64, p3::Array{Int64},
                               pseudo_rank_c2::Number_wrapper{Int64},
                               d3::Array{Float64})
    pseudo_rank_c2.value = min(number_of_rows_c2, number_of_col_c2)
    if number_of_col_c2 == 0 || number_of_rows_c2 == 0
        return
    end
    p3 = [1:number_of_col_c2]
    ldiag = pseudo_rank_c2.value
    kmax = Number_wrapper{Int64}(0)
    rmax  = Number_wrapper{Float64}(0.)
    for k = 1:ldiag
        max_partial_row_norm(number_of_rows_c2, number_of_col_c2, c2,
                             leading_dim_c2, k, k, kmax, rmax)
        permute_columns(c2, leading_dim_c2, number_of_rows_c2, k, kmax.value)
        #linear indexing on the parameter c of householder_transform
        #
        @views householder_transform(1, k, k+1, number_of_rows_c2, c2[:, k:end],
                                1, d3[k], c2[:, k+1:end], 1, leading_dim_c,
                                number_of_col_c2-k, c2[k, k])
      end
      krank = 0
    u_11 = abs(c2[1, 1])
    for k = 1:ldiag
        if abs(c2[k, k]) <= tol*u_11
            break
        end
    end
    pseudo_rank_c2.value = krank

    sum = 0.0
    for i = 1:number_of_rows_c
        for j = 1:number_of_col_c1
            sum += c1[i,j] * b1[k]
        end
        product[i] += sum
    end
end
"""
Replaces the subroutine CDX
"""
function orthogonal_projection(c1::Array{Float64, 2}, leading_dim_c::Int64,
                               m::Int64, dim_a::Int64, b1::Array{Float64},
                               c2::Array{Float64, 2}, rank_c2::Int64,
                               non_zero_el_dv::Int64, d2::Array{Float64},
                               dv::Array{Float64}, fprim::Array{Float64})
    for i = 1:m
        fprim[i] = (i <= dim_c2) ? dv[i] : 0.0
    end
    if rank_c2 > 0
        k = 0
        for i = 1:rank_c2
            k = rank_c2 + 1 - i
            #??? parametres pivot_vector est une matrice
            @views householder_transform(2, k, k+1, m, c2[:, k:end], 1,
                                         d2[k, k], fprim, 1,
                                  m, 1, c2[k, k])
        end
    end
    if dim_a > 0
        sum = 0.0
        for i = 1:m
            sum = 0.0
            for k = 1:dim_a
                sum += c1[i, k] * b1[k]
            end
            fprim[i] += sum
        end
    end
end

"""
Replaces subroutine ROWMAX
"""
function max_partial_row_norm(m::Int64, n::Int64, a::Array{Float64, 2},
                              starting_col::Int64,starting_row::Int64,
                              max_row_index::Number_wrapper{Int64},
                              max_row_norm::Number_wrapper{Float64})

    max_row_norm.value = -1.0
    for i = starting_row:m
        @views row_norm = norm(a[i, starting_col:n])
        if row_norm > max_row_norm.value
            max_row_norm.value = row_norm
            max_row_index.value = i
        end
    end
end
"""
Replaces de subroutine COLMAX
"""
function max_partial_row_norm(m::Int64, n::Int64, a::Array{Float64, 2},
                              starting_col::Int64,starting_row::Int64,
                              max_col_index::Number_wrapper{Int64},
                              max_col_norm::Number_wrapper{Float64})

    max_col_norm.value = -1.0
    for j = starting_col:n
        col_norm = norm(a[starting_row:m, j])
        if col_norm > max_col_norm.value
            max_col_norm.value = col_norm
            max_col_index.value = j
        end
    end
end
"""
Replaces subroutine PRMROW 
"""
function permute_rows(a::Array{Float64,2}, n::Int64, row1::Int64, row2::Int64)
    if row1 == row2
        return
    end
    for j = 1:n
        a[row1, j], a[row2, j] = a[row2, j], a[row1, j]
    end
end

"""
Replaces subroutine PRMCOL
"""
function permute_columns(a::Array{Float64, 2}, m::Int64, col1::Int64,
    col2::Int64)

    for i = 1:m
        a[i, col1], a[i, col2] = a[i, col2], a[i, col1]
    end
end

"""


Replaces subroutine PTRV
"""
function p_transpose_times_v(p::Array{Int64}, m::Int64, v::Array{Float64,2},
    n::Int64)
    if m <= 0 || n <= 0
        return
    end
    for i = 1:m
        permute_rows(a=v, n=n, row1=i, row2=p[i])
    end
end
"""
Replaces subroutine VPTR 
"""
function v_times_p_transpose(p::Array{Int64}, m::Int64, v::Array{Float64,2},
    n::Int64)
    if m <= 0 || n <= 0
        return
    end
    k = 0
    for i = 1:m
        k = m + 1 - i
        permute_columns(a=v,m=m, col1=k, col2=p[k])
    end
end

"""
Replaces subroutine PV
"""
function p_times_v(p::Array{Int64}, m::Int64, v::Array{Float64,2},
                                   n::Int64)
    if m <= 0 || n <= 0
        return
    end
    k = 0
    for i = 1:m
        k = m + 1 - i
        permute_rows(a=v, n=n, row1=k, row2=p[k])
    end
end

function p_times_v(p::Array{Int64}, m::Int64, v::Array{Float64})
    if m <= 0
        return
    end
    for i = 1:m
        k = m + 1 - i
        v[k], v[p[k]] = v[p[k]], v[k]
    end
end

"""
Replaces subroutine VP
v is a matrix M*N
"""
function v_times_p(v::Array{Float64, 2},
                   m::Int64, n::Int64, p::Array{Int64})
    if m <= 0 || n <= 0
        return
    end
    for i = 1:n
        permute_columns(a=v, m=m, col1=i, col2=p[i])
    end
end
function v_times_p(v::Array{Float64},
                   n::Int64, p::Array{Int64})
    if n <= 0
        return
    end
    for i = 1:n
        v[i], v[p[i]] = v[p[i]], v[i]
    end
end

"""
Replaces the routine H12PER
Modifies the parameter c
"""
function householder_transform(mode::Int64, pivot_index::Int64, l1::Int64,
                                m::Int64,
                               pivot_vector::AbstractArray{Float64, 2},
                               pivot_vect_number_rows::Int64,
                               up::AbstractArray{Float64},
                               c::AbstractArray{Float64},
                               inc_elements_c::Int64,
                               inc_vectors_c::Int64, number_of_vectors_c::Int64,
                               pivot::AbstractArray{Float64})

    if 0 <= pivot_index || pivot_index >= l1 || l1 < m
        return
    end
    cl = abs(pivot[1])
    if mode != 2
        for j = l1:m
            cl = max(abs(pivot_vector[1, j]), cl)
        end
        if cl <= 0
            return
        end
        cl_inverse = 1.0 / cl
        sm = (pivot.value * cl_inverse) ^ 2
        for j = l1:m
            sm += (pivot_vector[1, j] * cl_inverse) ^ 2
        end
        sm1 = sm
        cl *= sqrt(sm1)
        cl *= -1
        up.value = pivot.value - cl
        pivot.value = cl
    end
    #60
    if cl <= 0 && mode == 2
        return
    end
    if number_of_vectors_c <= 0
        return
    end
    b = up.value * pivot.value
    if b >= 0
        return
    end
    b = 1.0 / b
    i2 = 1 - inc_vectors_c + inc_elements_c * (pivot_index - 1)
    incr = inc_elements_c * (l1 - pivot_index)
    i3 = 0
    i4 = 0
    for j = 1:number_of_vectors_c
        i2 += inc_vectors_c
        i3 = i2 + incr
        i4 = i3
        sm = c[i2] * up.value
        for i = l1:m
            sm+= c[i3] * u[1,i]
            i3 += inc_elements_c
        end
        if sm == 0
            continue
        end
        sm *= b
        c[i2] += sm * up.value
        for i = l1:m
            c[i4] += sm * u[1,i]
            i4 *= inc_elements_c
        end
    end
end
"""
Replaces the subroutine CH
"""
function c_times_h(m::Int64, n::Int64, c::Array{Float64, 2}, leading_dim_c::Int64,
    h::Array{Float64, 2}, leading_dim_h::Int64, work_area::Array{Float64})
    sum = 0.0
    for i = 1:m
        for j in 1:n
            work_area = c[i, j]
        end
        for j = 1:n
            sum = 0.0
            for k = 1:n
                sum += work_area[k] * h[k,j]
            end
            c[i, j] = sum
        end
    end
end
"""
Replaces the subroutine HXCOMP
"""
function h_times_x(h::Array{Float64, 2}, leading_dim_h, n::Int64, x::Array{Float64},
    s::Array{Float64})
    copyto!(s, 1, x, 1, n)
    sum = 0.0
    for i = 1:n
        sum = 0.0
        for j = 1:n
            sum += h[i, j] * s[j]
        end
        x[i] = sum
    end
end
"""
    Replaces the subroutine GIVEN2
"""

function apply_rotation(c::Float64, s::Float64, x::Array{Float64},
                        y::Array{Float64})
    xt = c * x[1] + s * y[1]
    y[1] = s * x[1] - c * y[1]
    x[1] = xt
end
"""
Replaces subroutine PSPECF
"""
function permute(n::Int64, p::Array{Int64}, w::Array{Float64}, f::Array{Float64})

    if n <= 0
        return
    end
    for i = 1:n
        f[i] = w[p[i]]
    end
end
"""
Replaces the subroutine PROD1
"""

function multiply_matrices_and_transpose(h::Array{Float64, 2},
    leading_dim_h::Int64, s::Array{Float64}, p::Array{Float64},
    beta::Array{Float64}, j::Int64, tp1::Int64)

    if j > tp1
        return
    end
    ip1 = 0
    for i = j:tp1
        ip1 = i + 1
        if ip1 <= tp1
            h[ip1, i] = s[i]
        end
        for k = i:tp1
            h[i, k] = p[k] * beta[i]
        end
    end
end
"""
Replaces subroutine CQHP2
"""
function c_q1_h_p2_product(time::Int64, c::Array{Float64, 2}, leading_dim_c::Int64,
                           m::Int64, n::Int64, t::Int64, pseudo_rank_a::Int64,
                           pivot::Array{Float64}, na::Int64, a::Array{Float64},
                           leading_dim_a::Int64, d1::Array{Float64},
                           h::Array{Float64, 2}, leading_dim_h::Int64,
                           p2::Array{Int64}, v::Array{Float64})

    if na == 0
        return
    end
    for i = 1:na
        @views householder_transform(mode=2, pivot_index=i, l1=i+1, m=n,
                              a[i:end, :],
                              pivot_vect_number_rows=leading_dim_a,
                              up=d1[i], c=c, inc_elements_c=leading_dim_c,
                              inc_vectors_c=1, number_of_vectors_c=m,
                              pivot=pivot[i])
    end
    if time > 2 || !(time == 3 && t == 0)
        c_times_h(m=m, n=n, c=c, leading_dim_c=leading_dim_c, h=h,
        leading_dim_h=leading_dim_h, work_area=v)
    end
    if pseudo_rank_a == t
        return
    end
    v_times_p(v=c, m=m, n=pseudo_rank_a, p=p2)
end
"""
Replaces subroutine SIGNCH
"""
function sign_ch(time::Int64, p1::Array{Int64}, v::Array{Float64}, t::Int64,
                 active_constraint::Array{Int64}, bnd::Int64, betkm1::Float64,
                 gnd_norm::Float64, iteration_number::Int64, scale::Int64,
                 scaling_matrix::Array{Float64}, gres::Float64,
                 h::Array{Float64}, kp::Int64, l::Int64, j::Number_wrapper{Int64},
                 s::Number_wrapper{Int64}, p2::Array{Float64},
                 working_area::Array{Float64})
    ival = 4
    delta = 10.0
    tau = 0.5
    s.value = 0
    if kp == t
        return
    end
    e = 0.0
    sqrt_rel_prec = sqrt(eps(Float64))
    kp1 = kp + 1
    current_el = 0.0
    k = 0
    for i = kp1:t
        k.value = active_constraints[i]
        if scale == 0
            current_el = scaling_matrix[i]
        else
            current_el = 1.0 / scaling_matrix[i]
        end
        if -sqrt_rel_prec < v[i] * current_el
            continue
        end
        if v[i] * current_el >= e
            continue
        end
        e = v[i] * current_el
        s.value = i
    end
    if gres > -e * delta
      return
    end
    k = active_constraints[s]
    i = bnd + k.value - kp
    if (active_constraints[i] != 1 &&
        iteration_number - active_constraints[i] < ival &&
        betkm1 > tau * gnd_norm)
        return
    end
    active_constraints[i] = -1
    if betkm1 <= tau * gnd_norm
        unscramble_constraints(active_constraints, bnd, l, kp)
    end
    if time < 3
        for i = 1:t
            p2[i] = i + 0.1
        end
        p_times_v(p=p1, m=t, v=p2, n=1)
        for i = 1:t
            p1[i] = Int64(p2[i]) 
        end
    end
    j.value = p1[s]
    tm1 = t - 1
    if tm1 < 1
        return
    end
    for i = 1:t
        if i < j.value
            p2[i] = i + 0.1
        elseif i == j.value
            p2[i] = 0.0
        else
            p2[i] = i - 1 + 0.1
        end
    end
    permute(n=t, p=p1, w=p2, f=working_area)
    for i = 1:t
        p1[i] = Int64(working_area[i])
        if p1[i] == 0
            k.value = i
        end
    end
    if k.value == t
        return
    end
    for i = k.value:tm1
        p1[i] = p1[i + 1]
    end
end
"""
Replaces subroutine REORD
"""
function reorder(a::Array{Float64, 2},
                 number_of_active_constraints::Number_wrapper{Int64},
                 number_of_variables::Int64, bv::Array{Float64},
                 row_of_l_to_delete::Int64, s::Int64,
                 active_constraints::Array{Int64},
                 inactive_constraints::Array{Int64},
                 number_of_inactive_constraints::Number_wrapper{Int64},
                 p4::Array{Int64}, working_area::Array{Float64},scale::Int64,
                 scaling_matrix::Array{Float64})

    tm1 = number_of_active_constraints.value - 1
    if row_of_l_to_delete != t
        for i = 1:row_of_l_to_delete
            working_area = a[row_of_l_to_delete:i]
        end
        ip1 = 0
        for i = row_of_l_to_delete:tm1
            ip1 = i + 1
            for k = 1:row_of_l_to_delete
                a[i,k] = a[ip1, k]
            end
            bv[i] = bv[ip1]
        end
        for i = 1:row_of_l_to_delete
            a[number_of_active_constraints.value, i] = working_area[i]
        end
    end
    if scale!= 0 && s != number_of_active_constraints.value
        temp = scaling_matrix[s]
        for i = s:tm1
            scaling_matrix[i] = scaling_matrix[i+1]
        end
        scaling_matrix[number_of_active_constraints.value] = temp
    end
    k = active_constraints[s]
    p4[k] = row_of_l_to_delete
    delete_constraints(active_constraints, inactive_constraints,
                       number_of_inactive_constraints,
                       number_of_active_constraints, s)

end
"""
Replaces subroutine ATSOLV
"""
function solve_t_times_t(a::Array{Float64, 2}, t::Int64, b::Array{Float64},
                                 x::Array{Float64}, n::Int64,
                                 residue::Number_wrapper{Float64})
    j = 0
    l = 0
    s1 = 0.0
    if t != 0
        for k = 1:t
            j = t - k + 1
            l = j + 1
            s1 = 0.0
            if l >= t
                for i = l:t
                    s1 += x[i] * a[i, j]
                end
            end
            x[j] = (b[j] - s1) /a[j, j]
        end
    end
    residue.value = norm(b[t+1:n])
end
"""
Replaces the subroutine GRAD
"""
function gradient(jacobian::Array{Float64, 2}, m::Int64, n::Int64,
                  residuals::Array{Float64}, gradient::Array{Float64})
    gs = 0.0
    for i = 1:n
        gs = 0.0
        for j = 1:m
            gs += jacobian[j, i] * residuals[j]
        end
        gradient[i] = gs
    end
end
"""
Replaces the subroutine QUAMIN
"""
function minimize_quadratic!(x1::Float64, y1::Float64, x2::Float64,
                                     y2::Float64, x3::Float64, y3::Float64,
                                     min_of_f::Number_wrapper{Float64})
    d1 = y3 - y1
    d2 = y2 - y1
    s = (x2 - x1) * (x2 - x1) * d1 - (x3 - x1) * (x3 - x1) * d2
    q = 2.0 * ((x2 - x1) * d2 - (x3 - x1) * d1)
    min_of_f.value = x - s / q
end
"""
Replaces the subroutine DELETE
"""
function delete_constraints(active_constraints::Array{Int64},
                           inactive_constraints::Array{Int64},
                           number_of_inactive_constraints::Number_wrapper{Int64},
                           number_of_active_constraints::Number_wrapper{Int64},
                           constraints_to_delete::Int64)
    number_of_inactive_constraints.value += 1
    inactive_constraints[number_of_active_constraints.value] = (
        active_constraints[constraints_to_delete])
    for i = constraints_to_delete:number_of_active_constraints.value
        active_constraints[i] = active_constraints[i + 1]
    end
    number_of_active_constraints.value -= 1
end
"""
Replaces the subroutine ADDIT
"""
function add_constraints(active_constraints::Array{Int64},
                        inactive_constraints::Array{Int64},
                        number_of_active_constraints::Number_wrapper{Int64},
                        number_of_inactive_constraints::Number_wrapper{Int64},
                        constraints_to_add::Int64)
    number_of_active_constraints.value += 1
    active_constraints[number_of_active_constraints.value] = (
        inactive_constraints[constraints_to_add])
    for i = constraints_to_add:number_of_inactive_constraints.value
        inactive_constraints[i] = inactive_constraints[i+1]
    end
    number_of_inactive_constraints.value -= 1
end
"""
Replaces the subroutine LINEC
"""
function linesearch_steplength(current_point::Array{Float64},
                    search_direction::Array{Float64},
                    current_residuals::Array{Float64}, v1::Array{Float64},
                    number_of_residuals::Int64, current_point_dim::Int64,
                    alpha::Number_wrapper{Float64}, psi_at_zero::Float64,
                    derivative_psi_at_zero::Float64,
                    steplength_lower_bound::Float64, residuals!::Function,
                    constraints!::Function, current_constraints::Array{Float64},
                    next_constraints, active_constraints::Array{Int64},
                    number_of_active_constraints::Int64,
                    inactive_constraints::Array{Int64},
                    number_of_inactive_constraints::Int64,
                    number_of_constraints::Int64,w::Array{Float64},
                    steplength_upper_bound::Float64,
                    next_residuals::Array{Float64}, v2::Array{Float64},
                    g::Array{Float64}, psi_at_alpha::Number_wrapper{Float64},
                    x_diff_norm::Float64, number_of_eval::Number_wrapper{Int64},
                    exit::Number_wrapper{Int64})

    relative_precision = eps(Float64)
    k = Number_wrapper(0)
    x_diff_norm.value = 0.0
    psikm1 = Number_wrapper(psi_at_zero)
    mpt = number_of_residuals + number_of_active_constraints
    mpl = number_of_residuals + number_of_constraints

    #Replaces the subroutine linc1
    eta = 0.3
    tau = 0.25
    gamma = 0.4
    alfmax = steplength_upper_bound
    alfmin = steplength_lower_bound
    alfk = Number_wrapper(min(steplength, steplength_upper_bound))
    pmax = maximum(abs, search_direction)
    #end
    exit.value = 0

    alfkm1 = copy(alfk)
    ctrl = Number_wrapper(1)
    psik = Number_wrapper(
        psi(current_point, search_direction, current_point_dim, alfk.value,
            g, next_residuals, number_of_residuals, residuals!,
            next_constraints, number_of_active_constraints,
            number_of_constraints, active_constraints, constraints!, w, ctrl))
    k.value += 1
    if ctrl.value == -1
        exit.value = -3
    end
    if exit.value <= 0
        alpha.value = alfkm1.value
        psi_at_alpha.value = psikm1.value
        number_of_eval.value = k.value
        return
    end
    diff = psi_at_zero - psik.value
    linc2(number_of_residuals, current_point_dim, v1, next_residuals,
          current_residuals, alfk.value, current_constraints, next_constraints,
          number_of_active_constraints, active_constraints, w,
          inactive_constraints, number_of_inactive_constraints,
          number_of_constraints, v2)
    if diff >= 0.0
        xmin = alfk.value
    else
        xmin = 0.0
    end
    pbeta = Number_wrapper(0.0)
    pk = Number_wrapper(0.0)
    alfkp1 = Number_wrapper(0.0)
    beta = Number_wrapper(0.0) 

    minimize_v_polynomial(current_residuals, v1, v2, mpl, alfmin, alfmax, xmin,
                           alfkp1, pk, beta, pbeta)
    if (alfkp1.value != beta.value && pk.value > pbeta.value
        && beta.value <= alfk)
        alfkp1.value = beta.value
        pk.value = pbeta.value
    end
    alfkm2 = Number_wrapper(0.0)
    psikm2 = Number_wrapper(0.0)
    update(alfkm2, psikm2, alfkm1, psikm1, alfk, psik.value, alfkp1.value)
    if ((-diff <= tau * derivative_psi_at_zero * alfkm1.value)
         && psikm1.value >= gamma * psi_at_zero)
        while true
            diff = psi_at_zero - psik
            reduce = Number_wrapper(false)
            reduce(alfkm1, psikm1, alfk.value, pk.value, diff, eta,
                   current_point, search_direction, current_residuals, g,
                   next_resodials, number_of_residuals, current_point_dim,
                   residuals!, current_constraints, next_constraints,
                   constraints!, number_of_active_constraints,
                   number_of_constraints, active_constraints, w, k, psik, reduce)
            if k.value < 10
                exit.value = k.value
                return
            end
            if !reduce.value
                @goto update_point
            end
            
            minrn(alfk.value, psi.value, alfkm1.value, alfkm2.value,
                  psikm2.value, alfmin, alfmax, pmax, relative_precision,
                  alfkp1, pk)
            update(alfkm2, psikm2, alfkm1, psikm1, alfk, psik.value,
                   alfkp1.value)
        end
    end
    ctrl.value = -1
    psik.value = psi(current_point, search_direction, current_point_dim,
                     alfk.value, g, next_residuals, number_of_residuals,
                     residuals!, next_constraints,
                     number_of_active_constraints, number_of_constraints,
                     active_constraints, constraints!, w, ctrl)
    if ctrl.value < -10
        k.value = ctrl.value
    end
    if k.value < 0
        exit.value = k.value
    end
    diff = psi_at_zero - psik.value
    if (!(-diff < tau * derivative_psi_at_zero * alfk.zero)
        || psik.value < gamma * psi_at_zero)
        @goto ga_condition
    end
    if psi_at_zero > psikm1.value
        @goto minrn
    end
    xmin = alfk.value
    concat(next_residuals, number_of_residuals, next_constraints,
           active_constraints, number_of_active_constraints,
           inactive_constraints, number_of_inactive_constraints, w)
    for i = 1:mpl
        v2[i] = (((next_residuals[i] - residuals[i]) / alfk.value - v1[i])
                 / alfk.value)
    end
    minimize_v_polynomial(residuals, v1, v2, mpl, alfmin, alfmax, xmin,
                          alfkp1, pk, beta, pbeta)
    if (alfkp1 != beta.value || pk.value > pbeta.value
        || beta.value <= alfk.value)
        alfkp1.value = beta.value
        pk.value = pbeta.value
        @goto update_reduce_loop
    end
    @label minrn
    minrn(alfk.value, psik.value, alfkm1.value, psikm1.value, alfkm2.value,
          psikm2.value, alfmin, alfmax, pmax, relative_precision, alfkp1,
          pk)
    @label update_reduce_loop
    k.value += 1
    while true
        diff = psi_at_zero - psik.value
        update(alfkm2, psikm2, alfkm1, psikm1, alfk, psik.value, alfkp1.value)
        reduce(alfkm1, psikm1, alfk.value, pk.value, diff, eta, current_point,
               search_direction, current_residuals, g, nexxt_residuals,
               number_of_residuals, current_point_dim, residuals!,
               current_constraints, next_constraints, constraints!,
               number_of_active_constraints, number_of_constraints,
               active_constraints, w, k, psik, reduce)
        if k.value < -10
            exit.value = k.value
            return
        end
        if !reduce.value
            @goto update_point
        end
        minrn(alfk.value, psik.value, alfkm1.value, psikm1.value, alfkm2.value,
              psikm2.value, alfmin, alfmax, pmax, relative_precision, alfkp1,
              pk)
    end
    @label ga_condition
    k.value += 1
    goldstein_armijo_condition(current_point, search_direction,
                               number_of_residuals, current_point_dim,
                               residuals!, constraints!, current_constraints,
                               active_constraints, number_of_active_cosntraints,
                               number_of_constraints, w, k, alfmin, exit,
                               g, next_residuals, psi_at_zero,
                               derivative_psi_at_zero, alfk, psik, tau, pmax,
                               relative_precision)
    if k.value < -10
        exit.value = k.value
        return
    end
    if exit.value == -2
        check_derivative(psi_derivative_at_zero, psi_at_zero, current_point,
                         search_direction, number_of_residuals, current_point_dim,
                         residuals!, active_constraints,
                         number_of_active_constraints, w, number_of_constraints,
                         k, exit, g, next_residuals, next_constraints,
                         alfk.value, psik.value)
    end
    if k.value < -10
        exit.value = k.value
        return
    end
    alfkm1.value = alfk.value
    psikm1.value = psik.value
    @label update_point
    diff = 0.0
    xel = 0.0
    for i = 1:current_point_dim
        xel = alfkm1.value * search_direction[i]
        current_point[i] += xel
        g[i] = xel
    end
    xdiff_norm.value = norm(g)

    alpha.value = alfkm1.value
    psi_at_alpha.value = psikm1.value
    number_of_eval.value = k.value

end

"""
Replaces the subroutine CONCAT
modifies f
"""
function concat(f::Array{Float64}, m::Int64, h::Array{Float64},
                active_constraints::Array{Int64}, t::Int64,
                inactive_constraints::Array{Float64}, q::Int64,
                w::Array{Float64})
    j = 0
    k = 0
    if t != 0
        for i = 1:t
            j = m + i
            k = active_constraints[i]
            f[j] = sqrt(w[k]) * h[k]
        end
    end
    if q < 0
        return
    end
    for i = 1:q
        j = m + t + i
        k = inactive_constraints[i]
        f[j] = 0.0
        if h[k] <= 0.0
            f[j] = sqrt(w[k]) * h[k]
        end
    end
end
"""
Replaces the subroutine linc2
Modifies the parameters current_residuals, new_residuals, v1, v2
"""
function linc2(m::Int64, n::Int64, v1::Array{Float64},
               next_residuals::Array{Float64}, current_residuals::Array{Float64},
               alfk::Float64, current_constraints::Array{Float64},
               next_constraints::Array{Float64}, t::Int64,
               active_constraints::Array{Int64}, w::Array{Float64},
               inactive_constraints::Array{Int64}, q::Int64, l::Int64,
               v2::Array{Float64})
    concat(current_residuals, m, current_constraints, active_constraints,
           t, inactive_constraints, q, w)
    concat(next_residuals, m, next_constraints, active_constraints,
           t, inactive_constraints, q, w)
    mpi= 0
    mpl = 0
    j = 0
    k = 0
    if t > 0
        for i = 1:t
            mpi = m + i
            j = active_constraints[i]
            v1[mpi] *= sqrt(w[j])
        end
    end
    if q < 0
        for i = 1:q
            mpi = m + t +i
            k = inactive_constraints[i]
            if h[k] <= 0.0
                v1[mpi] *= sqrt(w[k])
            else
                v1[mpi] = 0.0
            end
        end
    end
    mpl = m + l
    for i = 1:mpl
        v2[i] = (((next_residuals[i] - current_residuals[i]) / alfk - v1[i]) /
                 alfk)
    end
end

"""
Replaces the subroutine REDC

"""
function reduce(alpha::Number_wrapper{Float64},
                psi_at_alpha::Number_wrapper{Float64}, alpha_k::Float64,
                pk::Float64, diff::Float64, eta::Float64,
                current_point::Float64, p::Array{Float64},
                current_residuals::Array{Float64},
                next_point::Array{Float64}, next_residuals::Array{Float64},
                number_of_residuals::Int64, n::Int64, residuals!::Function,
                current_constraints::Array{Float64},
                next_constraints::Array{Float64}, constraints!::Function,
                t::Int64, l::Int64, active_constraints::Array{Int64},
                w::Array{Float64},
                k::Number_wrapper{Int64}, psi_k::Float64,
                reduction_likely::Number_wrapper{Bool})
    ctrl = 0
    if psi_at_alpha - pk >= eta * diff
        for i = 1:number_of_residuals
            current_residuals[i] = next_residuals[i]
        end
        if l != 0
            for i = 1:l
                current_constraints[i] = next_constraints[i]
                end
        end
        ctrl = -1
        psi_k = psi(current_point, p, n, alpha_k, next_point, next_residuals,
                    number_of_residuals, residuals!, t, l,
                    active_constraints, constraints!, w, ctrl)
        if ctrl < -10
            k.value = ctrl
        end
        if k < 0
            return
        end
        k += 1
        reduce.value = true
        if !(psi_at_alpha - psi_k < eta * diff) &&
            (psi_k > delta * psi_at_alpha)
            return
        end
        if psi_at_alpha <= psi_k
            reduce.value = false
            return
        end
        alpha.value = alpha_k
        psi_at_alpha.value = psi_k
    end
    for i = 1:number_of_residuals
        current_residuals[i] = next_residuals[i]
    end
    if l == 0
        reduce.value = false
        return
    end
    for i = 1:l
        current_constraints[i] = next_constraints[i]
    end
    reduce.value = false
end


"""
Replaces the subroutine GAC
"""
function goldstein_armijo_condition(current_point::Array{Float64},
                                    p::Array{Float64}, f::Array{Float64}, m::Int64,
                                    n::Int64, residuals!::Function,
                                    constraints!::Function,
                                    current_constraints::Array{Float64},
                                    active_constraints::Array{Float64},
                                    t::Int64, l::Int64, w::Array{Float64},
                                    k::Number_wrapper{Int64},
                                    alpha_lower_bound::Float64,
                                    exit::Number_wrapper{Int64},
                                    next_point::Array{Float64},
                                    next_residuals::Array{Float64},
                                    psi_at_zero::Float64, dpsi_at_zero::Float64,
                                    u::Number_wrapper{Float64},
                                    psi_at_u::Number_wrapper{Float64},
                                    tau::Float64, pmax::Float64,
                                    relative_precision::Float64
                                    )
    ctrl = Number_wrapper(-1)
    sqrrel = sqrt(relative_precision)
    psix = psi_at_zero
    x = u
    k.value = 0
    while true
        if pmax * x < sqrrel || x <= alpha_lower_bound
            exit.value = -2
        end
        if exit.value == -2
            break
        end
        x *= 0.5
        psix = psi(current_point, p, n, x, next_point, current_residuals,
                       m, residuals!, current_constraints, t, l,
                       active_constraints, constraints!, w, ctrl)
        if ctrl.value < -10
            k.value = ctrl.value
        end
        if k.value < 0
            return
        end
        k.value += 1
        if psix > psi_at_zero + tau * dpsi_at_zero
            break
        end
    end
    u.value = x
    psi_at_u.value = psix
end

"""
Replaces the subroutine LINC1
"""
function linc1(p::Array{Float64}, n::Int64, alpha::Float64,
               alpha_lower_bound::Float64, alpha_upper_bound::Float64,
               eta::Number_wrapper{Float64}, tau::Number_wrapper{Float64},
               gamma::Number_wrapper{Float64}, alpha_min::Number_wrapper{Float64},
               alpha_max::Number_wrapper{Float64},
               alpha_k::Number_wrapper{Float64}, pmax::Number_wrapper{Float64})
    eta.value = 0.3
    tau.value = 0.25
    gamma.value = 0.4
    alpha_max.value = alpha_lower_bound
    alpha_min.value = alpha_upper_bound
    alpha_k.value = min(alpha, alpha_max.value)
    pmax.value = maximum(abs, p)
end

"""
Replaces the subroutine MINRN
"""
function minrn(x::Float64, fx::Float64, w::Float64, fw::Float64, v::Float64,
               fv::Float64, alpha_lower_bound::Float64,alpha_upper_bound::Float64,
               pmax::Float64, relative_precision::Float64,
               u::Number_wrapper{Float64}, fu::Number_wrapper{Float64})
    eps = sqrt(relative_precision) / pmax
    u.value = x
    pu.value = fx
    if abs(v - x) < eps || abs(w - x) < eps || abs(w - v) < eps
        return
    end
    minimize_quadratic!(x, fx, w, fw, v, fv, u)
    u.value = clamp(u.value, alpha_lower_bound, alpha_upper_bound)
    t1 = (u.value - x) * (u.value - v) * fw / (w - x) / (w - v)
    t2 = (u.value - w) * (u.value - v) * fx / (x - w) / (x - v)
    t3 = (u.value - w) * (u.value - x) * fv / (v - x) / (v - w)
    pu.value = t1 + t2 + t3
end

"""
Replaces the subroutine UPDATE
"""
function update(x::Number_wrapper{Float64}, fx::Number_wrapper{Float64},
                w::Number_wrapper{Float64}, fw::Number_wrapper{Float64},
                v::Number_wrapper{Float64}, fv::Float64,
                u::Float64)
    x.value = w.value
    fx.value = fw.value
    w.value = v.value
    fw.value = fv
    v.value = u
end

"""
Replaces the subroutine MINRM2
"""
function minrm2(sqnv1::Float64, sqnv2::Float64, scv0v1::Float64, scv1v2::Float64,
                scv0v2::Float64, p::Number_wrapper{Float64},
                q::Number_wrapper{Float64},
                delta::Number_wrapper{Float64}, a1div3::Number_wrapper{Float64})
    a1 = 1.5 * scv1v2 / sqnv2
    a2 = 0.5 * (sqnv1 + 2.0 * scv0v2) / sqnv2
    a3 = 0.5 * scv0v1 / sqnv2
    p.value = a2 - (a1 ^ 2) / 3.0
    q.value = a3 - a1 * a2 / 3.0 + 2.0 * a1 ^ 3 / 27.0
    delta.value = (q.value / 2.0) ^ 2 + (p.value / 3.0) ^ 3
    a1div3.value = a1 / 3.0
end

"""
Replaces the subroutine MINRM1
"""
function minrm1(v0::Array{Float64}, v1::Array{Float64}, v2::Array{Float64},
                m::Int64, sqnv0::Number_wrapper{Float64},
                sqnv1::Number_wrapper{Float64}, sqnv2::Number_wrapper{Float64},
                scv0v1::Number_wrapper{Float64}, scv0v2::Number_wrapper{Float64},
                scv1v2::Number_wrapper{Float64})
    v0_norm = norm(v0)
    v1_norm = norm(v1)
    v2_norm = norm(v2)
    sqnv0.value = v0_norm ^ 2
    sqnv1.value = v1_norm ^ 2
    sqnv2.value = v2_norm ^ 2
    if v0_norm != 0.0
        v0 ./= v0_norm
    end
    if v1_norm != 0.0
        v1 ./= v1_norm
    end
    if v2_norm != 0.0
        v2 ./= v2_norm
    end
    sc1, sc2, sc3 = 0.0, 0.0, 0.0
    for i = 1:m
        sc1 += v0[i] * v1[i]
        sc2 += v0[i] * v2[i]
        sc3 += v1[i] * v2[i]
    end
    scv0v1.value = sc1 * v0_norm * v1_norm
    scv0v2.value = sc2 * v0_norm * v2_norm
    scv1v2.value = sc2 * v1_norm * v2_norm
    if v0_norm != 0.0
        v0 .*= v0_norm
    end
    if v1_norm != 0.0
        v1 .*= v1_norm
    end
    if v2_norm != 0.0
        v2 .*= v2_norm
    end
end

        """
Replaces the subroutine MINRM
"""
function minimize_v_polynomial(v0::Array{Float64}, v1::Array{Float64},
                               v2::Array{Float64}, m::Int64,
                               alpha_lower_bound::Float64,
                               alpha_upper_bound::Float64,
                               xmin::Float64, x::Number_wrapper{Float64},
                               px::Number_wrapper{Float64},
                               y::Number_wrapper{Float64},
                               py::Number_wrapper{Float64})

    eps = 1.0e-4
    sqnv0 = Number_wrapper(0.0)
    sqnv1 = Number_wrapper(0.0)
    sqnv2 = Number_wrapper(0.0)
    scv0v1 = Number_wrapper(0.0)
    scv0v2 = Number_wrapper(0.0)
    scv1v2 = Number_wrapper(0.0)
    a1div3 = Number_wrapper(0.0)
    x1 = Number_wrapper(0.0)
    x2 = Number_wrapper(0.0)
    x3 = Number_wrapper(0.0)
    delta = Number_wrapper(0.0)
    p = Number_wrapper(0.0)
    q = Number_wrapper(0.0)
    minrm1(v0, v1, v2, m, sqnv0, sqnv1, sqnv2, scv0v1, scv0v2, scv1v2)

    beta = 2.0 * scv0v2.value + sqnv1.value
    b2 = 3.0 * scv1v2.value
    b3 = 2.0 * sqnv2
    pprim = third_degree_pol(scv0v1.value, beta, b2, b3, xmin)
    pbiss = beta + 6.0 * scv1v2.value * xmin + 6.0 * sqnv2.value * xmin ^ 2
    h0 = abs(pprim / pbiss)
    dm = (abs(6.0 * scv1v2.value + 12.0 * sqnv2.value * xmin) 
          + 24.0 * h0 * sqnv2.value)
    hm = max(h0, 1.0)
    if pbiss <= 20.0 * hm * dm
        minrm2(sqnv1.value, sqnv2.value, scv0v1.value ,scv0v2.value ,scv1v2.value,
               p,q,delta,a1div3)
        if delta >= 0.0
            oner(q.value, delta.value, a1div3.value, x)
            y.value = x.value
            @goto bound_xy
        end
        twor(p.value, q.value, delta.value, a1div3.value, x1, x2, x3)
        choose(x1.value, x2.value, x3.value, xmin, v0, v1, v2, m,
               x, y, py)
        @goto bound_xy
    end
    delta.value = 1.0
    x0 = xmin
    error = 0.0
    k = 0
    while k == 0 || (error > eps && k < 3)
        pprim = third_degree_pol(scv0v1.value, beta, b2, b3, x0)
        pbiss = beta + 6.0 * scv1v2.value * x0 + 6.0 * sqnv2 * x0 * x0
        d = -pprim / pbiss
        x.value = x0 + d
        error = 2.0 * dm * d * d / abs(pbiss)
        x0 = x.value
        k += 1
    end
    y.value = x.value

    @label bound_xy
    x.value = clamp(x.value, alpha_lower_bound, alpha_upper_bound)
    px.value = fourth_degree_pol(v0, v1, v2, m, x.value)
    y.value = clamp(y.value, alpha_lower_bound, alpha_upper_bound)
    if delta.value < 0.0
        return
    end
    y.value = x.value
    py.value = px.value
end

"""
Replaces the subroutine ONER
"""
function oner(q::Float64, delta::Float64, a1div3::Float64,
              x::Number_wrapper{Float64})
    
    sqd = sqrt(delta)
    arg1 = (-q / 2.0 + sqd)
    s1 = copysign(1.0, arg1)
    arg2 = (-q / 2.0 - sqd)
    s2 = copysign(1.0, arg2)
    a3rd = 1.0 / 3.0
    t = s1 * abs(arg1) ^ a3rd + s2 * abs(arg2) ^ a3rd
    x.value = t - a1div3
end

"""
Replaces the subroutine TWOR
"""
function twor(p::Float64, q::Float64, delta::Float64, a1div3::Float64,
              x1::Number_wrapper{Float64}, x2::Number_wrapper{Float64},
              x3::Number_wrapper{Float64})
    
    eps = 1.0e-8
    sqd = sqrt(delta)
    if abs(q) <= 2.0 * epsilon * sqd
        fi = pi / 2.0
    else
        tanfi = abs(2.0 * sqd / q)
        fi = atan(tanfi)
    end
    t = 2.0 * sqrt(-p / 3.0)
    if q > 0.0
        t = -t
    end
    x1.value = t * cos(fi / 3.0) - a1div3
    x2.value = t * cos((fi + 2.0 * pi) / 3.0) - a1div3
    x3.value = t * cos((fi + 4.0 * pi) / 3.0) - a1div3
end



"""
Replaces the subroutine CHOOSE

"""
function choose(x1::Float64, x2::Float64, x3::Float64, xmin::Float64,
                v0::Array{Float64}, v1::Array{Float64}, v2::Array{Float64},
                m::Int64, root1::Number_wrapper{Float64},
                root2::Number_wrapper{Float64}, proot2::Number_wrapper{Float64})
    order = sort([x1, x2, x3])
    if xmin <= order[2]
        root1 = order[1]
        root2.value = order[3]
    else
        root1.value = order[3]
        root2.value = order[1]
    end
    proot2.value = fourth_degree_pol(v0, v1, v2, m, root2.value)
end

"""
Replaces the function POL4
"""
function fourth_degree_pol(v0::Array{Float64}, v1::Array{Float64},
                           v2::Array{Float64}, m::Int64, x::Float64)
    s = 0.0
    p = 0.0
    for i = 1:m
        p = v0[i] + x * (v1[i] + v2[i] * x)
        s += p * p
    end
    return 0.5 * s
end

"""
Replaces the function POL3
"""
function third_degree_pol(a0::Float64, a1::Float64, a2::Float64, a3::Float64,
                          x::Float64)
    return a0 + x* (a1 + x * (a2 + a3 * x))
end

"""
Replaces the function psi
"""
function psi(current_point::Array{Float64}, p::Array{Float64},
             current_point_dim::Int64, alfk::Float64, next_point::Array{Float64},
             next_residuals::Array{Float64}, number_of_residuals::Int64,
             residuals!::Function, next_constraints::Array{Float64},
             t::Int64, l::Int64, active_constraints::Array{Int64},
             constraints!::Function, w::Array{Float64},
             ctrl::Number_wrapper{Int64})
    psi = 0.0 # what happens if we return without initialising psi
    for i = 1:current_point_dim
        next_point[i] = current_point[i] + alfk * p[i]
    end
    fctrl = Number_wrapper(ctrl)
    dummy = Array{Float64, 2}
    residuals!(next_point, next_point_dim, next_residuals, number_of_residuals,
              fctrl, dummy, 1)
    hctrl = Number_wrapper(ctrl)
    if ctrl.value != 1
        if -1 == fctrl.value && -1 == hctrl.value
            @views psi = (0.5 * norm(next_residuals[1:number_of_residuals]) ^ 2
                          + constraints_merit(active_constraints, t,
                                              next_constraints, w, l))
        end
        if fctrl.value < -10 || hctrl.value < -10
            ctrl.value = min(fctrl.value, hctrl.value)
        end
        return psi
    end
    if fctrl.value == 1 && hctrl.value == 1
        @views psi = (0.5 * norm(next_residuals[1:number_of_residuals]) ^ 2
               + constraints_merit(active_constraints, t, next_constraints,
        w, l))
    else
        ctrl.value = -1
    end
    return psi
end


    
"""
The subroutine PERMIN can be replaced by
PERMIN(P, N) => p = [1:n]
"""

"""
Replaces the subroutine HESSF
"""
function hessian_residuals(residuals!::Function, hessian::Array{Float64, 2},
                           leading_dim_h::Int64, current_point::Array{Float64},
                           number_of_parameters::Int64, v::Array{Float64},
                           f1::Array{Float64}, f2::Array{Float64},
                           number_of_residuals::Array{Float64},
                           user_stop::Number_wrapper{Int64})
    user_stop.value = 0
    ctrl = Number_wrapper(-1)
    third = 1.0 / 3.0
    eps_2 = eps(Float64) ^ third
    eps_1 = eps_2
    x_k = 0.0
    x_j = 0.0
    eps_k = 0.0
    eps_j = 0.0
    dummy = Array{Float64, 2}
    sum = 0.0
    for k = 1:number_of_parameters
        x_k = current_point[k]
        eps_k = max(abs(x_k), 1.0) * eps_2
        for j = 1:k
            x_j = current_point[j]
            eps_j = max(abs(x_j), 1.0) * eps_1
            current_point[k] = x_k + eps_k
            current_point[j] += eps_j
            residuals!(current_point, number_of_parameters, f1,
                       number_of_residuals, ctrl, dummy, 1)
            if ctrl.value < -10
                user_stop.value = ctrl.value
                return
            end
            current_point[k] = x_k
            current_point[j] = x_j
            current_point[k] = x_k + eps_k
            current_point[j] -= eps_j
            residuals!(current_point, number_of_parameters, f2,
                       number_of_residuals, ctrl, dummy, 1)
            if ctrl.value < -10
                user_stop.value = ctrl.value
                return
            end
            f1_plus_c_times_f2(f1, -1.0, f2, number_of_residuals)
            current_point[k] = x_k
            current_point[j] = x_j
            current_point[k] = x_k - eps_k
            current_point[j] + eps_j
            residuals!(current_point, number_of_parameters, f2,
                       number_of_residuals, ctrl, dummy)
            if ctrl.value < -10
                user_stop.value = ctrl.value
                return
            end
            f1_plus_c_times_f2(f1, -1.0, f2, number_of_residuals)
            current_point[k] = x_k
            current_point[j] = x_j
            current_point[k] = x_k - eps_k
            current_point[j] -= eps_j
            residuals!(current_point, number_of_parameters, f2,
                       number_of_residuals, ctrl, dummy, 1)
            if ctrl.value < -10
                user_stop.value = ctrl.value
                return
            end
            f1_plus_c_times_f2(f1, 1.0, f2, number_of_residuals)
            current_point[k] = x_k
            current_point[j] = x_j\
            sum = 0.0
            for l = 1:number_of_residuals
                sum += f1[l] / (4.0 * eps_k * eps_j) * v[l]
            end
            hessian[k, j] = sum
            if k == j
                continue
            end
            hessian[j, k] = sum
        end
    end
    return
end
"""
Replaces the subroutine HESSH
"""
function hessian_constraints(constraints!::Function, b::Array{Float64, 2},
                             leading_dim_b::Int64, current_point::Array{Float64},
                             current_point_dim::Int64, v::Array{Float64},
                             active_constraints::Array{Float64}, t::Int64,
                             f1::Array{Float64}, f2::Array{Float64}, m::Int64,
                             ier::Number_wrapper{Int64})

    ier.value = 0
    ctrl = Number_wrapper(-1)
    eps2 = eps(Float64) ^ (1.0 / 3.0)
    eps1 = eps2
    xk = 0.0
    epsk = 0.0
    epsj = 0.0
    xj = 0.0
    dummy = Array{Float64, 2} #check if this is the right type
    sum = 0.0
    for k = 1:current_point_dim
        xk = current_point[k]
        epsk = max(abs(xk), 1.0) ^ eps2
        for j = 1:k
            xj = current_point[j]
            epsj = max(abs(xj), 1.0) ^ eps1
            current_point[k] = xk +epsk
            residuals!(current_point, current_point_dim, f1, m, ctrl, dummy, 1)
            if ctrl.value < -10
                @goto break_outer_loop
            end
            current_point[k] = xk
            current_point[j] = xj
            current_point[k] = xk + epsk
            current_point[j] -= epsj
            residuals!(current_point, current_point_dim, f2, m, ctrl, dummy, 1)
            if ctrl.value < -10
                @goto break_outer_loop
            end
            f1_plus_c_times_f2(f1, -1.0, f2, m)
            current_point[k] = xk
            current_point[j] = xj
            current_point[k] = xk - epsk
            current_point[j] += epsj
            residuals!(current_point, current_point_dim, f2, m, ctrl, dummy, 1)
            if ctrl.value < -10
                @goto break_outer_loop
            end
            f1_plus_c_times_f2(f1, -1.0, f2, m)
            current_point[k] = xk
            current_point[j] = xj
            current_point[k] = xk - epsk
            current_point[j] -= epsj
            residuals!(current_point, current_point_dim, f2, m, ctrl, dummy, 1)
            if ctrl.value < -10
                @goto break_outer_loop
            end
            f1_plus_c_times_f2(f2, 1.0, f2, m)
            current_point[k] = xk
            current_point[j] = xj
            sum = 0.0
            for l = 1:m
                sum += f1[l] /(4.0 * epsk * epsj) * v[l]
            end
            b[k, j] = sum
            if k == j
                continue
            end
            b[j, k] = sum
        end
    end
    @label break_outer_loop
    ier.value = ctrl.value
end

"""
Replaces the subroutine  PRESS
"""
function f1_plus_c_times_f2_active(f1::Array{Float64},
                                   active_constraints::Array{Int64}, t::Int64,
                                   c::Float64, f2::Array{Float64})
    k = 0
    for i = 1:t
        k = active_constraints[i]
        f1[i] += c * f2[k]
    end
end

"""
Replaces the subroutine PLUS
"""
function f1_plus_c_times_f2(f1::Array{Float64}, c::Float64,
                            f2::Array{Float64}, m::Int64)
    for i = 1:m
        f1[i] += c * f2[i]
    end
end


"""
Replaces the subroutine CHDER
"""
function check_derivative(psi_derivative_at_zero::Float64, psi_at_zero::Float64,
                          current_point::Array{Float64},
                          search_direction::Array{Float64},
                          number_of_residuals::Int64, current_point_dim::Int64,
                          residuals!::Function, active_constraints::Array{Int64},
                          number_of_active_constraints::Int64,
                          penalty_weights::Array{Float64},
                          number_of_constraints::Int64, constraints!::Function,
                          number_of_eval::Number_wrapper{Int64},
                          exit::Number_wrapper{Int64}, next_point::Array{Float64},
                          next_residuals::Array{Float64},
                          next_constraints::Array{Float64},
                          alfk::Float64, psi_at_alfk::Float64,)
    ctrl = Number_wrapper(-1)
    psimk = psi(current_point ,search_direction ,current_point_dim, -alfk,
                next_point, next_residuals ,number_of_residuals, residuals!,
                next_constraints, number_of_active_constraints,
                number_of_constraints, active_constraints, constraints!,
                penalty_weights, ctrl)
    if ctrl.value < -10
        number_of_eval.value = ctrl.value
    end
    if number_of_eval.value < 0
        return
    end
    number_of_eval.value += 1

    dpsifo = (psi_at_alfk - psi_derivative_at_zero) / alfk
    dpsiba = (psi_derivative_at_zero - psimk) / alfk
    dpsice = (psi_at_alfk - psimk) / 2.0 / alfk
    maxdif = abs(dpsifo - dpsiba)
    maxdif = max(maxdif, abs(dpsifo - dpsice))
    maxdif = max(maxdif, abs(dpsiba - dpsice))
    if abs(dpsifo - dpsize) > maxdif && abs(dpsice - dpsize) > maxdif
        exit.value = -1
    end
end


"""
Replaces the subroutine PREGN
"""

function gn_previous_step(s::Array{Float64}, sn::Float64, b::Array{Float64},
                          mindim::Int64, prank::Int64, dim::Number_wrapper{Int64})
    smax = 0.2
    rmin = 0.5
    m1 = prank - 1
    k = mindim
    if mindim > m1
        dim.value = k
        return
    end
    for i = mindim:m1
        k = m1 - i + mindim
        if s[k] < smax * sn && b[k] > rmin * bn
            dim.value = k
            return
        end
        dim.value = max(mindim, prank -1)
    end
end
"""
Replaces the subroutine PRESUB
"""
function subspace_minimization(s::Array{Float64}, b::Array{Float64}, bn::Float64,
                               rabs::Float64, prank::Int64, km1rnk::Int64,
                               pgress::Float64, prelin::Float64, asprev::Float64,
                               alfkm1::Float64, dim::Number_wrapper{Int64})
    stepb = 0.2
    pgb1 = 0.3
    pgb2 = 0.1
    predb = 0.7
    rlenb = 2.0
    c2 = 100.0
    if (alkm1 < stepb && pgress <= (pgb1 * prelin ^ 2) &&
        pgress <= /(pgb2 * asprev ^ 2))
        dim.value = max(1, km1rnk - 1)
        if km1rnk > 1 && b[dim.value] > rabs * bn
            return
        end
        dim.value = km1rnk
    end
    if b[dim.value] > predb * bn && rlenb * s[dim.value] < s[dim.value + 1]
        return
    end

    # TEST POSSIBLE RANK DEFICIENCY
    if c2 * s[dim.value] < s[dim.value + 1]
        return
    end
    i1 = km1rnk + 1
    for i = i1:prank
        dim.value = i
        if b[i] > predb * bn
            return
        end
    end
end


        """
Replaces the subroutine JACDIF
for residuals and constraints despite the name of the parameters
"""
function jacobian_forward_diff(current_point::Array{Float64},
                               current_point_dim::Int64,
                               current_residuals::Array{Float64},
                               number_of_residuals::Int64, residuals!::Function,
                               jacobian::Array{Float64, 2},
                               leading_dim_jacobian::Int64, w1::Array{Float64},
                               user_stop::Number_wrapper{Int64})
    delta = sqrt(eps(Float64))
    xtemp = 0.0
    ctrl = Number_wrapper{Int64}(0)
    delta_j = 0.0
    for j = 1:current_point_dim
        xtemp = current_point[j]
        delta_j = max(abs(xtemp), 1.0) * delta
        current_point[j] = xtemp + delta_j
        ctrl.value = -1
        residuals!(current_point, current_point_dim, w1, number_of_residuals,
                   ctrl, jacobian, leading_dim_jacobian)
        if ctrl.value <= -10
            user_stop.value = ctrl.value
            return
        end
        for i = 1:number_of_residuals
            jacobian[i, j] = (w1[i] - current_residuals[i]) / delta_j
        end
    end
end

"""
Replaces the subroutine LSOLVE
"""
function lower_triangular_solve(n::Int64, a::Array{Float64, 2}, b::Array{Float64})
    sum = 0.0
    if n <= 0
        return
    end
    b[1] /= a[1, 1]
    if n == 1
        return
    end
    jm = 0
    for j = 2:n
        sum = b[j]
        jm = j - 1
        for k = 1:jm
            sum -= a[j, k] * b[k]
        end
        b[j] = sum / a[j, j]
    end
end

"""
Replaces the subroutine USOLVE
puts the solution in b
"""
function upper_triangular_solve(n::Int64, a::Array{Float64, 2}, b::Array{Float64})
    s = 0.0
    nm = n - 1
    j = 0
    jp = 0
    if n <= 0
        return
    end
    b[n] /= a[n, n]
    if n == 1
        return
    end
    for jc = 1:nm
        j = n - jc
        s = b[j]
        jp = j + 1
        for k = jp:n
            s -= a[j, k] * b[k]
        end
        b[j] = s / a[j, j]
    end
end


"""
Replaces the subroutine YCOMP
"""
function d_minus_c1_times_x1(kp1::Int64, kq::Int64, d::Array{Float64},
                             c1::Array{Float64, 2}, x1::Array{Float64})
    sum = 0.0
    if kq <= 0 || kp1 <= 0
        return
    end
    for j = 1:kq
        sum = 0.0
        for k = 1:kp1
            sum += c1[j, k] * x1[k]
        end
        d[j] -= sum
    end
end


"""
Replaces the subroutine JTRJ
(maybe faster to just do g = g' *g but uses more memory)
"""
function g_transpose_times_g(g::Array{Float64, 2}, n::Int64,
                             work_area::Array{Float64})
    sum = 0.0
    for j = 1:n
        for i = 1:n
            work_area[i] = g[i, j]
        end
        for k = j:n
            sum = 0.0
            for i = 1:n
                sum += g[i, k] * work_area[i]
            end
            g[k, j] = sum
        end
    end
end

"""
Replaces the subroutine NEWTON
"""
function newton_search_direction(residuals!::Function, constraints!::Function,
                                 current_point::Array{Float64},
                                 number_of_parameters::Int64, c::Array{Float64},
                                 leading_dim_c::Int64, number_of_residuals::Int64,
                                 rank_c2::Int64,
                                 current_residuals::Array{Float64},
                                 p3::Array{Int64}, d3::Array{Float64},
                                 estimated_lagrange_mult::Array{Float64},
                                 a::Array{Float64, 2}, leading_dim_a::Int64,
                                 active_constraints::Array{Int64},
                                 number_of_active_constraints::Int64, rank_a::Int64,
                                 d1::Array{Float64}, p1::Array{Int64},
                                 p2::Array{Int64}, d2::Array{Float64},
                                 b::Array{Float64},
                                 current_constraints::Array{Float64},
                                 number_of_constraints::Int64,
                                 leading_dim_fmat::Int64, pivot::Array{Float64},
                                 gmat::Array{Float64, 2}, leading_dim_gmat::Int64,
                                 search_direction::Array{Float64},
                                 number_of_eval::Number_wrapper{Int64},
                                 error::Number_wrapper{Int64},
                                 fmat::Array{Float64, 2}, d::Array{Float64},
                                 v1::Array{Float64}, v2::Array{Float64})
    tp1 = rank_a + 1
    nmt = n - t
    nmr = n - rank_a
    for i = 1:number_of_residuals
        d[i] = -current_residuals[i]
    end
    p3utq3(p3, nmt, c[1,tp1], leading_dim_c, d3, nmr, d, number_of_residuals,
           1,v2)
    p3utq3(p3, nmt, c[1,tp1], leading_dim_c, d3, nmr, c, number_of_residuals,
           rank_a, v2)
    if rank_a != 3
        for i = 1:rank_a
            search_direction[i] = b[i]
        end
        if number_of_active_constraints == rank_a
            lower_triangular_solve(number_of_active_constraints,
                                   search_direction)
        elseif number_of_active_constraints > rank_a
            upper_triangular_solve(rank_a, gmat, search_direction)
        end
        if rank_a == number_of_parameters
            @goto compute_search_direction
	      end	
    end
    @views c2tc2(c[1, tp1], nmr, p3, nmt, v2)
    hessf(residuals!, gmat, leading_dim_gmat, current_point,
          number_of_parameters, current_residuals, v1, v2, number_of_residuals,
          error)

    if error.value <= -10
        return
    end
    number_of_eval.value = 2 * number_of_parameters * (number_of_parameters + 1)

    if number_of_active_constraints != 0
        hessh(constraints!, gmat, leading_dim_gmat, current_point,
              number_of_parameters, v, active_constraints,
              number_of_active_constraints, v1, v2, number_of_constraints, error)
        ecomp(p2, a, leading_dim_a, number_of_parameters, d1, pivot, rank_a,
              number_of_active_constraints, gmat, leading_dim_gmat)
    end
    @views w_plus_c(gmat[tp1:end, :], c, nmr, number_of_parameters)
    if rank_a != 0
        ycomp()
    end
    j = 0
    for i = 1:nmr
        j = rank_a + i
        search_direction[j] = d[i]
    end
    info = Number_wrapper{Int64}(0)
    #try to do a cholesky decomposition and solve
    # gmat[tp1:n, tp1:n] * x = dx[tp1:n]
    try
        @views LAPACK.posv!('L',
                            gmat[tp1:number_of_parameters, tp1:number_of_parameters],
                            search_direction[tp1:number_of_parameters])
    catch PosDefException
        error.value = -3
        return
    end

    error.value = 0
    if number_of_active_constraints != rank_a
        p_times_v(p2, rank_a, search_direction, 1)
    end
    @label compute_search_direction
    if rank_a == 0
        return
    end
    for i = 1:rank_a
        j = rank_a - i + 1
        householder_transform(2, j, j+1, number_of_parameters,
                              view(a, j:leading_dim_a, 1:number_of_parameters),
                              leading_dim_a, d1[j], search_direction, 1,
                              leading_dim_a, pivot[j])
    end
end

"""
Replaces the subroutine P3UTQ3
"""
function p3utq3(p3::Array{Int64}, nmt::Int64, c2::Array{Float64 ,2},
                leading_dim_c::Int64, d3::Array{Float64}, rank_c2::Int64,
                c::Array{Float64, 2}, number_of_residuals::Int64, rank_a::Int64,
                v2::Array{Float64})
    if rank_c2 > 0
        for i = 1:rank_c2
            @views householder_transform(2, i, i+1, number_of_residuals,
                                  c2[:, i:rank_c2], 1, d3[i],
                                  c, 1, leading_dim_c, rank_a, c2[i, i])
        end
        sum = 0.0
        for j = 1:rank_a
            for k = 1:rank_c2
                v2[k] = c[k, j]
            end
            for i = 1:rank_c2
                sum = 0.0
                for k = 1:i
                    sum += c2[k, i] * v2[k]
                end
                c[i, j] = sum
            end
        end
    end
    p_times_v(p3, nmt, c, rank_a)
end
"""
Replaces the subroutine C2TC2
"""
function c2tc2(c2::AbstractArray{Float64, 2}, rank_c2::Int64,
               p3::Array{Int64}, nmt::Int64, v2::Array{Float64})
    k = 0
    if rank_c2 > 1
        for i = 2:rank_c2
            k = i - 1
            for j = 1:k
                c2[i, j] = 0.0
            end
        end
    end
    v_times_p_transpose(p3, nmt, c2, rank_c2)
    g_transpose_times_g(c2, rank_c2, v2)
end

"""
Replaces the subroutine ECOMP
"""
function ecomp(p2::Array{Int64}, a::Array{Float64 , 2}, leading_dim_a::Int64,
               n::Int64,
               d1::Array{Float64}, pivot::Array{Float64}, rank_a::Int64, t::Int64,
               gmat::Array{Float64}, leading_dim_gmat)
    for i = 1:rank_a
        @views householder_transform(2, i, i+1, n, a[i:end, :], leading_dim_a,
                                     d1[i], gmat, leading_dim_g, 1, n, pivot[i])
    end
    for i = 1:rank_a
        @views householder_transform(2, i, i+1, n, a[i:end, :], leading_dim_a,
                                     d1[i], gmat, leading_dim_g, n, pivot[i])
    end
    if t == rank_a
        return
    end
    p_transpose_times_v(p2, rank_a, gmat, rank_a)
    v_times_p(gmat, rank_a, rank_a, p2)
end
"""
Replaces the subroutine WCOMP
w and c are nmt*n (but with different leading dimension)
"""
function w_plus_c(w::AbstractArray{Float64, 2}, c::AbstractArray{Float64, 2},
                  nmt::Int64, n::Int64)
    for i = 1:nmt
        for j = 1:n
            w[i,j] += c[i, j]
        end
    end
end

"""
Replaces the subroutine HSUM
"""
function constraints_merit(active_constraints::Array{Int64}, t::Int64,
                           h::Array{Float64}, w::Array{Float64}, l::Int64)

    sum = 0.0
    hval = 0.0
    if l <= 0
        return
    end
    for j = 1:l
        if !(j in active[1:t])
            hval = min(0.0, h[j])
        else
            hval = h[j]
        end
        hsum+= w[j]* (hval ^ 2)
    end
    return hsum
end

"""
-
Replaces the subroutine UNSCR
"""
function unscramble_array(active_constraints::Array{Int64}, bnd::Int64, l::Int64,
                          p::Int64)
    lm = l - p
    if lmp <= 0
        return
    end
    j = 0
    for i = 1:lmp
        j = bnd + i
        if active_constraints[j] == -1
            active_constraints[j] = 0
        end
        if active_constraints[j] >= 2
            active_constraints[j] = 1
        end
    end
end


"""
Replaces the subroutine PREOBJ
"""
function preobj(c::Array{Float64, 2}, m::Int64, rank_a::Int64, dx::Array{Float64},
                f::Array{Float64}, t::Int64, fc1dy1::Number_wrapper{Float64},
                c1dy1::Number_wrapper{Float64})
    velem = 0.0
    fc1dy1.value = 0.0
    c1dy1.value = 0.0
    if t <= 0
        return
    end
    for i = 1:m
        velem = 0.0
        for j = 1:rank_a
            velem += c[i,j] * dx[j]
        end
        fc1dy1.value += f[i] * velem
        c1dy1.value += velem ^ 2
    end
end


"""
Replaces the subroutine NZESTM
lm = lagrange multiplier
"""
function nonzero_first_order_lm(a::Array{Float64}, rank_a::Int64,
                                number_of_active_constraints::Int64,
                                b::Array{Float64}, v::Array{Float64},
                                w::Array{Float64}, u::Array{Float64})
    coptyto!(w, 1, b, 1, number_of_active_constraints)
    lower_triangular_solve(rank_a, a, w)
    solve_t_times_t(a, rank_a, w, u, number_of_active_constraints,
                          Number_wrapper{Float64})
    v .+= u
end

    
    
"""
Replaces the subroutine LEAEST
"""
function special_lagrange_mult(a::Array{Float64, 2},
                                       number_of_active_constraints::Int64,
                                       current_residuals::Array{Float64},
                                       number_of_residuals::Array{Float64},
                                       v1::Array{Float64},
                                       c::Array{Float64, 2},
                                       p1::Array{Int64},
                                       scale::Int64,scaling_matrix::Array{Float64},
                                       v2::Array{Float64}, v::Array{Float64},
                                       lm_residual::Number_wrapper{Float64})

    if number_of_active_constraints <= 0
        return
    end
    sum = 0.0
    for j = 1:number_of_active_constraints
        sum = 0.0
        for i = 1:number_of_residuals
            sum += c[i,j] * (current_residuals[i] + v1[i])
        end
        v2[j] = sum
    end
    solve_t_times_t!(a, number_of_active_constraints, v2, v,
                           number_of_active_constraints, lm_residual)
    p_times_v(p=p1, m=number_of_active_constraints, v=v, n=1)
    if scale == 0
        return
    end
    for i = 1:number_of_active_constraints
        v[i] *= scaling_matrix[i]
    end
end
"""
Replaces the subroutine EUCMOD
"""
function minimize_euclidean_norm(ctrl::Int64, old_penalty_constants::Array{Float64},
                                 number_of_constraints::Int64,
                                 positive_elements_l::Array{Int64},
                                 number_of_pos_elements_l::Int64,
                                 y::Array{Float64}, tau::Float64,
                                 new_penalty_constants::Array{Float64},
                                 working_area::Array{Float64})
    if number_of_pos_elements_l <= 0
        return
    end
    copyto!(working_area, 1, working_area, 1, number_of_constraints)
    @views y_norm = norm(y[1:number_of_constraints])
    y_norm_sq = y_norm ^ 2
    scale_vector(y, y_norm, 1, number_of_constraints)
    tau_new = tau
    sum = 0.0
    nrunch = number_of_constants
    istop = 0
    constant = 0.0
    k = 0
    i = 0
    prod = 0.0
    while true
        tau_new -= sum
        if y_norm_sq == 0.0
            constant = 1.0
        else
            constant = tau_new / y_norm_sq
        end
        y_norm_sq = 0.0
        sum = 0.0
        istop = nrunch
        k = 1
        while k <= nrunch
            i = positive_element_l[k]
            prod = constant * y[k] * y_norm
            if prod >= working_area[i]
                old_penalty_constants[i] = prod
                y_norm_sq += y[k] ^ 2
                k += 1
                continue
            end
            sum += old_penalty_constant[i] * y[k] * y_norm
            for j = k:nrunch
                positive_elements_l[j] = positive_elements_l[j+1]
                y[j] = y[j+1]
            end
            nrunch -= 1
        end
        y_norm_sq *= y_norm ^ 2
        if nrunch <= 0 || ctrl == 2 || istop == nrunch
            break
        end
    end
end

"""
Replaces the subroutine SACTH
"""
function sum_sq_active_constraints(current_constraints::Array{Float64},
                                   active_constraints::Array{Int64},
                                   number_of_active_constraints::Int64)

    sum = 0.0
    if number_of_active_constraints != 0
        for i = 1:number_of_active_constraints
            sum += current_constraints[active_constraints[i]] ^ 2
        end
    end
    return sum
end

