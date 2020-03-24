using Printf
using ForwardDiff
using LinearAlgebra
"""
Replaces the common variable block PREC
"""
mutable struct Last_two_points
    rkakm2::Int
    rkckm2::Int
    kodkm2::Int
    rkakm1::Int
    rkckm1::Int
    kodkm1::Int
    tkm2::Int
    tkm1::Int
    betkm2::Float64
    d1km2::Float64
    dkm2::Float64
    fsqkm2::Float64
    hsqkm2::Float64
    b1km2::Float64
    dxnkm2::Float64
    alfkm2::Float64
    betkm1::Float64
    d1km1::Float64
    dkm1::Float64
    fsqkm1::Float64
    hsqkm1::Float64
    b1km1::Float64
    dxnkm1::Float64
    alfkm1::Float64
    pgress::Float64
    prelin::Float64
end
"""
Replaces common variable block BACK
"""
mutable struct Restart_steps
    bestrk::Float64
    bestpg::Float64
    nrrest::Int
    lattry::Int
end
"""
Replaces common variable block negdir
"""
mutable struct Negdir
    ifree::Int
end

#######
# The wrapper below is used to simulate passing parameters by reference

#######
mutable struct Number_wrapper{T<:Number}
    value::T
end

Base.copy(w::Number_wrapper{T}) = Number_wrapper(w.value)



"""
Replaces the subroutine NEWPNT
Compute the jacobian of the constraints and the residuals and put them
respectively in `jac_constraints` and `jac_residuals`

"""

function new_point(current_point::Array{Float64}, number_of_parameters::Int,
                   current_constraints::Array{Float64},
                   number_of_constraints::Int, current_residuals::Array{Float64},
                   number_of_residuals, constraints!::Function,
                    residuals!::Function, leading_dim_jac_constraints::Int,
                    leading_dim_jac_residuals::Int,
                    number_of_eval::Number_wrapper{Int},
                    jac_constraints::Array{Float64, 2},
                    jac_residuals::Array{Float64, 2},
                    b::Array{Float64}, d::Array{Float64}
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
        if user_stop.valu < -10
            return
        end
        number_of_eval.value += number_of_parameters
    end
    if number_of_constraints <= 0
        return
    end
    ctrla = Number_wrapper(2)

    constraints!(current_point, current_point_dim, current_constraints,
                 number_of_constraints, ctrla, jac_constraints,
                 leading_dim_jac_constraints)

    if ctrla.value < 10
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
        b[i] = -constraints[i]
    end
end
"""
Replaces the subroutine EQUAL
"""
function equal(b::Array{Float64}, l::Int, a::Array{Float64,2},
                leading_dim_a::Int, n::Int, active_constraints::Array{Float64},
                t::Int, p::Int, p4::Array{Int})
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
                       leading_dim_jacobian::Int, number_of_active_constraints,
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
function gn_seach(fmat_is_identity::Int, a::Array{Float64,2},
                  leading_dim_a::Int, number_of_constraints::Int,
                  number_of_parameters::Int, d1::Array{Float64}, p1::Array{Int},
                  rank_a::Int, number_of_householder::Int, b::Array{Float64},
                  fmat::Array{Float64,2}, leading_dim_f::Int,
                  jac_residuals::Array{Float64,2},
                  leading_dim_jac_residuals::Int,
                  current_residuals::Array{Float64},pivot::Array{Float64},
                  tau::Float64, leading_dim_g::Int, scale::Int,
                  diag::Array{Float64,2}, inactive_constraints::Array{Int},
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

    c_q1_h_p2_product(fmat_is_identity, c, leading_dim_c, number_of_parameters,
                      number_of_residuals, number_of_active_constraints,
                      rank_a, pivot, number_of_householder, a, leading_dim_a, d1,
                      fmat, leading_dim_f, p2, work_area_s)
    kc2 = rank_a + 1
    nmp = number_of_parameters - rank_a
    tol = sqrt(Float64(nmp)) * tau
    c2_to_upper_to_triangular(number_of_residuals, nmp, @view c[:, kc2:end],
    leading_dim_c, tol, p3, rank_c2, d3)
    sub_search_direction(fmat_is_identity, a, leading_dim_a,
                         number_of_active_constraints, number_of_parameters,
                         d1, p1, rank_a, rank_a, number_of_householder,
                         b, fmat, leading_dim_f, c, leading_dim_c,
                         number_of_residuals, current_residuals, pivot, gmat,
                         leading_dim_g, d2, p2, p3, d3, rank_c2.value,
                         inactive_constraints, number_of_inactives_constraints,
                         p4, rank_c2.value, code, scale, diag, d, gn_direction,
                         v1, d1_norm, d_norm, b1_norm, work_area_u)
end

function min_max_langrange_mult(number_of_equality_constraints::Int,
                                number_of_active_constraints::Int,
                                estimated_lagrange_mult::Array{Float64},
                                scale::Int,
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
function check_last_step(iteration_number::Int, restart::Bool,
                         code::Number_wrapper{Int}, sq_sum_residuals::Float64,
                         d1_norm::Number_wrapper{Float64},
                         d_norm::Number_wrapper{Float64},
                         c::Array{Float64, 2}, leading_dim_c::Int,
                         number_of_residuals::Int, number_of_parameters::Int,
                         rank_c2::Int, d::Array{Float64},
                         current_residuals::Array{Float64}, p3::Array{Float64},
                         d3::Array{Float64}, active_constraints::Array{Int},
                         estimated_lagrange_mult::Array{Float64},
                         inactive_constraints::Array{Int},
                         number_of_inactive_constraints::Int,
                         p4::Array{Int}, deleted_constraints::Int,
                         a::Array{Float64, 2}, leading_dim_a::Int,
                         number_of_equality_constraints::Int,
                         number_of_active_constraints::Int,
                         rank_a::Int, b1_norm::Number_wrapper{Float64},
                         sq_sum_constraints::Float64, number_of_householder::Int,
                         d1::Array{Float64},
                         p1::Array{Int}, d2::Array{Float64}, p2::Array{Int},
                         b::Array{Float64}, h::Array{Float64},
                         number_of_constraints::Int,
                         fmat::Array{Float64, 2}, leading_dim_f::Int,
                         pivot::Array{Float64}, gmat::Array{Float64, 2},
                         leading_dim_g::Int, residuals!::Function,
                         constraints!::Function,
                         current_point::Array{Float64}, hessian::Bool,
                         constraint_added:Bool, constraint_deleted::Bool,
                         scale::Int, scalingmat::Array{Float64, 2},
                         gn_direction::Array{Float64},
                         gn_direction_norm::Number_wrapper{Float64},
                         v1::Array{Float64}, relative_termination_tol::Float64,
                         error::Number_wrapper{Int},
                         number_of_eval::Number_wrapper{Int},
                         d1_plus_b1_norm::Number_wrapper{Float64},
                         dim_a::Number_wrapper{Int},
                         dim_c2::Number_wrapper{Int}, v2::Array{Float64},
                         work_area::Array{Float64},
                         ltp::Last_two_points, restart_steps::Restart_steps)

    ind = 0
    if !restart
        ind = ltp.rkckm1 + ltp.tm1 - number_of_active_constraints
        ind -= 1
        ltp.d1km2 = norm(d[1:ind])
    end
    #gndchk
    number_of_eval.value = 0
    code.value = ind
    error.value = 0
    if ind == 1
        dim_a.value = rank_a
        dim_c2.value = rank_c2
        return
    end
    if ind != 2
        #subspc()
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
function sub_search_direction(deleted_constraints::Int, a::Array{Float64, 2},
                              leading_dim_a::Int,
                              number_of_active_constraints::Int,
                              number_of_parameters::Int,
                              d1::Array{Float64}, p1::Array{Int}, dim_a::Int,
                              rank_a::Int, number_of_householder::Int,
                              b::Array{Float64}, fmat::Array{Float64, 2},
                              leading_dim_f::Int,
                              c::Array{Float64, 2}, leading_dim_c::Int,
                              number_of_residuals::Int,
                              current_residuals::Array{Float64},
                              pivot::Array{Float64}, gmat::Array{Float64,2},
                              leading_dim_g::Int, d2::Array{Float64},
                              p2::Array{Int}, p3::Array{Int},
                              d3::Array{Float64}, dim_c2::Int,
                              inactive_constraints::Array{Int},
                              number_of_inactive_constraints::Int, p4::Array{Int},
                              rank_c2::Int, code::Int, scale::Int,
                              scaling_matrix::Array{Float64}, d::Array{Float64},
                              search_direction::Array{Float64},
                              v1::Array{Float64}, d1_norm::Number_wrapper{Float64},
                              d_norm::Number_wrapper{Float64},
                              b1_norm::Number_wrapper{Float64},
                              work_area::Array{Float64})
    b1_norm.value = 0.0
    if number_of_active_constraints > 0
        copyto!(seach_direction, 1, b, 1,
                number_of_active_constraints)
        b1_norm.value = norm(search_direction[1:dim_a])
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
    d3_i_dummy = Number_wrapper{Float64}
    c_ik_dummy = Number_wrapper{Float64}
    if rank_c2 > 0
        for i=1:rank_c2
            k = rank_a + i
            d3_i_dummy.value = d3[i]
            c_ik_dummy.value = c[i, k]
            householder_transform(2, i, i+1, number_of_residuals, @view c[:, k:end],
                                  1, d3_i_dummy, d, 1, number_of_residuals, 1,
                                  c_ik_dummy)
            d3[i] = d3_i_dummy.value
            c[i, k] = c_ik_dummy.value
        end
    end
    d1_norm = d[1:dim_c2]
    k = rank_a + 1
    nmt = number_of_parameters - rank_a
    orthogonal_projection(c, leading_dim_c, number_of_residuals, dim_a,
                          search_direction, @view c[:, k:end], rank_c2, dim_c2, d3,
                          d, v1)
    if nmt > 0
        copyto!(search_direction, rank_a+1, d, 1, nmt - 1)
        upper_triangular_solve(leading_dim_c, @view c[:, k:end],
                               @view search_direction[k:end])
        if dim_c2 != nmt
            k = dim_c2 + 1
            for i = k, nmt
                j = rank_a + i
                search_direction[i] = 0.0
            end
        end
    end
    #??????
    p_times_v()
    if code != 1
        #pv()
    end
    if deleted_constraints > 2 || (deleted_constraints !== 3 || length(b) != 0)
        h_times_x(fmat, leading_dim_f, number_of_parameters, search_direction,
                  work_area)
    end

    no::Int = number_of_householder - length(b)
    irow::Int = 0
    i2::Int = 0
    i3::Int = 0
    no_elem::Int = 0
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
    d1_k_dummy = Number_wrapper{Float64}
    pivot_k_dummy = Number_wrapper{Float64}
    for i=1:number_of_householder
        k = number_of_householder - i + 1
        d1_k_dummy.value = d1[k]
        pivot_k_dummy.value = pivot[k]
        #a[k:end, :]???
        householder_transform(2, k, k+1, number_of_parameters, @view a[k:end, :],
                              leading_dim_a, d1_k_dummy, search_direction, 1,
                              leading_dim_a, 1,
                              pivot_k_dummy)
        d1[k], pivot[k] = d1_k_dummy.value, pivot_k_dummy.value
    end
end

"""
Replaces the subroutine UPBND
"""
function upper_bound_steplength(a::Array{Float64, 2},
                                leading_dim_a::Int,
                                number_of_inactive_constraints::Int,
                                number_of_householder::Int,
                                number_of_active_constraints::Int,
                                number_of_parameters::Int,
                                current_constraints::Array{Float64},
                                inactive_constraints::Array{Int}, p4::Array{Int},
                                v1::Array{Float64}, number_of_residuals::Int,
                                search_direction::Array{Float64},
                                upper_bound::Number_wrapper{Float64},
                                upper_bound_cons_index::Number_wrapper{Int})

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
                                   active_constraints::Array{Int},
                                   number_of_active_constraints::Number_wwrapper{Int},
                                   min_l_n::Int, number_of_equality::Int,
                                   inactive_constraints::Array{Int},
                                   number_of_inactive_constraints::Number_wrapper{Int},
                                   ind::Int, iteration_number::Int,
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
            kind = 0
            if t >= min_l_n
                kind = 1
                max_constraint, kk = 0.0, 0
                for j = 1:number_of_active_constraints
                    jj = active_constraints[j]
                    if current_constraints[jj] > max_constraint
                        max_constraint, kk = current_constraint[jj], j
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
            elseif active_constraints[j] == 0
                active_constraints[j] = 1
            end
            constraint_added.value = true
            if kind == 1
                i = number_of_inactive_constraints.value + 1
            end
        end
        i += 1
    end
end
"""
Replaces the subroutine SCALV

    vector ./= factor
"""

"""
Replaces the subroutine EVREST
"""
function evaluation_restart(current_point::Array{Float64},
                            previous_point::Array{Float64},
                            number_of_parameters::Int,
                            number_of_residuals::Int,
                            iteration_number::Number_wrapper{Int},
                            residuals!::Function,
                            number_of_eval::Number_wrapper{Int},
                            current_residuals::Array{Float64}, d1_norm::Float64,
                            d_norm::Float64,
                            sq_sum_previous_residuals::Number_Wrapper{Float64},
                            dim_c2::Int, code::Int,
                            search_direction_norm::Float64,
                            d1_plus_b1_norm::Float64, current_steplength::Float64,
                            lower_bound_steplength::Float64,
                            active_constraints::Array{Int},
                            current_constraints::Array{Float64},
                            number_of_constraints::Int,
                            number_of_active_constraints::Int,
                            constraints!::Funtion, b1_norm::Float64,
                            sq_sum_constraints::Number_wrapper{Float64},
                            dim_a::Int, error::Number_wrapper{Int},
                            restart::Number_wrapper{Bool}, ltp::Last_two_steps,
                            restart_steps::Restart_steps, negdir::Negdir)

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
            sq_sum_constraints.value = sum(x -> x ^ 2,
            current_constraints[active_constraints])
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
function output(printing_stepsize::Int, iteration_number::Int, unit::Int,
    gres::Float64, penalty_weights::Array{Float64},
    active_constraints::Array{Int}, convergence_factor::Number_wrapper{Float64},
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
        @printf("\n\n\n\n COLLECTED INFORMATION FOR ITERATION STEPS" +
                "K   FSUM(K)   HSUM(K)   LAGRES   DXNORM    KODA   KODC   ALPHA" +
                "CONV.SPEED    MAX(W)    PREDICTED    REDUCTION")
    end
    if ltp.tkm1 > 0
        @printf("%4d%e11.3%e11.3%e10.3%e10.3%4d%5d%e10.3%e10.3%e10.3%e10.3%e10.3",
        itno, ltp.fsqkm1, ltp.hsqkm1, gres, ltp.dxnkm1, ltp.rkakm1,
        ltp.alfkm1, convergence_factor.value, wmax, ltp.prelin,
        ltp.prgress)
    else
        @printf("%4d%e11.3%e11.3%e10.3%e10.3%4d%5d%e10.3%e10.3%e10.3%e10.3%e10.3",
        itno, ltp.fsqkm1, ltp.hsqkm1, gres, ltp.dxnkm1, ltp.rkakm1,
        ltp.alfkm1, convergence_factor.value, wmax, ltp.prelin, ltp.prgress)
    end
end


"""
Replaces the subroutine ADX
"""

function search_direction_product(code::Int, gmat::Array{Float64, 2},
                                  leading_dim_g::Int
                                  rank_a::Int, number_of_non_zero_in_b::Int,
                                  p1::Array{Int}, d2::Array{Float64},
                                  b::Array{Float64},
                                  number_of_active_constraints::Int,
                                  number_of_householder::Int,
                                  scale::Int, scaling_matrix::Array{Float64},
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
            householder_transform(2, k, k+1, number_of_active_cosntraints,
                                  gmat[1, k:end], 1, d2_k_dummy, product,
                                  1, number_of_active_constraints, gmat[:, k:end])
        end
    end
    if length(b) < number_of_householder
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
function l_to_upper_triangular(a::Array{Float64, 2}, leading_dim_a::Int,
                               rank_a::Int, number_of_parameters::Int,
                               b::Array{Float64}, leading_dim_g, p2::Array{Int},
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
    cmax = Number_wrapper{Int}
    collng = Number_wrapper{Float64}
    for i=1:rank_a
        max_partial_row_norm(number_of_parameters, rank_a, gmat, leading_dim_g,
                             i, i, cmax, collng)
        p2[i] = cmax
        permute_columns(gmat, leading_dim_g, number_of_parameters, i, cmax.value)
        d2_i_dummy = Number_wrapper(d2[i])
        gmat_ii_dummy = Number_wrapper(gmat[i, i])
        householder_transform(1, i, i+1, number_of_parameters, gmat,[:, i:end],
                              1, d2_i_dummy, gmat[:, i+1], 1,
                              leading_dim_g, rank_a-i, gmat_ii_dummy)
        householder_transform(2, i, i+1, number_of_parameters, gmat[:, i:end],
                              1, d2_i_dummy, b, 1, number_of_parameters, 1,
                              gmat_ii_dummy)
        d2[i] = d2_i_dummy.value
        gmat[i, i] = gmat_ii_dummy.value
    end
    return

end

"""
Replaces the subroutine ATOLOW
"""
function a_to_lower_triangular(number_of_rows_a::Int, length_g::Int,
                               a::Array{Float64, 2}, leading_dim_a::Int,
                               tol::Float64, current_gradient::Array{Float64},
                               p1::Array{Int}, pseudo_rank_a::Number_wrapper{Int},
                               d1::Array{Float64})
    pseudo_rank_a.value = number_of_rows_a
    if number_of_rows_a == 0
        return
    end
    ldiag = min(number_of_rows_a, length_g)
    p1[1:ldiag] = [1:ldiag]
    krank = 0
    imax = Number_wrapper{Int}
    rmax = Number_wrapper{Float64}
    for i = 1:ldiag
        krank = i
        max_partial_row_norm(number_of_rows_a, length_g, a, leading_dim_a,
                             i, i, imax, rmax)
        if rmax.value >= tol
            break
        end
        p1[i] = imax.value
        permute_row(a, leading_dim_a, length_g, i, imax.value)
        d1_i_dummy = Number_wrapper(d1[i])
        a_ii_dummy = Number_wrapper(a[i, i])
        householder_transform(1, i, i+1, length_g, a[i:end, :], leading_dim_a,
                             d1_i_dummy, a[i+1:end, :], leading_dim_a, 1, t-i,
                              a_ii_dummy)
        householder_transform(2, i, i+1, length_g_, a[i:end, :], leading_dim_a,
                              d1_i_dummy, g, 1, length_g, 1, a_ii_dummy)
        d1[i] = d1_i_dummy.value
        a[i, i] = a_ii_dummy.value
        krank = i + 1
    end
    pseudo_rank_a = krank - 1
    return
end
"""
check how to pass view
Replaces subroutine C2TOUP
"""

function c2_to_upper_trianguar(number_of_rows_c2::Int, number_of_col_c2::Int,
                               c2::Array{Float64, 2}, leading_dim_c2::Int,
                               tol::Float64, p3::Array{Int},
                               pseudo_rank_c2::Number_wrapper{Int},
                               d3::Array{Float64})
    pseudo_rank_c2.value = min(number_of_rows_c2, number_of_col_c2)
    if number_of_col_c2 == 0 || number_of_rows_c2 == 0
        return
    end
    p3 = [1:number_of_col_c2]
    ldiag = pseudo_rank_c2.value
    kmax = Number_wrapper{Int}
    rmax = Number_wrapper{Float64}
    for k = 1:ldiag
        max_partial_row_norm(number_of_rows_c2, number_of_col_c2, c2,
                             leading_dim_c2, k, k, kmax, rmax)
        permute_columns(c2, leading_dim_c2, number_of_rows_c2, k, kmax.value)
        d3_k_dummy = Number_wrapper(d3[k])
        c2_kk_dummy = Number_wrapper(c2[k, k])
        householder_transform(1, k, k+1, number_of_rows_c2, c2[:, k:end],
                              1, d3_k_dummy, c2[:, k+1:end], 1, leading_dim_c,
                              number_of_col_c2-k, c2_kk_dummy)
        d3[k] = d3_k_dummy.value
        c2[k, k] = c2_kk_dummy.value
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
function orthogonal_projection(c1::Array{Float64, 2}, leading_dim_c::Int,
                               m::Int, dim_a::Int, b1::Array{Float64},
                               c2::Array{Float64, 2}, rank_c2::Int,
                               non_zero_el_dv::Int, d2::Array{Float64},
                               dv::Array{Float64}, fprim::Array{Float64})
    for i = 1:m
        fprim[i] = (i <= dim_c2) ? dv[i] : 0.0
    end
    up_wrapper = Number_wrapper{Float64}
    pivot_wrapper = Number_wrapper{Float64}
    if rank_c2 > 0
        k = 0
        for i = 1:rank_c2
            k = rank_c2 + 1 - i
            up_wrapper.value = d2[k]
            pivot_wrapper = c2[k, k]
            #??? parametres pivot_vector est une matrice
            householder_transform(2, k, k+1, m, c2[:, k], 1, up_wrapper, fprim, 1,
                                  m, 1, pivot_wrapper)
            d2[k] = up_wrapper.value
            c2[k, k] = pivot_wrapper.value
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
function max_partial_row_norm(m::Int, n::Int, a::Array{Float64, 2},
                              starting_col::Int,starting_row::Int,
                              max_row_index::Number_wrapper{Int},
                              max_row_norm::Number_wrapper{Float64})

    max_row_norm.value = -1.0
    for i = starting_row:m
        row_norm = norm(a[i, starting_col:n])
        if row_norm > max_row_norm.value
            max_row_norm.value = row_norm
            max_row_index.value = i
        end
    end
end
"""
Replaces de subroutine COLMAX
"""
function max_partial_row_norm(m::Int, n::Int, a::Array{Float64, 2},
                              starting_col::Int,starting_row::Int,
                              max_col_index::Number_wrapper{Int},
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
function permute_rows(a::Array{Float64,2}, n::Int, row1::Int, row2::Int)
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
function permute_columns(a::Array{Float64, 2}, m::Int, col1::Int,
    col2::Int)

    for i = 1:m
        a[i, col1], a[i, col2] = a[i, col2], a[i, col1]
    end
end

"""


Replaces subroutine PTRV
"""
function p_transpose_times_v(p::Array{Int}, m::Int, v::Array{Float64,2},
    n::Int)
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
function v_times_p_transpose(p::Array{Int}, m::Int, v::Array{Float64,2},
    n::Int)
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
function p_times_v(p::Array{Int}, m::Int, v::Array{Float64,2},
                                   n::Int)
    if m <= 0 || n <= 0
        return
    end
    k = 0
    for i = 1:m
        k = m + 1 - i
        permute_rows(a=v, n=n, row1=k, row2=p[k])
    end
end
"""
Replaces subroutine VP
"""
function v_times_p(v::Array{Float64, 2},
                   m::Int, n::Int, p::Array{Int})
    if m <= 0 || n <= 0
        return
    end
    for i = 1:n
        permute_columns(a=v, m=m, col1=i, col2=p[i])
    end
end

"""
Replaces the routine H12PER
Modifies the parameter c
"""
function householder_transform(mode::Int, pivot_index::Int, l1::Int, m::Int,
    pivot_vector::Array{Float64, 2}, pivot_vect_number_rows::Int,
    up::Number_wrapper{Float64}, c::Array{Float64}, inc_elements_c::Int,
    inc_vectors_c::Int, number_of_vectors_c::Int, pivot::Number_wrapper{Float64})

    if 0 <= pivot_index || pivot_index >= l1 || l1 < m
        return
    end
    cl = abs(pivot.value)
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
function c_times_h(m::Int, n::Int, c::Array{Float64, 2}, leading_dim_c::Int,
    h::Array{Float64, 2}, leading_dim_h::Int, work_area::Array{Float64})
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
function h_times_x(h::Array{Float64, 2}, leading_dim_h, n::Int, x::Array{Float64},
    s::Array{Float64})
    s = copy(x)
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

function apply_rotation(c::Float64, s::Float64, x::Number_wrapper{Float64},
                        y::Number_wrapper{Float64})
    xt = c * x.value + s * y.value
    y.value = s * x.value - c * y.value
    x.value = xt
end
"""
Replaces subroutine PSPECF
"""
function permute(n::Int, p::Array{Int}, w::Array{Float64}, f::Array{Float64})

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
    leading_dim_h::Int, s::Array{Float64}, p::Array{Float64},
    beta::Array{Float64}, j::Int, tp1::Int)

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
function c_q1_h_p2_product(time::Int, c::Array{Float64, 2}, leading_dim_c::Int,
                           m::Int, n::Int, t::Int, pseudo_rank_a::Int,
                           pivot::Array{Float64}, na::Int, a::Array{Float64},
                           leading_dim_a::Int, d1::Array{Float64},
                           h::Array{Float64, 2}, leading_dim_h::Int,
                           p2::Array{Int}, v::Array{Float64})

    if na == 0
        return
    end
    pivot_wrapper = Number_wrapper{Float64}
    up_wrapper = Number_wrapper{Float64}
    for i = 1:na
        pivot_wrapper.value = pivot[i]
        up_wrapper.value = d1[i]
        householder_transform(mode=2, pivot_index=i, l1=i+1, m=n,
                              pivot_vector=a[i, 1],
                              pivot_vect_number_rows=leading_dim_a,
                              up=up_wrapper, c=c, inc_elements_c=leading_dim_c,
                              inc_vectors_c=1, number_of_vectors_c=m,
                              pivot=pivot_wrapper)
        pivot[i] = pivot_wrapper.value
        d1[i] = up_wrapper.value
    end
    if time > 2 || !(time == 3 && t == 0)
        c_times_h(m=m, n=n, c=c, leading_dim_c=leading_dim_c, h=h,
        leading_dim_h=leading_dim_h, work_area=v)
    end
    if pseudo_rank_a == t]
        return
    end
    v_times_p(v=c, m=m, n=pseudo_rank_a, p=p2)
end
"""
Replaces subroutine SIGNCH
"""
function sign_ch(time::Int, p1::Array{Int}, v::Array{Float64}, t::Int,
                 active_constraint::Array{Int}, bnd::Int, betkm1::Float64,
                 gnd_norm::Float64, iteration_number::Int, scale::Int,
                 scaling_matrix::Array{Float64}, gres::Float64,
                 h::Array{Float64}, kp::Int, l::Int, j::Number_wrapper{Int},
                 s::Number_wrapper{Int}, p2::Array{Float64},
                 working_area::Array{Float64})
    const ival = 4
    const delta = 10.0
    const tau = 0.5
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
        elseif v[i] * current_el >= e
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
            p1[i] = Int(p2[i]) 
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
        p1[i] = Int(working_area[i])
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
                 number_of_active_constraints::Number_wrapper{Int},
                 number_of_variables::Int, bv::Array{Float64},
                 row_of_l_to_delete::Int, s::Int, active_constraints::Array{Int},
                 inactive_constraints::Int,
                 number_of_inactive_constraints::Number_wrapper{Int},
                 p4::Array{Int}, working_area::Array{Float64},scale::Int,
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
function solve_t_times_t!(a::Array{Float64, 2}, t::Int, b::Array{Float64},
                                 x::Array{Float64}, n::Int,
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
function gradient!(jacobian::Array{Float64, 2}, m::Int, n::Int,
                  residuals::Array{Float64, gradient::Array{Float64}})
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
function delete_constraints(active_constraints::Array{Int},
                           inactive_constraints::Array{Int},
                           number_of_inactive_constraints::Number_wrapper{Int},
                           number_of_active_constraints::Number_wrapper{Int},
                           constraints_to_delete::Int)
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
function add_constraints(active_constraints::Array{Int},
                        inactive_constraints::Array{Int},
                        number_of_active_constraints::Number_wrapper{Int},
                        number_of_inactive_constraints::Number_wrapper{Int},
                        constraints_to_add:Int)
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
function steplength(current_point::Array{Float64},
                    search_direction::Array{Float64},
                    current_residuals::Array{Float64}, v1::Array{Float64},
                    number_of_residuals::Int, current_point_dim::Int,
                    alpha::Number_wrapper{Float64}, psi_at_zero::Float64,
                    derivative_psi_at_zero::Float64,
                    steplength_lower_bound::Float64, residuals!::Function,
                    constraints!::Function, current_constraints::Array{Float64},
                    next_constraints, active_constraints::Array{Int},
                    number_of_active_constraints::Int,
                    inactive_constraints::Array{Int},
                    number_of_inactive_constraints::Int,
                    number_of_constraints::Int,w::Array{Float64},
                    steplength_upper_bound::Float64,
                    next_residuals::Array{Float64}, v2::Aray{Float64},
                    g::Array{Float64}, psi_at_alpha::Number_wrapper{Float64},
                    x_diff_norm::Float64, number_of_eval::Number_wrapper{Int},
                    exit::Number_wrapper{Int})

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
                @goto 1000
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
        @goto 200
    end
    if psi_at_zero > psikm1.value
        @goto 120
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
        @goto 130
    end
    @label 120
    minrn(alfk.value, psik.value, alfkm1.value, psikm1.value, alfkm2.value,
          psikm2.value, alfmin, alfmax, pmax, relative_precision, alfkp1,
          pk)
    @label 130
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
            @goto 1000
        end
        minrn(alfk.value, psik.value, alfkm1.value, psikm1.value, alfkm2.value,
              psikm2.value, alfmin, alfmax, pmax, relative_precision, alfkp1,
              pk)
    end
    @label 200
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
    @label 1000
    diff = 0.0
    xel = 0.0
    for i = 1:current_point_dim
        xel = alfkm1.value * search_direction[i]
        current_point[i] += xel
        g[i] = xel
    end
    xdiff_norm.value = norm(g)
    @label 1020
    alpha.value = alfkm1.value
    psi_at_alpha.value = psikm1.value
    number_of_eval.value = k.value

end

"""
Replaces the subroutine CONCAT
modifies f
"""
function concat(f::Array{Float64}, m::Int, h::Array{Float64},
                active_constraints::Array{Int}, t::Int,
                inactive_constraints::Array{Float64}, q::Int,
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
function linc2(m::Int, n::Int, v1::Array{Float64},
               next_residuals::Array{Float64}, current_residuals::Array{Float64},
               alfk::Float64, current_constraints::Array{Float64},
               next_constraints::::Array{Float64}, t::Int,
               active_constraints::Array{Int}, w::Array{Float64},
               inactive_constraints::Array{Int}, q::Int, l::Int,
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
function reduce(alpha::Number_Wrapper{Float64},
                psi_at_alpha::Number_wrapper{Float64}, alpha_k::Float64,
                pk::Float64, diff::Float64, eta::Float64,
                current_point::Float64, p::Array{Float64},
                current_residuals::Array{Float64},
                next_point::Array{Float64}, next_residuals::Array{Float64},
                number_of_residuals::Int, n::Int, residuals!::Function,
                current_constraints::Array{Float64},
                next_constraints::Array{Float64}, constraints!::Function,
                t::Int, l::Int, active_constraints::Array{Int},
                w::Array{Float64}.
                k::Number_wrapper{Int}, psi_k::Float64,
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
                                    p::Array{Float64}, f::Array{Float64}, m::Int,
                                    n::Int, residuals!::Function,
                                    constraints!::Function,
                                    current_constraints::Array{Float64},
                                    active_constraints::Array{Float64},
                                    t::Int, l::Int, w::Array{Float64},
                                    k::Number_wrapper{Int},
                                    alpha_lower_bound::Float64,
                                    exit::Number_wrapper{Int},
                                    next_point::Array{Float64},
                                    next_residuals::Array{Float64},
                                    psi_at_zero::Float64, dpsi_at_zero::Float64,
                                    u::Number_wrapper{Float64},
                                    psi_At_u::Number_Wrapper{Float64},
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
        if exit.value = -2
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
function linc1(p::Array{Float64}, n::Int, alpha::Float64,
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
    if abs(v - x) < eps || abs(w - x) < eps || abs (w - v) < eps
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
Replaces the subroutine MINRM1
"""
function minrm1(v0::Array{Float64}, v1::Array{Float64}, v2::Array{Float64},
                m::Int, sqnv0::Number_wrapper{Float64},
                sqnv1::Number_wrapper{Float64}, sqnv2::Number_wrapper{Float64},
                scv0v1::Number_wrapper{Float64}, scv0v2::Number_wrapper{Float64},
                scv1v2::Number_wrapper{Float64})
    v0_norm = norm(v0])
    v1_norm = norm(v1])
    v2_norm = norm(v2])
    sqnv0.value = v0_norm ^ 2
    sqnv1.value = v1_norm ^ 2
    sqnv2.value = v2_norm ^ 2
    if v0_norm != 0.0
        v0 ./= v0_norm
    elseif v1_norm != 0.0
        v1 ./= v1_norm
    elseif v2_norm != 0.0
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
    elseif v1_norm != 0.0
        v1 .*= v1_norm
    elseif v2_norm != 0.0
        v2 .*= v2_norm
    end
end

        """
Replaces the subroutine MINRM
"""
function minimize_v_polynomial(v0::Array{Float64}, v1::Array{Float64},
                               v2::Array{Float64}, m::Int,
                               alpha_lower_bound::Float64,
                               alpha_upper_bound::Float64,
                               xmin::Float64, x::Number_wrapper{Float64},
                               px::Number_wrapper{Float64},
                               y::Number_wrapper{Float64},
                               py:::Number_wrapper{Float64})

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
    #minrm1(v0,v1,v2,m,sqnv0,sqnv1,sqnv2,scv0v1,scv0v2,scv1v2)

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
        #minrm2(sqnv1,sqnv2,scv0v1,scv0v2,scv1v2,p,q,delta,a1div3)
        if delta >= 0.0
            oner(q.value, delta.value, a1div3.value, x)
            y.value = x.value
            #goto 100
        end
        twor(p.value, q.value, delta.value, a1div3.value, x1, x2, x3)
        choose(x1.value, x2.value, x3.value, xmin, v0, v1, v2, m,
               x, y, py)
        #goto100
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

    x.value = clamp(x.value, alpha_lower_bound, alpha_upper_bound)
    px.value = fourth_degree_pol(v0 v1, v2, m, x.value)
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
    if abs(q) <= 2.0 * eps * sqd
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
                m::Int, root1::Number_wrapper{Float64},
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


"""
Replaces the function POL4
"""
function fourth_degree_pol(v0::Array{Float64}, v1::Array{Float64},
                           v2::Array{Float64}, m::Int, x::Float64)
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

"""
Replaces the function psi
"""
function psi(current_point::Array{Float64}, p::Array{Float64},
             current_point_dim::Int, alfk::Float64, next_point::Array{Float64},
             next_residuals::Array{Float64}, number_of_residuals::Int,
             residuals!::Function, next_constraints::Array{Float64},
             t::Int, l::Int, active_constraints::Array{Int},
             constraints!::Function, w::Array{Float64},
             ctrl::Number_wrapper{Int})
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
            psi = (0.5 * norm(next_residuals[1:number_of_residuals]) ^ 2
                   + constraints_merit(active_constraints, t,
            next_constraints, w, l))
        end
        if fctrl.value < -10 || hctrl.value < -10
            ctrl.value = min(fctrl.value, hctrl.value)
        end
        return psi
    end
    if fctrl.value == 1 && hctrl.value == 1
        psi = (0.5 * norm(next_residuals[1:number_of_residuals]) ^ 2
               + constraints_merit(active_constraints, t, next_constraints,
        w, l))
    else
        ctrl.value = -1
    end
    return psi
end


    
"""
The subroutine PERMIN can be replaced by
PERMIN(P, N) => p = 1:n
"""

"""
Replaces the subroutine HESSH
"""
function hessian_constraints(constraints!:Function, b:Array{Float64, 2},
                             leading_dim_b::Int, current_point::Array{Float64},
                             current_point_dim::Int, v::Array{Float64},
                             active_constraints::Array{Float64}, t::Int,
                             f1::Array{Float64}, f2::Array{Float64}, m::Int,
                             ier::Number_wrapper{Int})

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
                                   active_constraints::Array{Int}, t::Int,
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
                            f2::Array{Float64}, m::Int)
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
                          number_of_residuals::Int, current_point_dim::Int,
                          residuals!::Function, active_constraints::Array{Int},
                          number_of_active_constraints::Int,
                          penalty_weights::Array{Float64},
                          number_of_constraints::Int, constraints!::Function,
                          number_of_eval::Number_wrapper{Int},
                          exit::Number_wrapper{Int}, next_point::Array{Float64},
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
    elseif number_of_eval.value < 0
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


end
"""
Replaces the subroutine PREGN
"""

function gn_previous_step(s::Array{Float64}, sn::Float64, b::Array{Float64},
                          mindim::Int, prank::Int, dim::Number_wrapper{Int})
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
                               rabs::Float64, prank::Int, km1rnk::Int,
                               pgress::Float64, prelin::Float64, asprev::Float64,
                               alfkm1::Float64, dim::Number_wrapper{Int})
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
"""
function jacobian_forward_diff(current_point::Array{Float64},
                               current_point_dim::Int,
                               current_residuals::Array{Float64},
                               number_of_residuals::Int, residuals!::Function,
                               jacobian::Array{Float64, 2},
                               leading_dim_jacobian::Int, w1::Array{Float64},
                               user_stop::Number_wrapper{Int})
    delta = sqrt(eps(Float64))
    xtemp = 0.0
    ctrl = Number_wrapper{Int}(0)
    delta_j = 0.0
    for j = 1:current_point_dim
        xtemp = current_point[j]
        delta_j = max(abs(xtemp), 1.0) * delta
        current_point[j] = xtemp + delta_j
        ctrl.value = -1
        residuals!(current_point, current_point_dim, w1, number_of_residuals,
                   ctrl, jcobian, leading_dim_jacobian)
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
function lower_triangular_solve(n::Int, a::Array{Float64, 2}, b::Array{Float64})
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
function upper_triangular_solve(n::Int, a::Array{Float64, 2}, b::Array{Float64})
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
function d_minus_c1_times_x1(kp1::Int, kq::Int, d::Array{Float64},
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
function g_transpose_times_g!(g::Array{Float64, 2}, n::Int,
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
                                 number_of_parameters::Int, c::Array{Float64},
                                 leading_dim_c::Int, number_of_residuals::Int,
                                 rank_c2::Int,
                                 current_residuals::Array{Float64},
                                 p3::Array{Int}, d3::Array{Float64},
                                 estimated_lagrange_mult::Array{Float64},
                                 a::Array{Float64, 2}, leading_dim_a::Int,
                                 active_constraints::Array{Int},
                                 number_of_active_constraints::Int, rank_a::Int,
                                 d1::Array{Float64}, p1::Array{Int},
                                 p2::Array{Int}, d2::Array{Float64},
                                 b::Array{Float64},
                                 current_constraints::Array{Float64},
                                 number_of_constraints::Int,
                                 leading_dim_fmat::Int, pivot::Array{Float64},
                                 gmat::Array{Float64, 2}, leading_dim_gmat::Int,
                                 search_direction::Array{Float64},
                                 number_of_eval::Number_wrapper{Int},
                                 error::Number_wrapper{Int},
                                 fmat::Array{Float64, 2}, d::Array{Float64},
                                 v1::Array{Float64}, v2::Array{Float64})
    tp1 = rank_a + 1
    nmt = n - t
    nmr = n - rank_a
    for i = 1:number_of_residuals
        d[i] = -current_residuals[i]
    end
    p3utq3(p3, nmt, c[1,tp1], leading_dim_c, d3, nmr, d, number_of_residuals,
           1,v2))
    p3utq3(p3, nmt, c[1,tp1], leading_dim_c, d3, nmr, c, number_of_residuals,
           rank_a, v2)
    if rank_a != 3
        for i = 1:rank_a
            search_direction[i] = b[i]
        end
        if number_of_active_constraints == rank_a
            #lsolve(leading_dim_a, number_of_active_constraints, a,
            #search_direction)
        elseif number_of_active_constraints > rank_a
            #usolve(leading_dim_gmat, rank_a, gmat, search_direction)
        end
    end
    c2tc2(c[1, tp1], leading_dim_c, nmr, p3, nmt, v2)
    #hessf(residuals!, gmat, leading_dim_gmat, current_point,
    #number_of_parameters, current_residuals, v1, v2, number_of_residuals,
    #error)

    if error.value <= -10
        return
    end
    number_of_eval.value = 2 * number_of_parameters * (number_of_parameters + 1)

    if number_of_active_constraints != 0
        #hessh(constraints!, gmat, leading_dim_gmat, current_point,
        #number_of_parameters, v, active_constraints,
        #number_of_active_constraints, v1, v2, number_of_constraints, error)
        ecomp(p2, a, leading_dim_a, number_of_parameters, d1, pivot, rank_a,
              number_of_active_constraints, gmat, leading_dim_gmat)
    end
    w_plus_c(gmat[tp1, 1], c, nmr, number_of_parameters) 
    if rank_a != 0
        ycomp()
    end
    j = 0
    for i = 1:nmr
        j = rank_a + i
        search_direction[j] = d[i]
    end
    info = Number_wrapper{Int}(0)
    #dchdc()
    error.value = 0
    if nmr != info.value
        error.value = -3
        return
    end
    #dposl()
    if number_of_active_constraints != rank_a
        p_times_v(p2, rank_a, search_direction,  1)
    end
    if rank_a == 0
        return
    end
    for i = 1:rank_a
        j = rank_a - i + 1
        #householder_transform()
    end
end

"""
Replaces the subroutine P3UTQ3
"""
function p3utq3(p3::Array{Int}, nmt::Int, c2::Array{Float64 ,2},
                leading_dim_c::Int, d3::Array{Float64}, rank_c2::Int,
                c::Array{Float64, 2}, number_of_residuals::Int, rank_a::Int,
                v2::Array{Float64})
    if rank_c2 > 0
        for i = 1:rank_c2
            #householder_transform()
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
    #60
"""
Replaces the subroutine C2TC2
"""
function c2tc2(c2::Array{Float64, 2}, leading_dim_c2, rank_c2::Int,
               p3::Array{Int}, nmt::Int, v2::Array{Float64})
    k = 0
    if rank_c2 > 1
        for i = 2, rank_c2
            k = i - 1
            for j = 1:k
                c2[i, j] = 0.0
            end
        end
    end
    v_times_p_transpose(p3, nmt, c2, rank_c2)
    #jtrj()
end

"""
Replaces the subroutine ECOMP
"""
function ecomp(p2::Array{Int}, a::Array{Float64 , 2}, leading_dim_a::Int, n::Int,
               d1::Array{Float64}, pivot::Array{Float64}, rank_a::Int, t::Int,
               gmat::Array{Float64}, leading_dim_gmat)
    for i = 1:rank_a
        #householder_transform()
    end
    for i = 1:rank_a
        #householder_transform()
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
function w_plus_c(w::Array{Float64, 2}, c::Array{Float64, 2}, nmt::Int, n::Int)
    for i = 1:nmt
        for j = 1:n
            w[i,j] += c[i, j]
        end
    end
end

"""
Replaces the subroutine HSUM
"""
function constraints_merit(active_constraints::Array{Int}, t::Int,
                           h::Array{Float64}, w::Array{Float64}, l::Int)

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
function unscramble_array(active_constraints::Array{Int}, bnd::Int, l::Int,
                          p::Int)
    lm = = l - p
    if lmp <= 0
        return
    end
    j = 0
    for i = 1:lmp
        j = bnd + i
        if active_constraints[j] == -1
            active_constraints[j] = 0
        elseif active_constraints >= 2
            active_constraints[j] = 1
        end
    end
end


"""
Replaces the subroutine PREOBJ
"""
function preobj(c::Array{Float64, 2}, m::Int, rank_a::Int, dx::Array{Float64},
                f::Array{Float64}, t::Int, fc1dy1::Number_wrapper{Float64},
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
:

"""
Replaces the subroutine NZESTM
lm = lagrange multiplier
"""
function nonzero_first_order_lm(a::Array{Float64}, rank_a::Int,
                                number_of_active_constraints::Int,
                                b::Array{Float64}, v::Array{Float64},
                                w::Array{Float64}, u::Array{Float64})
    w[i] = copy(b[i])
    #lsolve(rank_a, a, w)
    solve_t_times_t!(a, rank_a, w, u, number_of_active_constraints,
                          Number_wrapper{Float64})
    v .+= u
end

    
    
"""
Replaces the subroutine LEAEST
"""
function lagrange_multipliers_estimate(a::Array{Float64, 2},
                                       number_of_active_constraints::Int,
                                       current_residuals::Array{Float64},
                                       number_of_residuals::Array{Float64},
                                       v1::Array{Float64},
                                       c::Array{Float64, 2},
                                       p1::Array{Int},
                                       scale::Int,scaling_matrix::Array{Float64},
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
function minimize_euclidean_norm(ctrl::Int, old_penalty_constants::Array{Float64},
                                 number_of_constraints::Int,
                                 positive_elements_l::Array{Int},
                                 number_of_pos_elements_l::Int,
                                 y::Array{Float64}, tau::Float64,
                                 new_penalty_constants::Array{Float64},
                                 working_area::Array{Float64})
    if number_of_pos_elements_l <= 0
        return
    end
    working_area = copy(old_penalty_constants)
    y_norm = norm(y[1:number_of_constants])
    y_norm_sq = y_norm ^ 2
    if y_norm != 0.0
        y[1:n] = 1.0: ./(y_norm * y[1:n])
    end
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
            if prod >= working_area [i]
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
