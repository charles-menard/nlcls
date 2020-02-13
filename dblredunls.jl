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


"""
Compute the jacobian of the constraints and the residuals and put them
respectively in `jac_constraints` and `jac_residuals`

"""

function new_point!(current_point::Array{Float64},
                    current_constraints::Array{Float64},
                    current_residuals::Array{Float64}, constraints!::Function,
                    residuals!::Function, number_of_eval::Int,
                    jac_constraints::Array{Float64, 2},
                    jac_residuals::Array{Float64}, user_stop::Int,
                    b::Array{Float64}, working_area::Array{Float64})

    ctrlc = 2
    residuals!(current_point, length(current_point), current_residuals,
               length(current_residuals), ctrlc, jac_residuals,
               size(jac_residuals)[1]))

    if ctrlc < -10 || user_stop < -10
        return
    end
    if ctrlc == 0
        jac_residuals = ForwardDiff.jacobian(residuals!, current_residuals,
        current_point)
        if user_stop < -10
            print_cons_res(current_constraints, current_residuals)
            return
        end
        number_of_eval += length(current_point)
    end
    if length(current_constraints) <= 0
        return
    end
    ctrla = 2

    constraints!(current_point, length(current_point), current_constraints,
    length(current_constraints), ctrla, jac_constraints,
    size(jac_constraints)[1])

    if ctrla < 10 || user_stop < -10
        return
    end
    if ctrlc == 0
        jac_residuals = ForwardDiff.jacobian(constraints!, current_constraints,
        current_point)

        if user_stop < -10
            return
        end
    end
    b = -h

end

function equal!(b::Array{Float64}, l::Int, a::Array{Float64,2},
    active_constraints::Array{Float64}, t::Int, p::Int, p4::Array{Int})
    if l > 0
        for i=1:l
            p4[i] = i
        end
    end
    if t <= 0 || t == p
        return
    end

    index::Int = 1
    ip::Int = 0
    ik::Int = 0
    for i=1:t
        index = active_constraints[i]
        ip = p4[index]
        if ip == i
            continue
        end
        for j=1:size(a)[2]
            a[i,j], a[ip, j] = a[ip, j], a[i, j]
        end

        b[i], b[ip]= b[ip], b[i]
        for j=1:t
            if i != p4[j]
                continue
            end
            ik = j
        end
    end

end

"""
C     SCALE THE SYSTEM  A*DX = B    IF SO INDICATED BY FORMING
C     A@D=DIAG*A      B@D=DIAG*B
"""
function scale_system!(scale::Int, jacobian::Array{Float64,2},
    neg_constraints::Array{Float64}, scaling_matrix::Array{Float64})
    if length(neg_constraints)== 0 || scale
        return
    end

    current_norm = 0.0
    for i=1:length(neg_constraints)
        row_norm = norm(jacobian[i,:]) #check for correctness
        if row_norm > current_norm
            #does it do something?
            current_norm = row_norm
        end
        scaling_matrix[i] = row_norm
        if scale == 0
            continue
        end

        if row_norm == 0.0
            row_norm == 1.0
        end
        for j=1:size(jacobian)[2]
            jacobian[i,j] /= row_norm
        end
        neg_constraints /= row_norm
        scaling_matrix[i] = 1.0/row_norm
    end
end

function gn_seach!(fmat_is_identity::Int, a::Array{Float64,2},
    d1::Array{Float64}, p_1::Array{Int}, rank_a::Int,
    number_of_householder::Int, b::Array{Float64}, fmat::Array{Float64,2},
    jac_residuals::Array{Float64,2}, residuals::Array{Float64},
    pivot::Array{Float64}, tau::Float64, scale::Int,
    diag::Array{Float64,2}, inactive_constraints::Array{Int},
    p4::Array{Float64,2}, p2::Array{Int}, p3::Array{Int},
    gn_direction::Array{Float64}, v1::Array{Float64}, d2::Array{Float64},
    d3::Array{Float64}, rank_c2::Int, d::Array{Float64},
    work_area_s::Array{Float64}, work_area_u::Array{Float64},
    gmat::Array{Float64,2})


    #@printf("In gn_search : rank_a, t = %i, %i", rank_a, length(p1))
    code = 1
    if length(p1) != rank_a
        code = -1
        #l_to_upper!()
    end

    cqhp2()
    kc2::Int = rank_a + 1
    nmp::Int = size(a)[2] - rank_a
    tol = sqrt(nmp)*tau#
    #c2_to_upper!()
    #subdir!()
end

function min_max_langrange_mult(estimated_lagrange_mult::Array{Float64},
    number_of_eq::Int, scale::Int,
    scaling_mat::Array{Float64},
    smallest_ineq_mult::Float64, max_mult::Float64 )
    if number_of_eq = length(estimated_lagrange_mult)
        return
    end
    tol::Float64 = sqrt(eps(Float64))
    smallest_ineq_mult = 1e6
    max_mult = 0.0
    current_scale = 0.0

    for i in 1:length(estimated_lagrange_mult)
        current_abs = abs(estimated_lagrange_mult[i])
        if current_abs > max_mult
            max_mult = current_abs
        elseif i <= number_of_eq
            continue
        end
        current_scale = scaling_mat[i]
        if scale != 0
            current_scale =1.0 / scaling_matrix[i]

        elseif -tol < estimated_lagrange_mult*current_scale
            continue

        elseif estimated_lagrange_mult < smallest_ineq_mult
            smallest_ineq_mult = estimated_lagrange_mult[i]
        end
    end
end
"""
replaces subroutine analys
"""
function check_last_step(iteration_number::Int, restart::Bool, code::Int,
    sq_sum_residuals::Float64, d1_norm::Float64, d_norm::Float64,
    c::Array{Float64, 2}, rank_c2::Int, d::Array{Float64},
    current_residuals::Array{Float64}, p3::Array{Float64},d2::Array{Float64},
    active_constraints::Array{Int}, estimated_lagrange_mult::Array{Float64},
    inactive_constraints::Array{Int}, p4::Array{Int}, deleted_constraints::Int,
    a::Array{Float64, 2}, rank_a::Int, norm_b1::Float64,
    sq_sum_constraints::Float64, number_of_householder::Int, d1::Array{Float64},
    p1::Array{Int}, d2::Array{Float64}, p2::Array{Int}, b::Array{Float64},
    h::Array{Float64}, f_mat::Array{Float64, 2}, pivot::Array{Float64},
    g_mat::Array{Float64, 2}, residuals!::Function, constraints!::Function,
    current_point::Array{Float64}, hessian::Bool, constraint_added:Bool,
    constraint_deleted::Bool, scale::Int, scaling_mat::Array{Float64, 2},
    gn_direction::Array{Float64}, gn_direction_norm::Array{Float64},
    v1::Array{Float64}, relative_termination_tol::Float64, error::Int,
    number_of_eval::Int, d1_plus_b1_norm::Float64, dim_a::Int, dim_c2::Int,
    v2::Array{Float64}, work_area::Array{Float64}, ltp::Last_two_points,
    restart_steps::Restart_steps)

    if !restart
        ind::Int = ltp.rkckm1 + ltp.tm1 - length(estimated_lagrange_mult)
        ind -= 1
        ltp.d1km2 = norm(d[1:ind])
    end
    #gndchk
    number_of_eval = 0
    code = ind
    error = 0
    if ind == 1
        dim_a = rank_a
        dim_c2 = rank_c2
        return
    end
    if ind != 2
        #subspc()
        restart_steps.nrrest += 1
        #subdir()
        if dim_a == rank_a && dim_c2 == rank_c2
            code = 1
            dx_norm = norm(dx)
        end

    end
    if !hessian
        error = -4

    end
    if ltp.kodkm1 != 2
        restart_steps.nrrest = 0
    end
    restart_steps.nrrest += 1
    #newton()
    dim_a = -(length(estimated_lagrange_mult))
    dim_c2 = -(size(a)[2]) + length(estimated_lagrange_mult)
    if restart_steps > 5
        error = -9
    end
    gn_direction_norm = norm(gn_direction)
end
"""
Replaces subroutine SUBDIR
"""
function sub_search_direction(deleted_constraints::Int, a::Array{Float64, 2},
    d1::Array{Float64}, p1::Array{Int}, dim_a::Int,
    rank_a::Int, number_of_householder::Int,
    b::Array{Float64}, f_mat::Array{Float64, 2},
    c::Array{Float64, 2},
    current_residuals::Array{Float64},
    pivot::Array{Float64}, g_mat::Array{Float64,2},
    d2::Array{Float64}, p2::Array{Int},
    p3::Array{Int}, d3::Array{Float64}, dim_c2::Int,
    inactive_constraints::Array{Int}, p4::Array{Int},
    rank_c2::Int, code::Int, scale::Int,
    scaling_matrix::Array{Float64}, d::Array{Float64},
    search_direction::Array{Float64},
    v1::Array{Float64}, d1_norm::float64,
    search_direction_norm::Float64, b1_norm::Float64,
    work_area::Array{Float64})
    #@printf("In sub_search_direction: code= %d", code)
    b1_norm = 0.0
    if t > 0
        search_direction = copy(b)
        b1_norm = norm(search_direction[1:dim_a])
        #@printf("b1_norm= %f", b1_norm)
        #adx()
        if code == 1
            #lsolve()
        else
            #usolve()
        end
    end
    d = -(copy(estimated_lagrange_mult))
    #ycomp()
    search_direction_norm = norm(d)
    #@printf("search_direction_norm= %f", search_direction_norm)
    #@printf("rank_c2= %d", rank_c2)
    k = 0
    if rank_c2 > 0
        for i=1:rank_c2
            k = rank_a + i
            #h12per()
        end
    end
    d1_norm = d[1:dim_c2]
    k = rank_a + 1
    nmt = size(a)[2] - rank_a
    #cdx()
    if nmt > 0
        search_direction[1+rank_a:nmt+rank_a] = copy(d[1:nmt])
        #usolve()
        if dim_c2 != nmt
            k = dim_c2 + 1
            search_direction[k+rank_a:nmt+rank_a] = zeros(nmt-k+1)
        end
    end
    #pv()
    if code != 1
        #pv()
    end
    if deleted_constraints > 2 || (deleted_constraints !== 3 || length(b) != 0)
        #hxcomp()
    end

    no::Int = number_of_householder - length(b)
    irow::Int = 0
    i2::Int = 0
    i3::Int = 0
    no_elem::Int = 0
    ait_search_direction::Float64 = 0.0
    if no > 0
        irow = length(b) + 1
        i2 = inactive_constraints[end]
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
    for i=1:number_of_householder
        k = number_of_householder - i + 1
        #h12per()
    end
end

function upper_bound_steplength(a::Array{Float64, 2}, number_of_householder::Int,
    number_of_working_cons::Int, current_constraints::Array{Float64},
    inactive_constraints::Array{Int}, p4::Array{Int}, v1::Array{Float64},
    number_of_residuals::Int, search_direction::Array{Float64},
    upper_bound::Float64, upper_bound_cons_index::Int)

    upper_bound = 1.0e6
    upper_bound_cons_index = 0
    q = length(inactive_constraints)
    n = length(search_direction)
    t = number_of_working_constraints
    mpt = number_of_residuals  + t
    l = q - number_of_householder + t
    k = 0
    ip = 0
    ir = 0
    ait_search_direction = 0.0
    alf = 0.0
    change_ait_search_dir = true
    if q > 0
        for i=1:q
            k = inactive_constraints[i]
            ip = mpt + i
            ir = p4[k]
            if i <= l
                ait_search_direction = 0.0
                for j=1:n
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
            upper_bound = alf
            ind = k
        end
    end
    upper_bound = min(upperbound, 3.0)
end

"""
Replaces subroutine EVADD
"""
function move_violated_constraints(current_constraints::Array{Float64},
    active_constraints::Array{Int}, min_l_n::Int, number_of_equality::Int,
    inactive_constraints::Array{Int}, ind::Int, iteration_number::Int,
    constraint_added::Bool)

    tol = sqrt(eps(Float64))
    delta = 0.1
    i = 1
    k = 0
    kind = 0
    jj = 0
    kk = 0
    max_constraint = 0.0
    q = length(inactive_constraints)
    t = length(active_constraints)
    while i <= q
        k = inactive_constraints[i]
        if (inactive_constraints[k] >= tol &&
            (k == ind || current_constraints[k] >= delta))
            kind = 0
            if t >= min_l_n
                kk = 0
                kind = 1
                max_constraint, kk = findmax(current_constraint[active_constraint])
                if max_constraint < 0
                    max_constraint = 100.0
                    for j=1:t
                        jj = active_constraints[j]
                        if abs(current_constraint[jj]) < max_constraint
                            max_constraint = abs(current_constraint)
                            kk = j
                        end
                    end
                end
                #@printf("Deleted constraint current_constraint[kk]= %d", kk)
                #delete()
            end
            #@printf("Added constraint: %d", k)
            #addit()
            j = min_l_n + k - number_of_equality
            if active_constraints[j] == -1
                active_constraints[j] = iteration_number
            elseif active_constraints[j] == 0
                active_constraints[j] = 1
            end
            constraint_added = true
            if kind == 1
                i = q + 1
            end
        end
        i += 1
    end
end
"""
Replaces the subroutine SCALV

    vector = 1 ./ (factor*vector)
"""


"""
Replaces the subroutine EVREST
"""
function evaluation_restart(current_point::Array{Float64},
    previous_point::Array{Float64}, iteration_number::Int, residuals!::Function,
    number_of_eval::Int, current_residuals::Array{Float64}, d1_norm::Float64,
    d_norm::Float64, sq_sum_previous_residuals::Float64, dim_c2::Int, code::Int,
    search_direction_norm::Float64, d1_plus_d1_norm::Float64,
    current_steplength::Float64, lower_bound_steplength::Float64,
    active_constraints::Array{Int}, current_constraints::Array{Float64},
    constraints!::Function, b1_norm::Float64, sq_sum_constraints::Float64,
    dim_a::Int, error::Int, restart::Bool, ltp::Last_two_steps,
    restart_steps::Restart_steps, negdir::Negdir)

    restart = false
    if (restart_steps.lattry != 0 || restart_steps.bestpg <= 0.0 ||
        (-1 != error && error > -3))
        if current_steplength <= lower_bound_steplength
            restart = true
            #goto50
        end

    end
    if restart != true
        iteration_number += 1
        if code != 2
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
        ltp.fsqkm1 = sq_sum_residuals
        ltp.b1km1 = b1_norm
        ltp.hsqkm1 = sq_sum_constraints
        ltp.dxnkm1 = serch_direction_norm
        ltp.alfkm1 = current_steplength
        ltp.rkakm1 = dim_a
        ltp.rkckm1 = dim_c2
        ltp.tkm1 = length(active_constraints)
        ltp.kodkm1 = code

        skip_count = false
        if -1 != error
            if error < 0                                                    
                return                                                      
            end                                                              
            sq_sum_constraints = sum(
            current_constraints[active_constraints].^2)
                ltp.hsqkm2 = sq_sum_constraints
            sq_sum_residuals = norm(current_residuals)^2
        elseif -1 == error
            skip_count = true
        end
    end
    #50
    if !skip_count
        restart_steps.nrrest += 1
        ltp.rkakm1 = dim_a
        rkckm1 = dim_c2
        if iteration_number == 0
            ltp.d1km1 = d1_norm
            ltp.b1km1 = b1_norm
        end
    end
    #55
    latest_point = copy(previous_point)
    if (abs(code)) == 2
        error =-5
    end
    ctrl = -1
    dummy = 0.0
    residuals!(current_point, length(current_point), current_residuals,
               length(current_residual), ctrl, dummy, 1)
    if ctrl < -10
        error = ctrl
    end
    ctrl = -1
    constraints!(current_point, length(current_point), current_constraints,
    length(current_constraints), ctrl, dummy, 1)
    if ctrl < -10
        error = ctrl
    end
    current_steplength = ltp.alfkm1
    number_of_eval += 1
    return
end

function output(print_info::Int, iteration_number::Int, output_file_number::Int,
    product_norm::Float64, penalty_weight::Array{Float64},
    active_constraints::Array{Int}, estimated_convergence_factor::Float64)

end
