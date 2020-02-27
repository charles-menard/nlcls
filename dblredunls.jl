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

mutable struct Linc1_result
    eta::Float64
    tau::Float64
    gamma::Float64
    alfmax::Float64
    alfmin::Float64
    alfk::Float64
    pmax::Float64
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
    active_constraints::Array{Int}, estimated_convergence_factor::Float64,
    ltp::Last_two_points, restart_steps::Restart_steps)
    if print_info <= 0
        return
    end
    iteration_number_minus = iteration_number - 1
    if ((iteration_number_minus / print_info * print_info)
        != iteration_number_minus)
        return
    end
    estimated_convergence_factor = 0.0
    if iteration_number_minus != 0
        estimated_convergence_factor = ltp.betkm1 / ltp.betkm2
    end
    max_weight = 0.0
    j = 0
    if ltp.tkm1 > 0
        max_weight = maximum(penalty_weight[active_constraints[1:ltp.tkm1]])
    end
    #TODO understand FORMAT string
    if iteration_number_minus = 0
        @printf()
    end
    if ltp.tkm1 > 0
        @printf(, iteration_number_minus, ltq.fsqkm1, ltp.hsqkm1, product_norm,
        ltp.dxnkm1, ltp.rkakm1, ltp.rkckm1, ltp.alfkm1,
        estimated_convergence_factor, max_weight, ltp.prelin, ltp.pgress, )
    end
    if ltp.tkm1 <= 0
        @printf()
    end
    return
end

"""
Replaces the subroutine ADX
"""

function search_direction_product(code::Int, g_mat::Array{Float64, 2},
    rank_a::Int, number_of_non_zero_in_b::Int, p1::Array{Int},
    d2::Array{Float64}, b::Array{Float64}, number_of_householder::Int,
    scale::Int, scaling_matrix::Array{Float64}, product::Array{Float64},
    work_area::Array{Float64})
    k = 0
    if dim_a != length(b)
        k = dim_a + 1
        for e in b[k:end]
            e = 0.0
        end
    end
    product = copy(b)
    if code != 1
        for i=1:rank_a
            k = rank_a + 1 - i
            #h12per()
        end  
    end
    if length(b) < number_of_householder
        #pspecf()
        product = work_area[1:t]
    else
        #pv()
    end
    if scale <= 0
        return
    end
    product = product ./ scaling_matrix
    return
end

"""
Replaces the subroutine LTOUP

"""
function l_to_upper_triangular(a::Array{Float64, 2}, rank_a::Int,
    b::Array{Float64}, p2::Array{Int}, g_mat::Array{Float64,2},
    d2::Array{Float64})

    cmax = 0
    collng = 0.0 
    p2 = 1:rank_a
    for i=1:length(b)
        for j=1:rank_a
            if i < j
                g_mat[i, j] = 0.0
            else
                g_mat[i, j] = a[i, j]
            end
        end
    end
    for i=1:rank_a
        #colmax()
        p2[i] = cmax
        #prmcol()
        #h12per()
        #h12per()
    end
    return

end

"""
from here i included the parameters for the size of unknown arrays
"""
"""
Replaces the subroutine ATOLOW
"""
function a_to_lower_triangular(number_of_rows_a::Int, length_g::Int,
    a::Array{Float64, 2}, leading_dim_a::Int, tol::Float64,
    current_gradient::Array{Float64}, p1::Array{Int}, pseudo_rank_a::Int,
                               d1::Array{Float64})
    pseudo_rank_a = number_of_rows_a
    if number_of_rows_a == 0
        return
    end
    ldiag = min(number_of_rows_a, length_g)
    p1[1:ldiag] = 1:ldiag
    krank = 0
    imax = 0
    rmax= 0.0
    for i = 1:ldiag
        krank = i
        #TI
        if rmax >= tol
            break
        end
        p1[i] = imax
        #prmrow()
        #h12per()
        #h12per()
        krank = i + 1
    end
    pseudo_rank_a = krank - 1
    return
end

"""
Replaces subroutine C2TOUP
"""

function c2_to_upper_trianguar(number_of_rows_c2::Int, number_of_col_c2::Int,
    c2::Array{Float64}, leading_dim_c2::Int, tol::Float64, p3::Array{Int},
    pseudo_rank_c2::Int, d3::Array{Float64})
    pseudo_rank_c2 = min(number_of_rows_c2, number_of_col_c2)
    if number_of_col_c2 == 0 || number_of_rows_c2 == 0
        return
    end
    p3 = 1:number_of_col_c2
    ldiag = pseudo_rank_c2
    for k = 1:ldiag
        #colmax()
        #prmcol()
        #h12per()
    end
    krank = 0
    u_11 = abs(c2[1, 1])
    for k = 1:ldiag
        if abs(c2[k, k]) <= tol*u_11
            break
        end
    end
    pseudo_rank_c2 = krank

    sum = 0.0
    for i = 1:number_of_rows_c
        for j = 1:number_of_col_c1
            sum += c1[i,j]*b1[k]
        end
        product[i] += sum
    end
end

"""
Replaces subroutine ROWMAX
"""
function max_partial_row_norm!(m::Int, n::Int, a::Array{Float64, 2},
    starting_col::Int,starting_row::Int, max_row_index::Int,
    max_row_norm::Float64)

    max_row_norm = -1.0
    for i = starting_row:m
        row_norm = norm(a[i, starting_col:n])
        if row_norm > max_row_norm
            max_row_norm = row_norm
            max_row_index = i
        end
    end
end

"""
Replaces de subroutine COLMAX
"""
function max_partial_row_norm!(m::Int, n::Int, a::Array{Float64, 2},
    starting_col::Int,starting_row::Int, max_col_index::Int,
    max_col_norm::Float64)

    max_col_norm = -1.0
    for j = starting_col:n
        col_norm = norm(a[starting_row:m, j])
        if col_norm > max_col_norm
            max_col_norm = col_norm
            max_col_index = j
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
function v_times_p(p::Array{Int}, v::Array{Float64, 2},
    m::Int, n::Int)
    if m <= 0 || n <= 0
        return
    end
    for i = 1:n
        permute_columns(a=v, m=m, col1=i, col2=p[i])
    end
end

"""
Replaces the routine H12PER
"""
function householder_transform(mode::Int, pivot_index::Int, l1::Int, m::Int,
    pivot_vector::Array{Float64, 2}, pivot_vect_number_rows::Int,
    up::Float64, c::Array{Float64}, inc_elements_c::Int,
    inc_vectors_c::Int, number_of_vectors_c::Int, pivot::Float64)

    if 0 <= pivot_index || pivot_index >= l1 || l1 < m
        return
    end
    cl = abs(pivot)
    if mode != 2
        for j = l1:m
            cl = max(abs(pivot_vector[1, j]), cl)
        end
        if cl <= 0
            return
        end
        cl_inverse = 1.0 / cl
        sm = (pivot * cl_inverse) ^ 2
        for j = l1:m
            sm += (pivot_vector[1, j] * cl_inverse) ^ 2
        end
        sm1 = sm
        cl *= sqrt(sm1)
        cl *= -1
        up = pivot - cl
        pivot = cl
    end
    #60
    if cl <= 0 && mode == 2
        return
    end
    if number_of_vectors_c <= 0
        return
    end
    b = up * pivot
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
        sm = c[i2] * up
        for i = l1:m
            sm+= c[i3] * u[1,i]
            i3 += inc_elements_c
        end
        if sm == 0
            continue
        end
        sm *= b
        c[i2] += sm * up
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
        work_area = copy(c[i, 1:n])
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
    s = copy(x[1:n])
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

function apply_rotation(c::Float64, s::Float64, x::Float64, y::Float64)
    xt = c * x + s * y
    y = s * x - c * y
    x = xt
end

"""
Replaces subroutine PSPECF
"""
function permute(n::Int, p::Array{Int}, w::Array{Float64}, f::Array{Float64})

    if n <= 0
        return
    end
    f = w[p]
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
                           m::Int, n::Int, t::Int, pseudo_rank_a::Int, pivot::Array{Float64}, na::Int,
                           a::Array{Float64}, leading_dim_a::Int, d1::Array{Float64},
                           h::Array{Float64, 2}, leading_dim_h::Int, p2::Array{Int}, v::Array{Float64})

    if na == 0
        return
    end
    for i = 1:na
        householder_transform(mode=2, pivot_index=i, l1=i+1, m=n,
        pivot_vector=a[i, 1], pivot_vect_number_rows=leading_dim_a,
        up=d1[i], c=c, inc_elements_c=leading_dim_c,
        inc_vectors_c=1, number_of_vectors_c=m, pivot=pivot[i]) 
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
                 h::Array{Float64}, kp::Int, l::Int, j::Int, s::Int,
                 p2::Array{Float64}, working_area::Array{Float64})
    const ival = 4
    const delta = 10.0
    const tau = 0.5
    s = 0
    if kp == t
        return
    end
    e = 0.0
    sqrt_rel_prec = sqrt(eps(Float64))
    kp1 = kp + 1
    current_el = 0.0
    k = 0
    for i = kp1:t
        k = active_constraints[i]
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
        s = i
    end
    if gres > -e * delta
      return
    end
    k = active_constraints[s]
    i = bnd + k - kp
    if (active_constraints[i] != 1 &&
        iteration_number - active_constraints[i] < ival &&
        betkm1 > tau * gnd_norm)
        return
    end
    active_constraints[i] = -1
    if betkm1 <= tau * gnd_norm
        #unscr(active_constraints, bnd, l, kp)
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
    j = p1[s]
    tm1 = t - 1
    if tm1 < 1
        return
    end
    for i = 1:t
        if i < j
            p2[i] = i + 0.1
        elseif i == j
            p2[i] = 0.0
        else
            p2[i] = i - 1 + 0.1
        end
    end
    permute(n=t, p=p1, w=p2, f=working_area)
    for i = 1:t
        p1[i] = int(working_area[i])
        if p1[i] == 0
            k = i
        end
    end
    if k == t
        return
    end
    for i = k:tm1
        p1[i] = p1[i + 1]
    end
end
"""
Replaces subroutine REORD
"""
function reorder(a::Array{Float64, 2}, number_of_active_constraints::Int,
                 number_of_variables::Int, bv::Array{Float64},
                 row_of_l_to_delete::Int, s::Int, active_constraints::Array{Int},
                 inactive_constraints::Int, number_of_inactive_constraints::Int,
                 p4::Array{Int}, working_area::Array{Float64},scale::Int,
                 scaling_matrix::Array{Float64})

    tm1 = number_of_active_constraints - 1
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
            a[number_of_active_constraints, i] = working_area[i]
        end
    end
    if scale!= 0 && s != number_of_active_constraints
        temp = scaling_matrix[s]
        for i = s:tm1
            scaling_matrix[i] = scaling_matrix[i+1]
        end
        scaling_matrix[number_of_active_constraints] = temp
    end
    k = active_constraints[s]
    p4[k] = row_of_l_to_delete
    delete_constraint(active_constraints, inactive_constraints,
                       number_of_inactive_constraints,
                       number_of_active_constraints, s)

end
"""
Replaces subroutine ATSOLV
"""
function solve_lower_triangular!(a::Array{Float64, 2}, t::Int, b::Array{Float64},
                                 x::Array{Float64}, residue::Float64)
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
    residue = norm(b[t+1:n])
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
            gs *= jacobian[j, i] * residuals[j]
        end
        gradient[i] = gs
    end
end

"""
Replaces the subroutine QUAMIN
"""
function minimize_quadratic_function!(x1::Float64, y1::Float64, x2::Float64,
                                     y2::Float64, x3::Float64, y3::Float64,
                                     min_of_f::Float64)
    d1 = y3 - y1
    d2 = y2 - y1
    s = (x2 - x1) * (x2 - x1) * d1 - (x3 - x1) * (x3 - x1) * d2
    q = 2.0 * ((x2 - x1) * d2 - (x3 - x1) * d1)
    min_of_f = x - s / q
end

"""
Replaces the subroutine DELETE
"""
function delete_constraint(active_constraints::Array{Int},
                           inactive_constraints::Array{Int},
                           number_of_inactive_constraints::Int,
                           number_of_active_constraints::Int,
                           constraint_to_delete::Int)
    number_of_inactive_constraints += 1
    inactive_constraints[number_of_active_constraints] = (
        active_constraints[constraint_to_delete])
    for i = constraint_to_delete:number_of_active_constraints
        active_constraints[i] = active_constraints[i + 1]
    end
    number_of_active_constraints -= 1
end

"""
Replaces the subroutine ADDIT
"""
function add_constraint!(active_constraints::Array{Int},
                        inactive_constraints::Array{Int},
                        number_of_active_constraints::Int,
                        number_of_inactive_constraints::Int,
                        constraint_to_add:int)
    number_of_active_constraints += 1
    active_constraints[number_of_active_constraints] = (
        inactive_constraints[constraint_to_add])
    for i = constraint_to_add:number_of_inactive_constraints
        inactive_constraints[i] = inactive_constraints[i+1]
    end
    number_of_inactive_constraints -= 1
end

"""
Replaces the subroutine LINEC
the non-array parameters that are changed are returned in a struct
It may be that some parameters of steplength! are useless due to this change.
"""
function steplength!(current_point::Array{Float64},
                     search_direction::Array{Float64}, v1::Array{Float64},
                     current_f_value::Array{Float64}, number_of_residuals::Int,
                     current_point_length::Int, steplength::Float64,
                     psi_at_zero::Float64, derivative_psi_at_zero::Float64,
                     steplength_lower_bound::Float64, residuals!::Function,
                     constraints!::Function, current_constraints::Array{Float64},
                     active_constraints::Array{Int},
                     number_of_active_constraints::Int,
                     inactive_constraints::Array{Int},
                     number_of_inactive_constraints::Int,
                     number_of_constraints::Int,
                     w::Array{Float64}, steplength_upper_bound::Float64,
                     psi_at_next_point::Float64, x_diff_norm::Float64,
                     exit::Int, number_of_eval::Int,fnew::yArray{Float64},
                     hnew::Array{Float64}, v2::Array{Float64}, g::Array{Float64})

    struct Result 
        alpha::Float64
        psi_at_alpha::Float64
        x_diff_norm::Float64
        exit::Int
        number_of_eval::Int
    end
    
    k = 0
    x_diff_norm = 0.0
    psikm1 = psi_at_zero
    mpt = number_of_residuals + number_of_active_constraints
    mpl = number_of_residuals + number_of_constraints
    #Replaces the subroutine linc1
    eta = 0.3
    tau = 0.25
    gamma = 0.4
    alfmax = steplength_upper_bound
    alfmin = steplength_lower_bound
    alfk = min(steplength, steplength_upper_bound)
    pmax = maximum(abs, search_direction)
    #end
    
    exit = 0
    alfkm1 = alfk
    ctrl = 1
    psik = psi(current_point, search_direction, current_point_length, param.alfk,
               g, fnew, number_of_residuals, residuals!, hnew,
               number_of_active_constraints, number_of_constraints,
               active_constraints, constraints!, w, ctrl)
    k += 1
    if ctrl == -1
        exit = -3
    end
    if exit <= 0
        #goto 1020
    end
    diff = psi_at_zero - psik
    #linc2()
    if diff >= 0.0
        xmin = linc1_res.alfk
    else
        xmin = 0.0
    end
    #minrm()
    if (alkp1 != linc1_res.beta && pk > pbeta &&
        linc1_res.beta <= linc1_res.alfk)
        alfkp1 = 0.0
        psikm1 = psi_at_zero
    end
    #update()
    

end
"""
Replaces the subroutine CONCAT
modifies f
"""
function concat!(f::Array{Float64}, m::Int, h::Array{Float64},
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
Modifies f, fnew, v1, v2
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
function reduce()
"""
Replaces the subroutine LINC1
Set some constants for the subroutine compute_steplength
"""
