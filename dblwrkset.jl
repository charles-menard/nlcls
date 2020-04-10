"""
Replaces the subroutine WRKSET
"""
function update_active_constraints(a::Array{Float64, 2}, leading_dim_a::Int64,
                                   (number_of_active_constraints::
                                    Number_wrapper{Int64}),
                                   number_of_equality_constraints::Int64,
                                   number_of_parameters::Int64,
                                   gradient_objective::Array{Float64},
                                   minus_active_constraints::Array{Float64},
                                   tau::Float64, leading_dim_fmat::Int64,
                                   scale::Int64, iteration_number::Int64,
                                   scaling_matrix::Array{Float64},
                                   active_Constraints::Array{Int64},
                                   min_l_n::Int64,
                                   inactive_constraints::Array{Int64},
                                   (number_of_inactive_constraints::
                                    Number_wrapper{Int64}),
                                   current_constraints::Array{Float64},
                                   gn_steplength_norm::Number_wrapper{Float64},
                                   p4::Array{Int64},
                                   jac_residuals::Array{Float64, 2},
                                   leading_dim_jac_residuals::Int64,
                                   number_of_residuals::Int64,
                                   current_residuals::Array{Float64},
                                   leading_dim_gmat::Int64,
                                   current_point::Array{Float64},
                                   constraints!:Function, residuals!::Function,
                                   number_of_eval::Number_wrapper{Int64},
                                   number_of_jac_eval::Number_wrapper{Int64},
                                   p2::Array{Int64}, p3::Array{Int64},
                                   gn_search_direction::Array{Float64},
                                   v1::Array{Float64}, d2::Array{Float64},
                                   d3::Array{Float64},
                                   rank_c2::Number_wrapper{Int64},
                                   d1_norm::Number_wrapper{Float64},
                                   d_norm::Number_wrapper{Float64},
                                   b1_norm::Number_wrapper{Float64},
                                   d::Array{Float64}, gmat::Array{Float64, 2},
                                   p1::Array{Int64}, v::Array{Float64},
                                   d1::Array{Float64}, fmat::Array{Float64},
                                   rank_a::Number_wrapper{Int64},
                                   gres::Number_wrapper{Float64},
                                   number_of_householder::Number_wrapper{Int64},
                                   (deleted_constraints_plus2::
                                    Number_wrapper{Int64}),
                                   deleted_constraints::Number_wrapper{Bool},
                                   pivot::Array{Float64}, v2::Array{Float64},
                                   s::Array{Float64}, u::Array{Float64},
                                   ltp::Last_two_points,
                                   restart_steps::Restart_steps,)

    j = Number_wrapper(0)
    user_stop = Number_wrapper{Int64}
    number_of_constraints = (number_of_active_constraints.value
                                           + number_of_inactive_constraints.value)
    deleted_constraints_plus2.value = 1
    tol = sqrt(Float64(number_of_active_constraints.value)) * tau
    del = false

    estimated_lagrange_mult(deleted_constraints_plus2, a, leading_dim_a,
                            number_of_active_constraints.value,
                            number_of_parameters, gradient_objective,
                            minus_active_constraints, (j.value), tol, d1, fmat
                            leading_dim_fmat, pivot, p1, scale, scaling_matrix,
                            v, rank_a, gres, s, u, v2)
    number_of_householder.value = rank_a.value
    noeq = Number_wrapper{Int64}
    if (number_of_residuals - number_of_active_constraints.value
        > number_of_parameters)
        sign_ch(deleted_constraints_plus2.value, p1, v,
                number_of_active_constraints.value, min_l_n, ltp.d1km1,
                gn_steplength_norm, iteration_number_ scale, scaling_matrix,
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
        estimated_lagrange_mult(deleted_constraints_plus2, a, leading_dim_a,
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
        estimated_lagrange_mult(deleted_constraints_plus2, a, leading_dim_a,
                                number_of_active_constraints.value,
                                number_of_parameters, minus_current_constraints,
                                j.value, tol, d1, fmat, leading_dim_fmat,
                                pivot, p1, scale, scaling_matrix, v, rank_a,
                                gres, s, u, v2)
    end
    gn_search(deleted_constraints_plus2.value, a, leading_dim_a,
              number_of_active_constraints.value, number_of_parameters, d1, p1,
              rank_a.value, number_of_householder.value, minus_active_constraints,
              fmat, leading_dim_fmat, jac_residuals, leading_dim_jac_residuals,
              number_of_residuals, current_residuals, pivot, tau, leading_dim_gmat,
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
            min_l_n, d1_norm, d_norm.value, iteration_number, scale,
            scaling_matrix, gres.value, current_constraints,
            number_of_equality_constraints, number_of_constraints, j, noeq,
            s, v2)
    if noeq.value == 0
        return
    end
    del = true
    reorder(a, number_of_active_constraints, number_of_parameters,
            minus_current_constraints, j.value, noeq.value, active_constraints,
            inactive_constraints, number_of_inactive_constraints, p4, u, scale,
            scaling_matrix)
    estimated_lagrange_mult(deleted_constraints_plus2, a, leading_dim_a,
                            number_of_active_constraints.value,
                            number_of_parameters, minus_current_constraints,
                            j.value, tol, d1, fmat, leading_dim_fmat,
                            pivot, p1, scale, scaling_matrix, v, rank_a,
                            gres, s, u, v2)
    residuals!(current_point, number_of_parameters, current_residuals,
               number_of_residuals, user_stop, jac_residuals,
               leading_dim_jac_residuals)
    if user_stop.value == 0
        jacobian_forward_dif(current_point, number_of_parameters,
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
