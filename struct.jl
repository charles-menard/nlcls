#This file contains the struct used to mimic the effect the different

import Base.==
"""
Replaces the common block PREC
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

struct Nlcls_scalars
    number_of_active_constraints::Int64
    exit::Int64
    number_of_iterations::Int64
    number_of_eval::Int64
    number_of_jac_eval::Int64
    number_of_hessian_eval::Int64
    number_of_linesearch_eval::Int64
    rank_a::Int64
    rank_ac::Int64
    objective_at_termination::Float64
    convergence_factor::Float64
end

#######
# The wrapper below is used to simulate passing parameters by reference

#######
mutable struct Number_wrapper{T<:Number}
    value::T
end
Base.isless(a::Number_wrapper{T}, b::T) where T = a.value < b
Base.isless(a::T, b::Number_wrapper{T}) where T = a < b.value
function Base.isless(a::Number_wrapper{T}, b::Number_wrapper{T}) where T
    a.value < b.value
end
==(a::Number_wrapper{T}, b::T) where T = a.value == b
==(a::T, b::Number_wrapper{T}) where T = a == b.value
==(a::Number_wrapper{T}, b::Number_wrapper{T}) where T = a.value == b.value
