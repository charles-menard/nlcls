#This file contains the struct used to mimic the effect the different
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

#######
# The wrapper below is used to simulate passing parameters by reference

#######
mutable struct Number_wrapper{T<:Number}
    value::T
end

