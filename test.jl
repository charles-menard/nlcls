include("nlcls.jl")

function constraints!(x, h)
    h[1] = 48.0 - x[1] ^ 2 - x[2] ^ 2 - x[3] ^ 2
    h[2] = x[1] + 4.5
    h[3] = x[2] + 4.5
    h[4] = x[3] + 5.0
    h[5] = -x[1] + 4.5
    h[6] = -x[2] + 4.5
    h[7] = -x[3] + 5.0
end

function residuals!(x, f)
    f[1] = x[1] - x[2]
    f[2] = (x[1] + x[2] - 10.0) / 3.0
    f[3] = x[3] - 5.0
end
function main()
    first_approx = [-5.0; 5.0; 0.0]
    
    number_of_residuals = 3
    number_of_equality_constraints = 0
    number_of_constraints = 7
    penalty_weights = Array{Float64}(undef, 7)
    current_residuals = Array{Float64}(undef, 10)
    current_constraints = Array{Float64}(undef, 7)
    active_constraints = Array{Int64}(undef, 10)
    
    res = easy_nlcls(first_approx, number_of_residuals, residuals!, constraints!,
               number_of_equality_constraints, penalty_weights, current_residuals,
               current_constraints, active_constraints)
    println(res)
    println("found point :")
    println(first_approx)
end
main()
