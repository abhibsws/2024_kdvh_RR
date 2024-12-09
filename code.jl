# Install packages
import Pkg
if v"1.10" <= VERSION < v"1.12-"
    # We provide Manifest files for perfect reproducibility
    # for Julia 1.10 and 1.11
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
else
    # We try to install reasonable versions of all packages
    # on other versions of Julia
    Pkg.activate(@__DIR__)
    Pkg.resolve()
end

# Load packages
using LinearAlgebra: UniformScaling, I, diag, diagind, mul!, ldiv!, lu, lu!, norm
using SparseArrays: sparse, issparse, dropzeros!

using DelimitedFiles: readdlm
using Interpolations: CubicSplineInterpolation, LinearInterpolation, Periodic

using SummationByPartsOperators

using LaTeXStrings
using Plots: Plots, plot, plot!, scatter, scatter!, savefig

using PrettyTables: PrettyTables, pretty_table, ft_printf


#####################################################################
# Utility functions
const FIGDIR = joinpath(dirname(@__DIR__), "img")
!isdir(FIGDIR) && mkdir(FIGDIR)

function plot_kwargs()
    fontsizes = (
      xtickfontsize = 14, ytickfontsize = 14,
      xguidefontsize = 16, yguidefontsize = 16,
      legendfontsize = 14)

    (; linewidth = 3, gridlinewidth = 2,
       markersize = 8, markerstrokewidth = 4,
       fontsizes...)
end


function compute_eoc(Ns, errors)
    eoc = similar(errors)
    eoc[begin] = NaN # no EOC defined for the first grid
    for idx in Iterators.drop(eachindex(errors, Ns, eoc), 1)
        eoc[idx] = -( log(errors[idx] / errors[idx - 1]) /
                      log(Ns[idx] / Ns[idx - 1]) )
    end
    return eoc
end


#####################################################################
# High-level interface of the equations and IMEX ode solver

rhs_full!(du, u, parameters, t) = rhs_full!(du, u, parameters.equation, parameters, t)
rhs_stiff!(du, u, parameters, t) = rhs_stiff!(du, u, parameters.equation, parameters, t)
rhs_nonstiff!(du, u, parameters, t) = rhs_nonstiff!(du, u, parameters.equation, parameters, t)
operator(rhs_stiff!, parameters) = operator(rhs_stiff!, parameters.equation, parameters)
dot_entropy(u, v, parameters) = dot_entropy(u, v, parameters.equation, parameters)

# Coefficients
struct ARS222{T} end
ARS222(T = Float64) = ARS222{T}()

function coefficients(::ARS222{T}) where T
    two = convert(T, 2)
    γ = 1 - 1 / sqrt(two)
    δ = 1 - 1 / (2 * γ)

    A_stiff = [0 0 0;
               0 γ 0;
               0 1-γ γ]
    b_stiff = [0, 1-γ, γ]
    c_stiff = [0, γ, 1]
    A_nonstiff = [0 0 0;
                  γ 0 0;
                  δ 1-δ 0]
    b_nonstiff = [δ, 1-δ, 0]
    c_nonstiff = [0, γ, 1]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end


struct ARS443{T} end
ARS443(T = Float64) = ARS443{T}()

function coefficients(::ARS443{T}) where T
    l = one(T)

    A_stiff = [0 0 0 0 0;
               0 l/2 0 0 0;
               0 l/6 l/2 0 0;
               0 -l/2 l/2 l/2 0;
               0 3*l/2 -3*l/2 l/2 l/2]
    b_stiff = [0, 3*l/2, -3*l/2, l/2, l/2]
    c_stiff = [0, l/2, 2*l/3, l/2, l]
    A_nonstiff = [0 0 0 0 0;
                  l/2 0 0 0 0;
                  11*l/18 l/18 0 0 0;
                  5*l/6 -5*l/6 l/2 0 0;
                  l/4 7*l/4 3*l/4 -7*l/4 0]
    b_nonstiff = [l/4 7*l/4 3*l/4 -7*l/4 0]
    c_nonstiff = [0, l/2, 2*l/3, l/2, l]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end


struct BPR343{T} end
BPR343(T = Float64) = BPR343{T}()

function coefficients(::BPR343{T}) where T
    l = one(T)

    A_stiff = [0 0 0 0 0;
               l/2 l/2 0 0 0;
               5*l/18 -l/9 l/2 0 0;
               l/2 0 0 l/2 0;
               l/4 0 3*l/4 -l/2 l/2]
    b_stiff = [l/4, 0, 3*l/4, -l/2, l/2]
    c_stiff = [0, l, 2*l/3, l, l]
    A_nonstiff = [0 0 0 0 0;
                  l 0 0 0 0;
                  4*l/9 2*l/9 0 0 0;
                  l/4 0 3*l/4 0 0;
                  l/4 0 3*l/4 0 0]
    b_nonstiff = [l/4, 0, 3*l/4, 0, 0]
    c_nonstiff = [0, l, 2*l/3, l, l]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end



"""
    LZ564(T = Float64)

H. Liu and J. Zou.
“Some new additive Runge–Kutta methods and their applications”.
In: Journal of Computational and Applied Mathematics 190.1-2 (2006), pp. 74–98.
"""
struct LZ564{T} end
LZ564(T = Float64) = LZ564{T}()

function coefficients(::LZ564{T}) where T
    l = one(T)

    A_stiff = [0 0 0 0 0 0 0;
               -l/6 l/2 0 0 0 0 0;
               l/6 -l/3 l/2 0 0 0 0;
               3*l/8 -3*l/8 0 l/2 0 0 0;
               l/8 0 3*l/8 -l/2 l/2 0 0;
               -l/2 0 3 -3 l l/2 0;
               l/6 0 0 0 2*l/3 -l/2 2*l/3]
    b_stiff = [l/6, 0, 0, 0, 2*l/3, -l/2, 2*l/3]
    c_stiff = [0, l/3, l/3, l/2, l/2, l, l]
    A_nonstiff = [0 0 0 0 0 0 0;
                  l/3 0 0 0 0 0 0;
                  l/6 l/6 0 0 0 0 0;
                  l/8 0 3*l/8 0 0 0 0;
                  l/8 0 3*l/8 0 0 0 0;
                  l/2 0 -3*l/2 0 2 0 0;
                  l/6 0 0 0 2*l/3 l/6 0]
    b_nonstiff = [l/6, 0, 0, 0, 2*l/3, l/6, 0]
    c_nonstiff = [0, l/3, l/3, l/2, l/2, l, l]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end


# ARK3(2)4L[2]SA–ERK:
# 3rd order type II ImEX-RK, explicit part - NOT FSAL, implicit part - SA; NOT GSA.
# This is a method by Kennedy and Carpenter.
struct ARK324L2SA_ERK{T} end
ARK324L2SA_ERK(T = Float64) = ARK324L2SA_ERK{T}()

function coefficients(::ARK324L2SA_ERK{T}) where T
    l = one(T)

    A_stiff = [
        0 0 0 0;
        (l * 1767732205903)/4055673282236 (l * 1767732205903)/4055673282236 0 0;
        (l * 2746238789719)/10658868560708 (l * -640167445237)/6845629431997 (l * 1767732205903)/4055673282236 0;
        (l * 1471266399579)/7840856788654 (l * -4482444167858)/7529755066697 (l * 11266239266428)/11593286722821 (l * 1767732205903)/4055673282236
    ]

    A_nonstiff = [
        0 0 0 0;
        (l * 1767732205903)/2027836641118 0 0 0;
        (l * 5535828885825)/10492691773637 (l * 788022342437)/10882634858940 0 0;
        (l * 6485989280629)/16251701735622 (l * -4246266847089)/9704473918619 (l * 10755448449292)/10357097424841 0
    ]

    c_nonstiff = [0, (l * 1767732205903)/2027836641118, (l * 3)/5, l]
    b_nonstiff = [(l * 1471266399579)/7840856788654 (l * -4482444167858)/7529755066697 (l * 11266239266428)/11593286722821 (l * 1767732205903)/4055673282236]

    c_stiff = [0, (l * 1767732205903)/2027836641118, (l * 3)/5, l]
    b_stiff = [(l * 1471266399579)/7840856788654 (l * -4482444167858)/7529755066697 (l * 11266239266428)/11593286722821 (l * 1767732205903)/4055673282236]


    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end


# ARK4(3)6L[2]SA–ERK:
# 4th order type II ImEX-RK, explicit part - NOT FSAL, implicit part - SA; NOT GSA.
# This is a method by Kennedy and Carpenter.
struct ARK436L2SA_ERK{T} end
ARK436L2SA_ERK(T = Float64) = ARK436L2SA_ERK{T}()

function coefficients(::ARK436L2SA_ERK{T}) where T
    l = one(T)

    A_nonstiff = [
        0 0 0 0 0 0;
        (l * 1/2) 0 0 0 0 0;
        (l * 13861)/62500 (l * 6889)/62500 0 0 0 0;
        (l * -116923316275)/2393684061468 (l * -2731218467317)/15368042101831 (l * 9408046702089)/11113171139209 0 0 0;
        (l * -451086348788)/2902428689909 (l * -2682348792572)/7519795681897 (l * 12662868775082)/11960479115383 (l * 3355817975965)/11060851509271 0 0;
        (l * 647845179188)/3216320057751 (l * 73281519250)/8382639484533 (l * 552539513391)/3454668386233 (l * 3354512671639)/8306763924573 (l * 4040)/17871 0
    ]

    A_stiff = [
        0 0 0 0 0 0;
        (l * 1/4) (l * 1/4) 0 0 0 0;
        (l * 8611)/62500 (l * -1743)/31250 (l * 1/4) 0 0 0;
        (l * 5012029)/34652500 (l * -654441)/2922500 (l * 174375)/388108 (l * 1/4) 0 0;
        (l * 15267082809)/155376265600 (l * -71443401)/120774400 (l * 730878875)/902184768 (l * 2285395)/8070912 (l * 1/4) 0;
        (l * 82889)/524892 0 (l * 15625)/83664 (l * 69875)/102672 (l * -2260)/8211 (l * 1/4)
    ]

    c_nonstiff = [0, (l * 1/2), (l * 83)/250, (l * 31)/50, (l * 17)/20, l]
    b_nonstiff = [(l * 82889)/524892 0 (l * 15625)/83664 (l * 69875)/102672 (l * -2260)/8211 (l * 1/4)]
    c_stiff = [0, (l * 1/2), (l * 83)/250, (l * 31)/50, (l * 17)/20, l]
    b_stiff = [(l * 82889)/524892 0 (l * 15625)/83664 (l * 69875)/102672 (l * -2260)/8211 (l * 1/4)]

    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end


# SSP2-IMEX(2,2,2):
# 2nd order L-stable type I ImEX-RK, explicit part - NOT FSAL, implicit part - NOT SA; NOT GSA.
struct SSP2ImEx222{T} end
SSP2ImEx222(T = Float64) = SSP2ImEx222{T}()

function coefficients(::SSP2ImEx222{T}) where T
    l = one(T)
    two = convert(T, 2)
    γ = 1 - 1 / sqrt(two)

    A_stiff = [γ 0;
               l−2*γ γ]
    b_stiff = [l/2 l/2]
    c_stiff = [γ, l−γ]

    A_nonstiff = [0 0;
                  l 0]
    b_nonstiff = [l/2 l/2]
    c_nonstiff = [0, l]

    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end


# SSP2-IMEX(3,3,2):
# 2nd order L-stable type I ImEX-RK, explicit part - NOT FSAL, implicit part - SA; NOT GSA.
struct SSP2ImEx332{T} end
SSP2ImEx332(T = Float64) = SSP2ImEx332{T}()

function coefficients(::SSP2ImEx332{T}) where T
    l = one(T)

    A_stiff = [l/4 0 0;
               0 l/4 0;
               l/3 l/3 l/3]
    b_stiff = [l/3 l/3 l/3]
    c_stiff = [l/4, l/4, l]

    A_nonstiff = [0 0 0;
               l/2 0 0;
               l/2 l/2 0]
    b_nonstiff = [l/3 l/3 l/3]
    c_nonstiff = [0, l/2, l]

    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end


# AGSA(3,4,2):
# 2nd order type I ImEX-RK, explicit part - FSAL, implicit part - SA; GSA.
struct AGSA342{T} end
AGSA342(T = Float64) = AGSA342{T}()

function coefficients(::AGSA342{T}) where T
    l = one(T)

    A_stiff = [ 168999711*l/74248304 0 0 0;
                44004295*l/24775207 202439144*l/118586105 0 0;
               -6418119*l/169001713 -748951821*l/1043823139 12015439*l/183058594 0;
               -370145222*l/355758315 l/3 0 202439144*l/118586105]

    b_stiff = [-370145222*l/355758315 l/3 0 202439144*l/118586105]
    c_stiff = [168999711*l/74248304, 10233769644823783*l/2937995298698735, -22277245178359531777915943*l/32292981880895017395247558 , 1]

    A_nonstiff = [ 0 0 0 0;
                  -139833537*l/38613965 0 0 0;
                   85870407*l/49798258 -121251843*l/1756367063 0 0;
                   l/6 l/6 2*l/3 0]

    b_nonstiff = [l/6 l/6 2*l/3 0]
    c_nonstiff = [0, -139833537*l/38613965, 144781823980515147*l/87464020145976254, 1 ]

    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

# SSP3-IMEX(3,4,3):
# 3rd order L-stable type I ImEX-RK, explicit part - NOT FSAL, implicit part - NOT SA; NOT GSA.

struct SSP3ImEx343{T} end
SSP3ImEx343(T = Float64) = SSP3ImEx343{T}()

function coefficients(::SSP3ImEx343{T}) where T
    l = one(T)
    α = 0.24169426078821
    β = 0.06042356519705
    η = 0.12915286960590

    A_stiff = [ α 0 0 0;
               -α α 0 0;
                0 l-α α 0;
               β η l/2−β−η−α α]
    b_stiff = [0 l/6 l/6 2*l/3]
    c_stiff = [α, 0,l,l/2]

    A_nonstiff = [0 0 0 0;
                  0 0 0 0;
                  0 l 0 0;
                  0 l/4 l/4 0]
    b_nonstiff = [0 l/6 l/6 2*l/3]
    c_nonstiff = [0, 0, l, l/2]

    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end


# IMEX ARK solver
# This assumes that the stiff part is linear and that the stiff solver is
# diagonally implicit.
function solve_imex(rhs_stiff!, rhs_stiff_operator, rhs_nonstiff!,
                    q0, tspan, parameters, alg;
                    dt,
                    relaxation = false,
                    callback = Returns(nothing))
    A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff = coefficients(alg)

    s = length(b_stiff)
    @assert size(A_stiff, 1) == s && size(A_stiff, 2) == s &&
            length(b_stiff) == s && length(c_stiff) == s &&
            size(A_nonstiff, 1) == s && size(A_nonstiff, 2) == s &&
            length(b_nonstiff) == s && length(c_nonstiff) == s
    Base.require_one_based_indexing(A_stiff, b_stiff, c_stiff,
                                    A_nonstiff, b_nonstiff, c_nonstiff)

    q = copy(q0) # solution
    y = similar(q) # stage value
    t = first(tspan)
    tmp = similar(q)
    k_stiff = Vector{typeof(q)}(undef, s) # stage derivatives
    k_nonstiff = Vector{typeof(q)}(undef, s) # stage derivatives
    for i in 1:s
        k_stiff[i] = similar(q)
        k_nonstiff[i] = similar(q)
    end

    # Setup system matrix template and factorizations
    W, factorization, factorizations = let
        a = findfirst(!iszero, diag(A_stiff))
        factor = a * dt
        W = I - factor * rhs_stiff_operator

        if W isa UniformScaling
            # This happens if the stiff part is zero
            factorization = W
        else
            factorization = lu(W)
        end

        # We cache the factorizations for different factors for efficiency.
        # Since we do not use adaptive time stepping, we will only have a few
        # different factors.
        factorizations = Dict(factor => copy(factorization))
        W, factorization, factorizations
    end

    while t < last(tspan)
        dt = min(dt, last(tspan) - t)

        # Compute stages
        for i in 1:s
            # RHS of linear system
            fill!(tmp, 0)
            for j in 1:(i-1)
                @. tmp += A_stiff[i, j] * k_stiff[j] + A_nonstiff[i, j] * k_nonstiff[j]
            end
            @. tmp = q + dt * tmp

            # Setup and solve linear system
            if iszero(rhs_stiff_operator) || iszero(A_stiff[i, i])
                copyto!(y, tmp)
            else
                factor = A_stiff[i, i] * dt

                F = let W = W, factor = factor,
                        factorization = factorization,
                        rhs_stiff_operator = rhs_stiff_operator
                    get!(factorizations, factor) do
                        fill!(W, 0)
                        W[diagind(W)] .= 1
                        @. W -= factor * rhs_stiff_operator
                        if issparse(W)
                            lu!(factorization, W)
                        else
                            factorization = lu!(W)
                        end
                        copy(factorization)
                    end
                end
                ldiv!(y, F, tmp)
            end

            # Compute new stage derivatives
            rhs_stiff!(k_stiff[i], y, parameters, t + c_stiff[i] * dt)
            rhs_nonstiff!(k_nonstiff[i], y, parameters, t + c_nonstiff[i] * dt)
        end

        # Update solution
        fill!(tmp, 0)
        for j in 1:s
            @. tmp += b_stiff[j] * k_stiff[j] + b_nonstiff[j] * k_nonstiff[j]
        end

        if relaxation # TODO? && (t + dt != last(tspan))
            @. y = dt * tmp # = qnew - q
            gamma = -2 * dot_entropy(q, y, parameters) / dot_entropy(y, y, parameters)
            @. q = q + gamma * y
            t += gamma * dt
        else
            @. q = q + dt * tmp
            t += dt
        end

        callback(q, parameters, t)

        if any(isnan, q)
            @error "NaNs in solution at time $t" q @__LINE__
            error()
        end
    end

    return (; u = (q0, q),
              t = (first(tspan), t))
end



#####################################################################
# General KdV interface

abstract type AbstractEquation end
Base.Broadcast.broadcastable(equation::AbstractEquation) = Ref(equation)


#####################################################################
# KdV discretization

struct KdV <: AbstractEquation end

get_u(u, equations::KdV) = u

function rhs_stiff!(du, u, equation::KdV, parameters, t)
    (; D3) = parameters

    # du = -D3 * u
    mul!(du, D3, u)
    @. du = -du

    return nothing
end

operator(::typeof(rhs_stiff!), equation::KdV, parameters) = parameters.minus_D3

function rhs_nonstiff!(du, u, equation::KdV, parameters, t)
    (; D1, tmp) = parameters
    one_third = one(eltype(u)) / 3

    # du = -1/3 * (D1 * (u.^2) + u .* (D1 * u))
    # This semidiscretization conserves the linear and quadratic invariants
    @. tmp = -one_third * u^2
    mul!(du, D1, tmp)
    mul!(tmp, D1, u)
    @. du = du - one_third * u * tmp

    return nothing
end

# TODO ?
# function rhs_full!(du, u, equation::KdV, parameters, t)
#     return nothing
# end

# TODO: operator(::typeof(rhs_full!), equation::KdV, parameters) ?

function dot_entropy(u, v, equation::KdV, parameters)
    (; D1, tmp) = parameters
    @. tmp = u * v
    return 0.5 * integrate(tmp, D1)
end

function setup(u_func, equation::KdV, tspan, D, D3 = nothing)
    if D isa PeriodicUpwindOperators && isnothing(D3)
        D1 = D.central
        D3 = sparse(D.plus) * sparse(D.central) * sparse(D.minus)
        minus_D3 = -D3
    elseif D isa FourierDerivativeOperator && isnothing(D3)
        D1 = D
        D3 = D1^3
        minus_D3 = -Matrix(D3)
    elseif D isa PeriodicDerivativeOperator && D3 isa PeriodicDerivativeOperator
        D1 = D
        D3 = D3
        minus_D3 = -sparse(D3)
    else
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = SummationByPartsOperators.grid(D1)
    u0 = u_func(tspan[1], x, equation)
    tmp = similar(u0)
    q0 = u0
    parameters = (; equation, D1, D3, minus_D3, tmp)
    return (; q0, parameters)
end


# Physical setup of a traveling wave solution with speed `c`
solitary_wave_setup() = (xmin = -40.0, xmax = 40.0, c = 1.2)

function solitary_wave_solution(t, x::Number, equation::KdV)
    (; xmin, xmax, c) = solitary_wave_setup()

    A = 3 * c
    K = sqrt(3 * A) / 6
    x_t = mod(x - c * t - xmin, xmax - xmin) + xmin

    return A / cosh(K * x_t)^2
end

function solitary_wave_solution(t, x::AbstractVector, equation::KdV)
    solitary_wave_solution.(t, x, equation)
end



# TODO: development stuff - unused?
function kdv_test(; semidiscretization = upwind_operators,
                    N = 2^8,
                    domain_traversals = 10,
                    accuracy_order = 4,
                    alg = ARS443(),
                    dt = 0.1,
                    kwargs...)

    equation = KdV()
    xmin, xmax, c = solitary_wave_setup()
    if semidiscretization === fourier_derivative_operator
        # Fourier
        D1 = fourier_derivative_operator(xmin, xmax, N)
        D3 = nothing
    elseif semidiscretization === periodic_derivative_operator
        # central FD
        D1 = periodic_derivative_operator(derivative_order = 1;
                                          accuracy_order, xmin, xmax, N)
        D3 = periodic_derivative_operator(derivative_order = 3;
                                          accuracy_order, xmin, xmax, N)
    elseif semidiscretization === upwind_operators
        # upwind FD
        D1 = upwind_operators(periodic_derivative_operator;
                              derivative_order = 1, accuracy_order,
                              xmin, xmax, N)
        D3 = nothing
    end

    tspan = (0.0, domain_traversals * (xmax - xmin) / c)
    q0, parameters = setup(solitary_wave_solution, equation,
                           tspan, D1, D3)

    @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                           rhs_nonstiff!,
                           q0, tspan, parameters, alg;
                           dt = dt, kwargs...)

    x = grid(parameters.D1)
    fig = plot(; xguide = L"x", yguide = L"u", plot_kwargs()...)
    plot!(fig, x, get_u(sol.u[begin], equation);
          label = L"u^0", plot_kwargs()...)
    plot!(fig, x, get_u(sol.u[end], equation);
          label = L"u", plot_kwargs()...)
    return fig
end

#=
julia> kdv_test()

=#


#####################################################################

struct HyperbolizedKdV{T} <: AbstractEquation
    τ::T
end

function get_u(q, equations::HyperbolizedKdV)
    N = length(q) ÷ 3
    return view(q, 1:N)
end

function rhs_stiff!(dq, q, equation::HyperbolizedKdV, parameters, t)
    (; D1) = parameters
    N = size(D1, 2)

    du = view(dq, (0 * N + 1):(1 * N))
    dv = view(dq, (1 * N + 1):(2 * N))
    dw = view(dq, (2 * N + 1):(3 * N))

    u = view(q, (0 * N + 1):(1 * N))
    v = view(q, (1 * N + 1):(2 * N))
    w = view(q, (2 * N + 1):(3 * N))

    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        # du .= -D₊ * w
        mul!(du, D1.plus, w, -1)

        # dv .= (D * v - w) / τ
        mul!(dv, D1.central, v)
        @. dv = inv_τ * (dv - w)

        # dw .= (-D₋ * u + v) / τ
        mul!(dw, D1.minus, u)
        @. dw = inv_τ * (-dw + v)
    else
        # du .= -D₊ * w
        mul!(du, D1, w)
        @. du = -du

        # dv .= (D * v - w) / τ
        mul!(dv, D1, v)
        @. dv = inv_τ * (dv - w)

        # dw .= (-D₋ * u + v) / τ
        mul!(dw, D1, u)
        @. dw = inv_τ * (-dw + v)
    end

    return nothing
end

function operator(::typeof(rhs_stiff!), equation::HyperbolizedKdV, parameters)
    D1 = parameters.D1
    τ = equation.τ
    inv_τ = inv(τ)

    if D1 isa PeriodicUpwindOperators
        Dm = sparse(D1.minus)
        D = sparse(D1.central)
        Dp = sparse(D1.plus)
        O = zero(D)
        jac = [O O -Dp;
               O inv_τ*D -inv_τ*I;
               -inv_τ*Dm inv_τ*I O]
        dropzeros!(jac)
        return jac
    elseif D1 isa FourierDerivativeOperator
        D = Matrix(D1)
        O = zero(D)
        jac = [O O -D;
               O inv_τ*D -inv_τ*I;
               -inv_τ*D inv_τ*I O]
        return jac
    else
        D = sparse(D1)
        O = zero(D)
        jac = [O O -D;
               O inv_τ*D -inv_τ*I;
               -inv_τ*D inv_τ*I O]
        dropzeros!(jac)
        return jac
    end
end

function rhs_nonstiff!(dq, q, equation::HyperbolizedKdV, parameters, t)
    (; D1, tmp) = parameters
    N = size(D1, 2)
    one_third = one(eltype(q)) / 3

    du = view(dq, (0*N+1):(1*N))
    dv = view(dq, (1*N+1):(2*N))
    dw = view(dq, (2*N+1):(3*N))

    u = view(q, (0*N+1):(1*N))

    if D1 isa PeriodicUpwindOperators
        D = D1.central
    else
        D = D1
    end

    @. tmp = -one_third * u^2
    mul!(du, D, tmp)
    mul!(tmp, D, u)
    @. du = du - one_third * u * tmp

    fill!(dv, zero(eltype(dv)))
    fill!(dw, zero(eltype(dw)))

    return nothing
end

# TODO ?
# function rhs_full!(dq, q, equation::HyperbolizedKdV, parameters, t)
# end

# TODO: operator(::typeof(rhs_full!), equation::HyperbolizedKdV, parameters) ?


function dot_entropy(q1, q2, equation::HyperbolizedKdV, parameters)
    (; D1, tmp) = parameters
    N = size(D1, 2)

    u1 = view(q1, (0 * N + 1):(1 * N))
    v1 = view(q1, (1 * N + 1):(2 * N))
    w1 = view(q1, (2 * N + 1):(3 * N))

    u2 = view(q2, (0 * N + 1):(1 * N))
    v2 = view(q2, (1 * N + 1):(2 * N))
    w2 = view(q2, (2 * N + 1):(3 * N))

    τ = equation.τ
    half = one(τ) / 2
    @. tmp = half * (u1 * u2 + τ * v1 * v2 + τ * w1 * w2)

    return integrate(tmp, D1)
end


function setup(u_func, equation::HyperbolizedKdV, tspan, D1, D3 = nothing)
    if !isnothing(D3)
        throw(ArgumentError("Combination of operators not supported"))
    end

    x = SummationByPartsOperators.grid(D1)
    u0 = u_func(tspan[1], x, equation)

    if D1 isa PeriodicUpwindOperators
        v0 = D1.minus * u0
        w0 = D1.central * v0
    else
        v0 = D1 * u0
        w0 = D1 * v0
    end

    q0 = vcat(u0, v0, w0)

    tmp = similar(u0)

    parameters = (; equation, D1, tmp)
    return (; q0, parameters)
end

function setup(u_func, v_func, w_func, equation::HyperbolizedKdV,
               tspan, D1)
    x = SummationByPartsOperators.grid(D1)
    u0 = u_func(tspan[1], x, equation)
    v0 = v_func(tspan[1], x, equation)
    w0 = w_func(tspan[1], x, equation)

    q0 = vcat(u0, v0, w0)

    tmp = similar(u0)

    parameters = (; equation, D1, tmp)
    return (; q0, parameters)
end


# Physical setup of a traveling wave solution with speed `c`
function solitary_wave_solution(t, x, equation::HyperbolizedKdV)
    solitary_wave_solution(t, x, KdV())
end



# TODO: development stuff - unused?
function hyperbolized_kdv_test(; τ = 1.0e-4,
                                 semidiscretization = upwind_operators,
                                 N = 2^8,
                                 domain_traversals = 10,
                                 accuracy_order = 4,
                                 alg = ARS443(),
                                 dt = 0.1,
                                 kwargs...)

    equation = HyperbolizedKdV(τ)
    xmin, xmax, c = solitary_wave_setup()
    if semidiscretization === fourier_derivative_operator
        # Fourier
        D1 = fourier_derivative_operator(xmin, xmax, N)
        D3 = nothing
    elseif semidiscretization === periodic_derivative_operator
        # central FD
        D1 = periodic_derivative_operator(derivative_order = 1;
                                          accuracy_order, xmin, xmax, N)
        # D3 = periodic_derivative_operator(derivative_order = 3;
        #                                   accuracy_order, xmin, xmax, N)
        D3 = nothing
    elseif semidiscretization === upwind_operators
        # upwind FD
        D1 = upwind_operators(periodic_derivative_operator;
                              derivative_order = 1, accuracy_order,
                              xmin, xmax, N)
        D3 = nothing
    end

    tspan = (0.0, domain_traversals * (xmax - xmin) / c)
    q0, parameters = setup(solitary_wave_solution, equation,
                           tspan, D1, D3)

    @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                           rhs_nonstiff!,
                           q0, tspan, parameters, alg;
                           dt = dt, kwargs...)

    x = grid(parameters.D1)
    fig = plot(; xguide = L"x", yguide = L"u", plot_kwargs()...)
    plot!(fig, x, get_u(sol.u[begin], equation);
          label = L"u^0", plot_kwargs()...)
    plot!(fig, x, get_u(sol.u[end], equation);
          label = L"u", plot_kwargs()...)
    return fig
end

#=
julia> hyperbolized_kdv_test()

=#



#####################################################################
# Convergence in the hyperbolic relaxation parameter

function convergence_tests_relaxation_imex(; latex = false,
                                             domain_traversals = 0.25,
                                             N = 2^9,
                                             alg = ARS443(),
                                             dt = 0.005,
                                             kwargs...)
    # Initialization of physical and numerical parameters
    (; xmin, xmax, c) = solitary_wave_setup()
    tspan = (0.0, domain_traversals * (xmax - xmin) / c)

    D1 = upwind_operators(periodic_derivative_operator;
                          derivative_order = 1, accuracy_order = 8,
                          xmin, xmax, N)

    τs = [1.0e0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7,
          1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12, 1.0e-13, 1.0e-14]
    errors_u_ana = Float64[]
    errors_u_num = Float64[]

    u_ref_num = let
        equation = KdV()
        (; q0, parameters) = setup(solitary_wave_solution, equation,
                                   tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)
        get_u(sol.u[end], equation)
    end

    for τ in τs
        equation = HyperbolizedKdV(τ)
        (; q0, parameters) = setup(solitary_wave_solution, equation,
                                   tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt = dt, kwargs...)

        x = grid(parameters.D1)
        u = get_u(sol.u[end], equation)

        u_ref_ana = solitary_wave_solution(sol.t[end], x, KdV())
        error_u = integrate(abs2, u - u_ref_ana, parameters.D1) |> sqrt
        push!(errors_u_ana, error_u)

        error_u = integrate(abs2, u - u_ref_num, parameters.D1) |> sqrt
        push!(errors_u_num, error_u)
    end

    let errors_u = errors_u_ana
        @info "Errors with respect to the analytical KdV soliton"
        eoc_u = compute_eoc(inv.(τs), errors_u)

        data = hcat(τs, errors_u, eoc_u)
        header = ["τ", "L2 error u", "L2 EOC u"]
        kwargs = (; header, formatters=(ft_printf("%.2e", [1, 2]),
                                        ft_printf("%.2f", [3])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end

    let errors_u = errors_u_num
        @info "Errors with respect to the numerical KdV solution"
        eoc_u = compute_eoc(inv.(τs), errors_u)

        data = hcat(τs, errors_u, eoc_u)
        header = ["τ", "L2 error u", "L2 EOC u"]
        kwargs = (; header, formatters=(ft_printf("%.2e", [1, 2]),
                                        ft_printf("%.2f", [3])))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
    end
end



#####################################################################
# Solitary wave of the KdV equation

function plot_kdv_solitary_wave(; τ = 1.0e-4,
                                  domain_traversals = 1.25,
                                  accuracy_order = 1, N = 2^8,
                                  alg = ARS443(), dt = 0.01,
                                  kwargs...)
    # Initialization of physical and numerical parameters
    (; xmin, xmax, c) = solitary_wave_setup()
    tspan = (0.0, domain_traversals * (xmax - xmin) / c)

    D1 = upwind_operators(periodic_derivative_operator;
                          derivative_order = 1, accuracy_order,
                          xmin, xmax, N)


    # Setup plot
    fig_sol = plot(; xguide = L"x", yguide = L"u", plot_kwargs()...)


    # KdV
    let equation = KdV()
        (; q0, parameters) = setup(solitary_wave_solution, equation,
                                   tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                                rhs_nonstiff!,
                                q0, tspan, parameters, alg;
                                dt, kwargs...)

        x = grid(parameters.D1)
        plot!(fig_sol, x, get_u(sol.u[begin], equation);
              label = L"u^0", plot_kwargs()...)
        plot!(fig_sol, x, get_u(sol.u[end], equation);
              label = "KdV", plot_kwargs()...)
    end


    # HyperbolizedKdV with IMEX
    let equation = HyperbolizedKdV(τ)
        (; q0, parameters) = setup(solitary_wave_solution, equation,
                                   tspan, D1)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, kwargs...)

        x = grid(parameters.D1)
        plot!(fig_sol, x, get_u(sol.u[end], equation);
              label = "HyperbolizedKdV", linestyle = :dot, plot_kwargs()...)
    end

    plot!(fig_sol; legend = :topleft)

    return fig_sol
end


function kdv_solitary_wave_error_growth(; τ = 1.0e-4,
                                          domain_traversals = 30,
                                          accuracy_order = 7, N = 2^8,
                                          alg = ARS443(), dt = 0.025,
                                          kwargs...)
    # Initialization of physical and numerical parameters
    (; xmin, xmax, c) = solitary_wave_setup()
    tspan = (0.0, domain_traversals * (xmax - xmin) / c)

    D1 = upwind_operators(periodic_derivative_operator;
                          derivative_order = 1, accuracy_order,
                          xmin, xmax, N)


    # Setup plot
    fig_err = plot(; xguide = L"t", yguide = L"Error of $u$", plot_kwargs()...)


    # Setup callback computing the error
    series_t = Vector{Float64}()
    series_error = Vector{Float64}()
    callback = let series_t = series_t, series_error = series_error
        function (q, parameters, t)
            (; tmp, equation) = parameters

            u = get_u(q, equation)
            u_ref = solitary_wave_solution(t, grid(parameters.D1), KdV())

            @. tmp = u - u_ref
            err = integrate(abs2, tmp, parameters.D1) |> sqrt

            push!(series_t, t)
            push!(series_error, err)
            return nothing
        end
    end


    # KdV
    let equation = KdV()
        (; q0, parameters) = setup(solitary_wave_solution, equation,
                                   tspan, D1)

        @info "KdV without relaxation"
        empty!(series_t)
        empty!(series_error)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, callback, kwargs...)
        plot!(fig_err, series_t, series_error;
              label = "KdV, baseline", plot_kwargs()...)

        @info "KdV with relaxation"
        empty!(series_t)
        empty!(series_error)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, callback, relaxation = true, kwargs...)
        plot!(fig_err, series_t, series_error;
              label = "KdV, relaxation", plot_kwargs()...)
    end


    # HyperbolizedKdV with IMEX
    let equation = HyperbolizedKdV(τ)
        (; q0, parameters) = setup(solitary_wave_solution, equation,
                                   tspan, D1)

        @info "Hyperbolized KdV without relaxation"
        empty!(series_t)
        empty!(series_error)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, callback, kwargs...)
        plot!(fig_err, series_t, series_error;
              label = "HKdV, baseline", plot_kwargs()...)

        @info "Hyperbolized KdV with relaxation"
        empty!(series_t)
        empty!(series_error)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, callback, relaxation = true, kwargs...)
        plot!(fig_err, series_t, series_error;
              label = "HKdV, relaxation", plot_kwargs()...)
    end


    plot!(fig_err; xscale = :log10, yscale = :log10, legend = :topleft, plot_kwargs()...)


    return fig_err
end



#####################################################################
# Traveling wave of the KdVH equation generated numerically
# FIXME: remove
function plot_kdvh_cusps(; domain_traversals = 1.0,
                           N = 999,
                           accuracy_order = 1,
                           alg = ARS443(),
                           dt, kwargs...)
    # Load numerically generated traveling wave solution
    data = open(joinpath(@__DIR__, "KdVH_cusp.txt"), "r") do io
        readdlm(io, comments = true)
    end
    c = 1.5
    τ = 1.0
    x = range(data[1, 1], data[end, 1], length = size(data, 1))
    xmin = x[begin]; xmax = x[end]
    u0 = data[:, 2]
    u0itp = LinearInterpolation((x,), u0, extrapolation_bc = Periodic())
    function u_traveling(t, x::Number, equation)
        x_t = mod(x - c * t - xmin, xmax - xmin) + xmin
        u0itp(x_t)
    end
    function u_traveling(t, x::AbstractVector, equation)
        u_traveling.(t, x, equation)
    end


    # Initialization of physical and numerical parameters
    tspan = (0.0, domain_traversals * (xmax - xmin) / c)

    D1 = upwind_operators(periodic_derivative_operator;
                          derivative_order = 1, accuracy_order,
                          xmin, xmax, N)

    # Setup plot
    fig_sol = plot(; xguide = L"x", yguide = L"u")

    # KdVH with IMEX
    (; q0, parameters) = setup(u_traveling, HyperbolizedKdV(τ), tspan, D1)
    @time sol = solve_imex(rhs_stiff!,
                           operator(rhs_stiff!, parameters),
                           rhs_nonstiff!,
                           q0, tspan, parameters, alg;
                           dt = dt, kwargs...)

    x = grid(parameters.D1)
    plot!(fig_sol, x, sol.u[begin][1:length(x)]; label = "initial", plot_kwargs()...)
    plot!(fig_sol, x, sol.u[end][1:length(x)]; label = "KdVH", plot_kwargs()...)
    plot!(fig_sol; legend = :topleft)

    return fig_sol
end



#####################################################################
# Traveling wave of the KdVH equation generated numerically
function plot_kdvh_traveling(; domain_traversals = 1.0,
                               N = nothing,
                               accuracy_order = 3,
                               alg = ARS443(),
                               name = "KdVH_periodic",
                               dt, kwargs...)
    # Load numerically generated traveling wave solution
    data = open(joinpath(@__DIR__, name * "_xuvw.txt"), "r") do io
        readdlm(io, comments = true)
    end
    c = 2.0
    τ = 1.0
    x = range(data[1, 1], data[end, 1], length = size(data, 1))
    xmin = x[begin]; xmax = x[end]
    u0 = data[:, 2]
    u0itp = LinearInterpolation((x,), u0, extrapolation_bc = Periodic())
    v0 = data[:, 3]
    v0itp = LinearInterpolation((x,), v0, extrapolation_bc = Periodic())
    w0 = data[:, 4]
    w0itp = LinearInterpolation((x,), w0, extrapolation_bc = Periodic())
    function u_traveling(t, x::Number, equation)
        x_t = mod(x - c * t - xmin, xmax - xmin) + xmin
        u0itp(x_t)
    end
    function u_traveling(t, x::AbstractVector, equation)
        u_traveling.(t, x, equation)
    end
    function v_traveling(t, x::Number, equation)
        x_t = mod(x - c * t - xmin, xmax - xmin) + xmin
        v0itp(x_t)
    end
    function v_traveling(t, x::AbstractVector, equation)
        v_traveling.(t, x, equation)
    end
    function w_traveling(t, x::Number, equation)
        x_t = mod(x - c * t - xmin, xmax - xmin) + xmin
        w0itp(x_t)
    end
    function w_traveling(t, x::AbstractVector, equation)
        w_traveling.(t, x, equation)
    end

    # Initialization of physical and numerical parameters
    tspan = (0.0, domain_traversals * (xmax - xmin) / c)

    if N === nothing
        N = size(data, 1) - 1 # for periodicity
    end
    D1 = upwind_operators(periodic_derivative_operator;
                          derivative_order = 1, accuracy_order,
                          xmin, xmax, N)

    # KdVH with IMEX
    (; q0, parameters) = setup(u_traveling,
                               v_traveling,
                               w_traveling,
                               HyperbolizedKdV(τ), tspan, D1)

    # Solution plot
    fig_sol = plot(; xguide = L"x")
    x = grid(parameters.D1)
    plot!(fig_sol, x, q0[1:length(x)]; label = L"u^0", plot_kwargs()...)
    plot!(fig_sol, x, q0[length(x)+1:2*length(x)]; label = L"v^0", plot_kwargs()...)
    plot!(fig_sol, x, q0[2*length(x)+1:3*length(x)]; label = L"w^0", plot_kwargs()...)
    savefig(fig_sol, joinpath(FIGDIR, name * "_initial.pdf"))

    # Setup callback computing the error
    series_t = Vector{Float64}()
    series_error_u = Vector{Float64}()
    series_error_v = Vector{Float64}()
    series_error_w = Vector{Float64}()
    callback = let series_t = series_t,
                   series_error_u = series_error_u,
                   series_error_v = series_error_v,
                   series_error_w = series_error_w
        function (q, parameters, t)
            (; tmp, equation) = parameters

            push!(series_t, t)

            u = view(q, 1:length(q) ÷ 3)
            u_ref = u_traveling(t, grid(parameters.D1), equation)
            @. tmp = u - u_ref
            err = integrate(abs2, tmp, parameters.D1) |> sqrt
            push!(series_error_u, err)

            v = view(q, length(q) ÷ 3 + 1:2 * length(q) ÷ 3)
            v_ref = v_traveling(t, grid(parameters.D1), equation)
            @. tmp = v - v_ref
            err = integrate(abs2, tmp, parameters.D1) |> sqrt
            push!(series_error_v, err)

            w = view(q, 2 * length(q) ÷ 3 + 1:length(q))
            w_ref = w_traveling(t, grid(parameters.D1), equation)
            @. tmp = w - w_ref
            err = integrate(abs2, tmp, parameters.D1) |> sqrt
            push!(series_error_w, err)

            return nothing
        end
    end

    # Solve KdVH system
    @time sol = solve_imex(rhs_stiff!,
                           operator(rhs_stiff!, parameters),
                           rhs_nonstiff!,
                           q0, tspan, parameters, alg;
                           dt, callback, kwargs...)

    # Error plot
    fig_err = plot(; xguide = L"t", yguide = "Error")
    plot!(fig_err, series_t, series_error_u; label = L"u", plot_kwargs()...)
    plot!(fig_err, series_t, series_error_v; label = L"v", plot_kwargs()...)
    plot!(fig_err, series_t, series_error_w; label = L"w", plot_kwargs()...)
    savefig(fig_err, joinpath(FIGDIR, name * "_error.pdf"))

    # Solution plot at the final time
    fig_sol = plot(; xguide = L"x")
    plot!(fig_sol, x, sol.u[end][1:length(x)]; label = L"u", plot_kwargs()...)
    plot!(fig_sol, x, sol.u[end][length(x)+1:2*length(x)]; label = L"v", plot_kwargs()...)
    plot!(fig_sol, x, sol.u[end][2*length(x)+1:3*length(x)]; label = L"w", plot_kwargs()...)
    savefig(fig_sol, joinpath(FIGDIR, name * "_final.pdf"))

    @info "Results saved in directory $FIGDIR" name
    return nothing
end

#=
Reproduce results as follows:

julia> plot_kdvh_traveling(name = "KdVH_periodic", dt = 0.1, domain_traversals = 100.0, relaxation = true)

julia> plot_kdvh_traveling(name = "KdVH_negative_cusp", dt = 0.025, domain_traversals = 100.0, relaxation = true)

julia> plot_kdvh_traveling(name = "KdVH_positive_cusp", dt = 0.0025, domain_traversals = 25.0, relaxation = false)

=#



#####################################################################
# Approximation of a traveling wave of the KdVH equation
function plot_kdvh_traveling_approximation(; domain_traversals = 1.0,
                                             N = 1000,
                                             accuracy_order = 3,
                                             alg = ARS443(),
                                             dt, kwargs...)
    name = "KdVH_traveling_approximation"
    c = -0.5
    τ = 1.0
    xmin = -45.0
    xmax = +45.0
    function u_traveling(t, x::Number, equation)
        x_t = mod(x - c * t - xmin, xmax - xmin) + xmin
        5 / (3 * sqrt(cosh(5 * x_t / 3))) - 1
    end
    function u_traveling(t, x::AbstractVector, equation)
        u_traveling.(t, x, equation)
    end
    function v_traveling(t, x::Number, equation)
        x_t = mod(x - c * t - xmin, xmax - xmin) + xmin
        u = u_traveling(t, x, equation)
        (1 - 0.25 * (1 + 2 * u)) * (-25 * sinh(5 * x_t / 3) / (18 * sqrt(cosh(5 * x_t / 3))^3))
    end
    function v_traveling(t, x::AbstractVector, equation)
        v_traveling.(t, x, equation)
    end
    function w_traveling(t, x::Number, equation)
        u = u_traveling(t, x, equation)
        -0.5 * u * (1 + u)
    end
    function w_traveling(t, x::AbstractVector, equation)
        w_traveling.(t, x, equation)
    end

    # Initialization of physical and numerical parameters
    tspan = (0.0, domain_traversals * (xmax - xmin) / abs(c))

    D1 = upwind_operators(periodic_derivative_operator;
                          derivative_order = 1, accuracy_order,
                          xmin, xmax, N)

    # KdVH with IMEX
    (; q0, parameters) = setup(u_traveling,
                               v_traveling,
                               w_traveling,
                               HyperbolizedKdV(τ), tspan, D1)

    # Solution plot
    fig_sol = plot(; xguide = L"x")
    x = grid(parameters.D1)
    plot!(fig_sol, x, q0[1:length(x)]; label = L"u^0", plot_kwargs()...)
    plot!(fig_sol, x, q0[length(x)+1:2*length(x)]; label = L"v^0", plot_kwargs()...)
    plot!(fig_sol, x, q0[2*length(x)+1:3*length(x)]; label = L"w^0", plot_kwargs()...)
    savefig(fig_sol, joinpath(FIGDIR, name * "_initial.pdf"))

    # Setup callback computing the error
    series_t = Vector{Float64}()
    series_error_u = Vector{Float64}()
    series_error_v = Vector{Float64}()
    series_error_w = Vector{Float64}()
    callback = let series_t = series_t,
                   series_error_u = series_error_u,
                   series_error_v = series_error_v,
                   series_error_w = series_error_w
        function (q, parameters, t)
            (; tmp, equation) = parameters

            push!(series_t, t)

            u = view(q, 1:length(q) ÷ 3)
            u_ref = u_traveling(t, grid(parameters.D1), equation)
            @. tmp = u - u_ref
            err = integrate(abs2, tmp, parameters.D1) |> sqrt
            push!(series_error_u, err)

            v = view(q, length(q) ÷ 3 + 1:2 * length(q) ÷ 3)
            v_ref = v_traveling(t, grid(parameters.D1), equation)
            @. tmp = v - v_ref
            err = integrate(abs2, tmp, parameters.D1) |> sqrt
            push!(series_error_v, err)

            w = view(q, 2 * length(q) ÷ 3 + 1:length(q))
            w_ref = w_traveling(t, grid(parameters.D1), equation)
            @. tmp = w - w_ref
            err = integrate(abs2, tmp, parameters.D1) |> sqrt
            push!(series_error_w, err)

            return nothing
        end
    end

    # Solve KdVH system
    @time sol = solve_imex(rhs_stiff!,
                           operator(rhs_stiff!, parameters),
                           rhs_nonstiff!,
                           q0, tspan, parameters, alg;
                           dt, callback, kwargs...)

    # Error plot
    fig_err = plot(; xguide = L"t", yguide = "Error")
    plot!(fig_err, series_t, series_error_u; label = L"u", plot_kwargs()...)
    plot!(fig_err, series_t, series_error_v; label = L"v", plot_kwargs()...)
    plot!(fig_err, series_t, series_error_w; label = L"w", plot_kwargs()...)
    savefig(fig_err, joinpath(FIGDIR, name * "_error.pdf"))

    # Solution plot at the final time
    fig_sol = plot(; xguide = L"x")
    plot!(fig_sol, x, sol.u[end][1:length(x)]; label = L"u", plot_kwargs()...)
    plot!(fig_sol, x, sol.u[end][length(x)+1:2*length(x)]; label = L"v", plot_kwargs()...)
    plot!(fig_sol, x, sol.u[end][2*length(x)+1:3*length(x)]; label = L"w", plot_kwargs()...)
    savefig(fig_sol, joinpath(FIGDIR, name * "_final.pdf"))

    @info "Results saved in directory $FIGDIR" name
    return nothing
end

#=
julia> plot_kdvh_traveling_approximation(dt = 0.025, N = 1000, domain_traversals = 10)

=#