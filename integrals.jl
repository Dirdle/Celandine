#Create, save, and load arrays of one- and two-electron integrals
module Integrals

using Combinatorics
using LinearAlgebra
using PyCall
#PyCall is supposed to be able to use the regular python install
#but in practice ignores it and uses conda instead

using HypergeometricFunctions

include("basis.jl")
include("moleculeReader.jl")
using .moleculeReader

SPECFN = pyimport("scipy.special")

mutable struct TwoElecItg{T,N} <: AbstractArray{T,N}
    #Purpose of this array type is to store a 4D array in 1/8th the space
    #by exploiting the 8-fold permutational symmetry of TEIs.
    #Behaviour must be exactly as if it were in fact the full 4D array.
    #i.e. should be indexable by [i,j,k,l]
    vals::Array{T,N}
    isTriangleIndexed::Bool
    triangleCache::Array{Int64}
    dims::Int64
end

function TwoElecItg(n::Int64)
    #Creates a blank TEI array with dimension n
    c = precomputeTriangles(triangle(n+1))
    return TwoElecItg(zeros(Float64, c[end]), true, c, n)
end

function Base.size(a::TwoElecItg)
    return ntuple(x->a.dims,4)
end

function Base.getindex(a::TwoElecItg, I::Vararg{Int, 4})
    i, j, k, l = I
    if a.isTriangleIndexed
        ij = getTriangleIndex(i, j, a.triangleCache)
        kl = getTriangleIndex(k, l, a.triangleCache)
        return a.vals[getTriangleIndex(ij, kl, a.triangleCache)]
    else
        return a.vals[i,j,k,l]
    end
end

function Base.setindex!(a::TwoElecItg, v, I::Vararg{Int, 4})
    i, j, k, l = I
    if a.isTriangleIndexed
        ij = getTriangleIndex(i, j, a.triangleCache)
        kl = getTriangleIndex(k, l, a.triangleCache)
        a.vals[getTriangleIndex(ij, kl, a.triangleCache)] = v
    else
        a.vals[i,j,k,l] = v
    end
end

function Base.show(io::IO, a::TwoElecItg)
    N = a.dims
    if a.isTriangleIndexed
        println("Indexed $N×$N×$N×$N TwoElecItg:")
        show(io, a.vals)
    else
        show(io, a.vals)
    end
end

mutable struct CoefficientExpander
    α::Float64
    β::Float64
    Q::Float64
    cache::Array{Float64, 3}
    cacheOffset::Array{Int64, 1}
end

function CoefficientExpander(a::Float64, b::Float64, q::Array{Float64, 1})
    x = CoefficientExpander(a, b, q[1], zeros(Float64, (1,1,1)), [0,0])
    y = CoefficientExpander(a, b, q[2], zeros(Float64, (1,1,1)), [0,0])
    z = CoefficientExpander(a, b, q[3], zeros(Float64, (1,1,1)), [0,0])

    return (x,y,z)
end

#=
Utility Functions
=#
function triangle(n::Int64)
    return n*(n+1) ÷ 2
end

#Faster indexing with a cache of triangle numbers
function getTriangleIndex(i::Int64, j::Int64, c::Array{Int64})
    return i > j ? c[i] + j : c[j] + i
end

function getHypertriangleIndex(i::Int64, j::Int64, k::Int64, l::Int64, c::Array{Int64})
    ij = getTriangleIndex(i, j, c)
    kl = getTriangleIndex(k, l, c)
    return getTriangleIndex(ij, kl, c)
end

function precomputeTriangles(lim::Int64)
    #Precalculate a cache of triangular numbers for faster lookup
    triangleCache = Vector{Int64}([1])
    for i = 2:lim
        push!(triangleCache, triangleCache[end] + i)
    end
    return triangleCache
end

function boys(n::Number, Z::Float64)
    #HypergeometricFunctions library form
    #return mFn([n+0.5], [n+1.5], BigFloat(-Z)) / (2n+1)
    #this routinely fails for large Z, which will come up in this

    #integral form
    # BZ = BigFloat(-Z)
    # int, err = quadgk(x -> x^2n * exp(BZ*x^2), 0, 1, rtol=1e-9)
    # return int
    #Also fails under same circumstances! Augh.

    #Explicit form
    #return hyp_1F1(n+0.5, n+1.5, -Z)/(2n+1)
    #Same problem. What the fuck is scipy doing?

    # PyCall version
    return SPECFN.hyp1f1(n+0.5, n+1.5, -Z)/(2n+1)
end

# function hyp_1F1(a::Float64, b::Float64, z::Float64, N=500::Int64, ϵ=1e-8)
#     # for k in range(500):
#     # term *= (a + k) * x / (b + k) / (k + 1)
#     # result += term
#     # if fabs(term) <= EPS * fabs(result):
#     #     break
#     term = result = 1
#     for k = 0:N
#         term *= (a+k)*z/(b+k)/k+1
#         result += term
#         if isnan(term) || isnan(result)
#             println("$a, $b, $z, $term, $result")
#         end
#         if abs(term) <= ϵ * abs(result)
#             return result
#         end
#     end
#     error("No result found for hyp1f1 $a, $b, $z !")
# end

function gaussianProductCentre(α::Float64, origA::Array{Float64,1}, β::Float64, origB::Array{Float64,1})
    origP = (α .* origA .+ β .* origB) ./ (α + β)
    return origP
end

#Expansion Coefficients
function expanCoeff(i::Int64, j::Int64, t::Int64,
                α::Float64, β::Float64, Qsep::Float64)
                #α, β: exponents on A, B
                #i, j: ang mom
                #t: hermite order
                #Qsep: separation along dim Q
    if !(0 <= t <= i+j)
        #oob!
        return 0
    else
        p = α + β
        q = α*β/p
        if t == i == j == 0
            #Terminus case
            return exp(-q*Qsep^2)
        elseif j == 0
            #Slide along j=0 instead of stepping along?
            E0 = expanCoeff(i-1, j, t-1, α, β, Qsep)
            E1 = expanCoeff(i-1, j, t,   α, β, Qsep)
            E2 = expanCoeff(i-1, j, t+1, α, β, Qsep)
            return E0/(2p) - E1*q*Qsep/α + E2*(t + 1)
        else
            E0 = expanCoeff(i, j-1, t-1, α, β, Qsep)
            E1 = expanCoeff(i, j-1, t,   α, β, Qsep)
            E2 = expanCoeff(i, j-1, t+1, α, β, Qsep)
            return E0/(2p) + E1*q*Qsep/β + E2*(t + 1)
        end
    end
end

function getExpCoeff(E::CoefficientExpander, i::Int64, j::Int64, t::Int64)
    if t < 0 || t > i+j
        #oob!
        return 0.0
    else
        expandCache!(E, i, j, t)

        indI = i + E.cacheOffset[1] + 1
        indJ = j + E.cacheOffset[2] + 1
        indT = t + 1
        if E.cache[indI, indJ, indT] != 0.0
            return E.cache[indI, indJ, indT]
        else
            #evaluate the expansion coefficient
            #Even if thise
            p = E.α + E.β
            q = E.α * E.β / p
            if t == i == j == 0
                #Terminus case
                term = exp(-q * E.Q^2)
                E.cache[indI, indJ, indT] = term
                return term
            elseif j < i
                #Slide along j=0 instead of stepping along?
                E0 = getExpCoeff(E, i-1, j, t-1)
                E1 = getExpCoeff(E, i-1, j, t)
                E2 = getExpCoeff(E, i-1, j, t+1)
                res = E0/(2p) - E1*q*E.Q/E.α + E2*(t + 1)
                E.cache[indI, indJ, indT] = res
                return res
            else
                E0 = getExpCoeff(E, i, j-1, t-1)
                E1 = getExpCoeff(E, i, j-1, t)
                E2 = getExpCoeff(E, i, j-1, t+1)
                res = E0/(2p) + E1*q*E.Q/E.β + E2*(t + 1)
                E.cache[indI, indJ, indT] = res
                return res
            end
        end

    end
end

function expandCache!(E::CoefficientExpander, i::Int64, j::Int64, t::Int64)
    cacheMaxI, cacheMaxJ, cacheMaxT = size(E.cache)
    #Note this is still 0-indexed. Only add 1 when addressing array
    oldIndI = i + E.cacheOffset[1]
    oldIndJ = j + E.cacheOffset[2]
    #No offset for t, but for convenience, gets the same name
    oldIndT = t

    if oldIndI < 0
        #println("i is out-of-bounds, and too negative")
        newIOffset = abs(i)
        newCacheMaxI = cacheMaxI + newIOffset - E.cacheOffset[1]
    elseif oldIndI >= cacheMaxI
        #println("i is out-of-bounds, and too positive")
        newIOffset = E.cacheOffset[1]
        newCacheMaxI = oldIndI + 1
    else
        #println("i is in-bounds")
        newIOffset = E.cacheOffset[1]
        newCacheMaxI = cacheMaxI
    end
    if oldIndJ < 0
        #println("j is out-of-bounds, and too negative")
        newJOffset = abs(j)
        newCacheMaxJ = cacheMaxJ + newJOffset - E.cacheOffset[2]
    elseif oldIndJ >= cacheMaxJ
        #println("j is out-of-bounds, and too positive")
        newJOffset = E.cacheOffset[2]
        newCacheMaxJ = oldIndJ + 1
    else
        #println("j is in-bounds")
        newJOffset = E.cacheOffset[2]
        newCacheMaxJ = cacheMaxJ
    end
    if oldIndT >= cacheMaxT
        #println("t is out of bounds (too positive)")
        newCacheMaxT = oldIndT + 1
    else
        #println("t is in bounds")
        newCacheMaxT = cacheMaxT
    end

    newcache = zeros(Float64, (newCacheMaxI, newCacheMaxJ, newCacheMaxT))
    cacheMinI = newIOffset - E.cacheOffset[1]
    cacheMinJ = newJOffset - E.cacheOffset[2]
    cacheMinT = newCacheMaxT - cacheMaxT
    # println("$cacheMinI, $cacheMinJ, $newCacheMaxI, $newCacheMaxJ")
    # display(newcache[cacheMinI+1:cacheMaxI,cacheMinJ+1:cacheMaxJ,1:cacheMaxT])
    newcache[cacheMinI + 1:cacheMaxI + newIOffset,
            cacheMinJ + 1:cacheMaxJ + newJOffset,
            cacheMinT + 1:newCacheMaxT] = E.cache
    E.cache = newcache
    E.cacheOffset = [newIOffset, newJOffset]
    return E
end

function cacheUpdateTest(i, j, t)
    testE = CoefficientExpander(1, 1, 2, zeros(Float64, (1,1,1)), [0,0])
    expandE = expandCache!(testE, i, j, t)
end

#Coulomb auxilliary Hermite Integral
function R(t::Int64, u::Int64, v::Int64, #order of Hermite derivative in x, y, z
            n::Int64, #Boys fn order
            p::Float64, #Sum of Gaussian exponents
            PCx::Float64, PCy::Float64, PCz::Float64, RPN::Float64) #distances
    Z = p * RPN^2
    if t == u == v == 0
        return (-2*p)^n * boys(n, Z)
    elseif t == u == 0
        q = PCz * R(t, u, v-1, n+1, p, PCx, PCy, PCz, RPN)
        if v > 1
            return q + (v-1) * R(t, u, v-2, n+1, p, PCx, PCy, PCz, RPN)
        else
            return q
        end
    elseif t == 0
        q = PCy * R(t, u-1, v, n+1, p, PCx, PCy, PCz, RPN)
        if u > 1
            return q + (u-1) * R(t, u-2, v, n+1, p, PCx, PCy, PCz, RPN)
        else
            return q
        end
    else
        q = PCx * R(t-1, u, v, n+1, p, PCx, PCy, PCz, RPN)
        if t > 1
            return q + (t-1) * R(t-2, u, v, n+1, p, PCx, PCy, PCz, RPN)
        else
            return q
        end
    end
end

#=
Overlap Integrals
=#
# # Uses fast coefficient expansion of dubious confidence
# function primOverlap(la::Int64, ma::Int64, na::Int64,
#                      lb::Int64, mb::Int64, nb::Int64,
#                      oriA::Array{Float64, 1}, oriB::Array{Float64, 1},
#                      expA::Float64, expB::Float64)
#     Ex, Ey, Ez = CoefficientExpander(expA, expB, oriA .- oriB)
#     Sx = getExpCoeff(Ex, la, lb, 0)
#     Sy = getExpCoeff(Ey, ma, mb, 0)
#     Sz = getExpCoeff(Ez, na, nb, 0)
#     return Sx * Sy * Sz * (π/(expA + expB))^(3//2)
# end

function primOverlap(la::Int64, ma::Int64, na::Int64,
                     lb::Int64, mb::Int64, nb::Int64,
                     oriA::Array{Float64, 1}, oriB::Array{Float64, 1},
                     expA::Float64, expB::Float64)
    Q = oriA .- oriB
    Sx = expanCoeff(la, lb, 0, expA, expB, Q[1])
    Sy = expanCoeff(ma, mb, 0, expA, expB, Q[2])
    Sz = expanCoeff(na, nb, 0, expA, expB, Q[3])
    return Sx * Sy * Sz * (π/(expA + expB))^(3//2)
end


function overlap(A::BasisFunction, B::BasisFunction)
    # println("Finding overlap of:")
    # display(A)
    # println("&")
    # display(B)
    S = .0
    for i in eachindex(A.coeff)
        for j in eachindex(B.coeff)
            S += A.norma[i] * A.coeff[i] * B.norma[j] * B.coeff[j] *
            primOverlap(A.angMom..., B.angMom..., A.origin, B.origin,
                            A.expon[i], B.expon[j])
        end
    end
    # println("Found overlap: $S \n")
    return S
end

#=
Kinetic Energy Integrals
=#
function primKinetic(la::Int64, ma::Int64, na::Int64,
                     lb::Int64, mb::Int64, nb::Int64,
                     Oa::Array{Float64, 1}, Ob::Array{Float64, 1},
                     expa::Float64, expb::Float64)
    q1 = expb * (2 * (lb + mb + nb) + 3) *
            primOverlap(la, ma, na, lb, mb, nb, Oa, Ob, expa, expb)
    q2 = -2 * expb^2 * (
        primOverlap(la, ma, na, lb + 2, mb, nb, Oa, Ob, expa, expb) +
        primOverlap(la, ma, na, lb, mb + 2, nb, Oa, Ob, expa, expb) +
        primOverlap(la, ma, na, lb, mb, nb + 2, Oa, Ob, expa, expb)
                    )
    q3 = -0.5 * (
      lb * (lb-1) * primOverlap(la, ma, na, lb - 2, mb, nb, Oa, Ob, expa, expb)
    + mb * (mb-1) * primOverlap(la, ma, na, lb, mb - 2, nb, Oa, Ob, expa, expb)
    + nb * (nb-1) * primOverlap(la, ma, na, lb, mb, nb - 2, Oa, Ob, expa, expb)
                )
    return q1 + q2 + q3
end


function kinetic(A::BasisFunction, B::BasisFunction)
    K = .0
    for i in eachindex(A.coeff)
        for j in eachindex(B.coeff)
            K += A.norma[i] * A.coeff[i] * B.norma[j] * B.coeff[j] *
            primKinetic(A.angMom..., B.angMom..., A.origin, B.origin,
                            A.expon[i], B.expon[j])
        end
    end
    return K
end

#=
Nuclear Potential Integrals
=#
function primNuclearAttraction(
        la::Int64, ma::Int64, na::Int64,
        lb::Int64, mb::Int64, nb::Int64,
        Oa::Array{Float64, 1}, Ob::Array{Float64, 1}, On::Array{Float64, 1},
        expa::Float64, expb::Float64)

    expp = expa + expb
    Op = gaussianProductCentre(expa, Oa, expb, Ob)
    RPN = norm(Op - On)

    tot = .0
    for t = 0:(la+lb+1)
        for u = 0:(ma+mb+1)
            for v = 0:(na+nb+1)
                tot +=  expanCoeff(la, lb, t, expa, expb, Oa[1] - Ob[1]) *
                        expanCoeff(ma, mb, u, expa, expb, Oa[2] - Ob[2]) *
                        expanCoeff(na, nb, v, expa, expb, Oa[3] - Ob[3]) *
                        R(t, u, v, 0, expp, (Op .- On)..., RPN)
            end
        end
    end
    return 2*π*tot/expp
end

function nuclear(A::BasisFunction, B::BasisFunction,
    On::Array{Float64, 1})
    V = .0
    for i in eachindex(A.coeff)
        for j in eachindex(B.coeff)
            V += A.norma[i] * A.coeff[i] * B.norma[j] * B.coeff[j] *
            primNuclearAttraction(A.angMom..., B.angMom...,
                            A.origin, B.origin, On,
                            A.expon[i], B.expon[j])
        end
    end
    return V
end
#Note this is only for one nucleus at On, and not scaled by nuclear charge

#=
Two-Electron Repulsion Integrals
=#
function primElecRepul(lmnA::Array{Int64,1},
                    lmnB::Array{Int64,1},
                    lmnC::Array{Int64,1},
                    lmnD::Array{Int64,1},
                    expA::Float64, expB::Float64, expC::Float64, expD::Float64,
                    oriA::Array{Float64, 1}, oriB::Array{Float64, 1},
                    oriC::Array{Float64, 1}, oriD::Array{Float64, 1})
    la, ma, na = lmnA
    lb, mb, nb = lmnB
    lc, mc, nc = lmnC
    ld, md, nd = lmnD
    expP = expA + expB # composite exponent for P (from Gaussians 'a' and 'b')
    expQ = expC + expD # composite exponent for Q (from Gaussians 'c' and 'd')
    α = expP * expQ / (expP + expQ)
    oriP = gaussianProductCentre(expA, oriA, expB, oriB)
    oriQ = gaussianProductCentre(expC, oriC, expD, oriD)
    sepAB = oriA .- oriB
    sepCD = oriC .- oriD
    RPQ = norm(oriP - oriQ)


    tot = .0
    for t = 0:(la+lb+1)
        for u = 0:(ma+mb+1)
            for v = 0:(na+nb+1)
                for τ = 0:(lc+ld+1)
                    for ν = 0:(mc+md+1)
                        for ϕ = 0:(nc+nd+1)
                            #Yes, really.
                            tot +=
                                expanCoeff(la, lb, t, expA, expB, sepAB[1]) *
                                expanCoeff(ma, mb, u, expA, expB, sepAB[2]) *
                                expanCoeff(na, nb, v, expA, expB, sepAB[3]) *
                                expanCoeff(lc, ld, τ, expC, expD, sepCD[1]) *
                                expanCoeff(mc, md, ν, expC, expD, sepCD[2]) *
                                expanCoeff(nc, nd, ϕ, expC, expD, sepCD[3]) *
                                (-1)^(τ+ν+ϕ) *
                                R(t+τ, u+ν, v+ϕ, 0, α, (oriP .- oriQ)..., RPQ)
                        end
                    end
                end
            end
        end
    end
    tot *= 2*π^2.5
    tot /= expP*expQ*√(expP+expQ)
    return tot
end

function repulsion(A::BasisFunction, B::BasisFunction, C::BasisFunction,
    D::BasisFunction)
    L = .0
    for i in eachindex(A.coeff)
        for j in eachindex(B.coeff)
            for k in eachindex(C.coeff)
                for l in eachindex(D.coeff)

                    L +=
                        A.norma[i] * A.coeff[i] * B.norma[j] * B.coeff[j] *
                        C.norma[k] * C.coeff[k] * D.norma[l] * D.coeff[l] *
                        primElecRepul(
                        A.angMom, B.angMom, C.angMom, D.angMom,
                        A.expon[i], B.expon[j], C.expon[k], D.expon[l],
                        A.origin, B.origin, C.origin, D.origin
                        )
                end
            end
        end
    end
    return L
end

#=
Shell and basis generation
=#
function shells(angMom::Int64)
    #Get the shells with given total angular momentum
    shells = Vector{Array{Int64, 1}}([])
    #N coins in m slots like picking orders for n coins and m-1 dividers
    for c in combinations(1:(2 + angMom), angMom)
        s = [0,0,0]
        for ci = 1:angMom
            #Not actually magic, just c - [0:angMom-1] but one loop
            s[c[ci] - ci + 1] += 1
        end
        push!(shells, s)
    end
    return shells
end

function getAllShells(atom::Int64)
    allShell = Vector{Array{Int64, 1}}([])
    pqns = Vector{Int64}([])
    maxElectrons = 0
    n = 0
    while maxElectrons < atom
        for L = 0:n
            sh = shells(L)
            for s in sh
                push!(allShell, s)
                push!(pqns, n)
                maxElectrons += 2
            end
        end
        n += 1
    end
    return zip(allShell, pqns)
end

function getBasisFunction(geom::Array{AtomGeom, 1}, basis::String)
    bsfn = Vector{BasisFunction}([])
    for atom in geom
        an = atom.atomNum
        orig = atom.center
        for shell in getAllShells(an)
            push!(bsfn, getNormalisedBasisFunction(basis, an,
                shell[2], shell[1], orig))
        end
    end
    return bsfn
end

#=
Integral Array Generators
=#
function generateOneElectronIntegrals(geom::Array{AtomGeom, 1}, basis::String)
    #Need to create a set of BasisFunctions, centered on geom.centers,
    #using values from basisset for those geom.atoms, with shells (?)
    #Then we just slam them pairways to get out oeis
    bsfn = getBasisFunction(geom, basis)
    n = length(bsfn)
    overlapIntegrals = ones(Float64, (n,n))
    kineticIntegrals = zeros(Float64, (n,n))
    nuclearIntegrals = zeros(Float64, (n,n))
    for i = 1:n
        for j = i:n
            a = bsfn[i]
            b = bsfn[j]
            #Overlap: self-overlap is always 1, so don't compute diagonals
            if j > i
                overlapIntegrals[i,j] = overlapIntegrals[j,i] = overlap(a,b)
            end
            #Kinetic
            kineticIntegrals[i,j] = kineticIntegrals[j,i] = kinetic(a,b)
            #Nuclear: sum over nuclei scaled by charge
            for atom in geom
                o = atom.center
                z = atom.atomNum
                incr = -z * nuclear(a, b, o)
                nuclearIntegrals[i,j] = nuclearIntegrals[j,i] += incr
            end
        end
    end
    return overlapIntegrals, kineticIntegrals, nuclearIntegrals
end

function generateTwoElectronIntegrals(geom::Array{AtomGeom, 1}, basis::String)
    bsfn = getBasisFunction(geom, basis)
    n = length(bsfn)
    tei = TwoElecItg(n)
    for i=1:n
        for j=i:n
            for k=1:i
                for l=k:n
                    #now we can generate the (i,k,k,l)'th integral and its
                    #permutational idents using bsfns i,j,k,l
                    A,B,C,D = bsfn[[i,j,k,l]]
                    #The TEI indexing already guarantees the permutational
                    # symmetry. Or, should do.
                    tei[i,j,k,l] = repulsion(A,B,C,D)
                end
            end
        end
    end
    return tei
end


function selftest()

    filepath = joinpath(@__DIR__, "res", "exampleWater.txt")
    geom = readFileToMoleculeGeom(filepath)

    over, kine, nucl = generateOneElectronIntegrals(geom, "sto3g")

    targetOver =
     [1.     0.237  0.     0.     0.     0.038  0.038
      0.237  1.     0.     0.     0.     0.386  0.386
      0.     0.     1.     0.     0.     0.268 -0.268
      0.     0.     0.     1.     0.     0.21   0.21
      0.     0.     0.     0.     1.     0.     0.
      0.038  0.386  0.268  0.21   0.     1.     0.182
      0.038  0.386 -0.268  0.21   0.     0.182  1.   ]

    targetKine =
    [29.0      -0.168    0.0     0.0    -0.0   -0.00842  -0.00842
     -0.168     0.808    0.0    -0.0     0.0    0.0705    0.0705
      0.0       0.0      2.53   -0.0     0.0    0.147    -0.147
      0.0      -0.0     -0.0     2.53   -0.0    0.115     0.115
     -0.0       0.0      0.0    -0.0     2.53  -0.0      -0.0
     -0.00842   0.0705   0.147   0.115  -0.0    0.76     -0.00398
     -0.00842   0.0705  -0.147   0.115  -0.0   -0.00398   0.76]

    targetNucl =
     [-61.6      -7.41    0.0   -0.0145   0.0      -1.23  -1.23
       -7.41    -10.0     0.0   -0.177    0.0      -2.98  -2.98
        0.0       0.0    -9.99   0.0     -0.0      -1.82   1.82
       -0.0145   -0.177   0.0   -9.94     0.0      -1.47  -1.47
        0.0       0.0    -0.0    0.0     -9.88      0.0    0.0
       -1.23     -2.98   -1.82  -1.47     0.0      -5.3   -1.07
       -1.23     -2.98    1.82  -1.47     0.0      -1.07  -5.3]

    # ERI = generateTwoElectronIntegrals(geom, "sto3g")
    # targetERI = getTestTargetERI()

    if any(map(abs, (over - targetOver)) .> 0.1)
        println("Overlap integrals failed, errors:")
        display(over - targetOver)
        println("Values:")
        display(over)
    end

    if any(map(abs, (kine - targetKine)) .> 0.1)
        println("Kinetic energy integrals failed, errors:")
        display(kine - targetKine)
        println("Values:")
        display(kine)
    end

    if any(map(abs, (nucl - targetNucl)) .> 0.1)
        println("Nuclear potential integrals failed, errors:")
        display(nucl - targetNucl)
    end

    if any(map(abs, (ERI.vals - targetERI.vals)) .> 0.1)
        println("Two-electron integrals failed, errors not shown.")
    end

end

function getTestTargetERI()
    basepath = @__DIR__
    folder = "res"
    filename = "eepoten.dat"
    path = joinpath(basepath, folder, filename)
    open(path) do f
        targ = TwoElecItg(7)
        for l in readlines(f)
            inds = split(l)[1:4]
            val = split(l)[5]
            i,j,k,l = map(Meta.parse, inds)
            targ[i, j, k, l] = Meta.parse(val)
        end
        return targ
    end
end

#end of module
end
