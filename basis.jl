#Creates and manages basis sets

module Basis
export BasisFunction, getNormalisedBasisFunction

using Printf
using Combinatorics
#define an extended double factorial that returns 1 for negative input
#This is for the cases where l, m or n = 0
function df!!(n::Integer)
    if n >= 0
        return doublefactorial(n)
    else
        return 1
    end
end

BSFILEFORM = ".csv"
#Chemists: this seems like a REALLY bad naming schema...
SHELLNAMES = ['s', 'p', 'd', 'f', 'g', 'h']

#Types

#Set of primitive gaussian exponents and coefficients associated with
#a specific shell, e.g. hydrogen 1s.
struct ShellBasis
    type::String
    expon::Array{Float64}
    coeff::Array{Float64}
end

#A particular contracted gaussian basis function, at a specific location,
#for a specific shell in a specific atom.
struct BasisFunction
    #Location
    origin::Array{Float64, 1}
    #Angular momenta
    angMom::Array{Int64, 1}
    #Primitive Gaussian exponents, coefficients
    expon::Array{Float64}
    coeff::Array{Float64}
    #Primitive Gaussian normalisation coefficients
    norma::Array{Float64}
end

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, b::BasisFunction)
    write(io, "Origin: $(b.origin) \n")
    write(io, "Exponents: $(b.expon) \n")
    write(io, "Coefficients: $(b.coeff) \n")
    write(io, "Shell: $(b.angMom) \n")
    write(io, "Normalisation: $(b.norma)")
end

#Functions

function triangle(n::Int64)
    return (n^2 + n) ÷ 2
end

function loadBasisSet(bsname::String)
    bsfilepath = joinpath(@__DIR__, "res", "basis", bsname * BSFILEFORM)
    open(bsfilepath) do f
        basis = Vector{Vector{ShellBasis}}()
        for l in readlines(f)
            push!(basis, readShellBases(l))
        end
        return basis
    end
end

function readShellBases(s::String)
    #drop the first element (the atomic number)
    spl = split(s, ',')[2:end]
    shells = Vector{ShellBasis}([])
    #spl will alternate between letters giving the type (s, p, d, f)
    #and
    for i = 1:2:length(spl)
        t = spl[i]
        expocoeff = map(Meta.parse, split(spl[i+1]))
        r = reshape(expocoeff, (2,3))
        shell = ShellBasis(t, r[1,:], r[2,:])
        push!(shells, shell)
    end
    return shells
end

function getNormalisedBasisFunction(name::String, atom::Int64, shell::Int64,
    angMom::Array{Int64, 1}, origin::Array{Float64, 1})

    #Loading the basis set from file will be SLOW, this should only be done once
    bSet = loadBasisSet(name)[atom]
    #Need the values specifically for this shell
    l, m, n = angMom
    L = sum(angMom)
    #is this even right...?
    shBs = bSet[triangle(shell) + L + 1]

    coeff = shBs.coeff
    expon = shBs.expon

    #Normalisation
    #See http://www.chem.unifr.ch/cd/lectures/files/module5.pdf for
    #detailed method

    #Defined in order to simplify future expressions
    q = π^1.5 * df!!(2*l - 1) * df!!(2*m - 1) * df!!(2*n - 1)
    primCount = length(expon)

    # println("Normalising: ")
    # display(BasisFunction(origin, angMom, expon, coeff, ones(Float64, size(coeff))))

    #Normalisation factors for primitives
    norm = .√(2^(2*L + 1.5) * expon .^ (L + 1.5) / q)
    # println("Norm: $norm")
    #Normalisation factor for contracted gaussians
    prefactor = q / 2^L

    N = .0
    for i = 1:primCount
        for j = 1:primCount
            N += norm[i] * norm[j] * coeff[i] * coeff[j] /
                ((expon[i] + expon[j]) ^ (L + 1.5))
        end
    end
    coeff *= (N * prefactor)^-0.5

    return BasisFunction(origin, angMom, expon, coeff, norm)
end


function testBasis()

    println("Testing basis import...")
    testbasisname = "sto3g"
    testbasis = loadBasisSet(testbasisname)
    println("Created basis ")
    println("Checking basis contents...")
    println(testbasis[1])
    println(testbasis[11])

    println("Testing basis function generation...")
    testbf = getNormalisedBasisFunction(testbasisname, 1, 0, [0,0,0], [0.,0.,0.])
    println("Basis function: Origin - $(testbf.origin),
    Angular Momentum - $(testbf.angMom), Exponents - $(testbf.expon),
    Coefficients - $(testbf.coeff)")

end

#end of module
end