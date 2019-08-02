#=
This code is intended to follow the procedure described at
http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project3
in order to gain a clear understanding of the Hatree-Fock method in direct
implementation
=#

using LinearAlgebra

function getCachedNuclearRepulsion(;
    basepath = @__DIR__,
    folder::String = "cache",
    filename::String = "nucrepl",
    fileformat::String = ".txt")
    path = joinpath(basepath, folder, filename*fileformat)
    open(path) do f
        return Meta.parse(readline(f))
    end
end

function getCachedOneElectronIntegrals(filename::String;
    basepath = @__DIR__,
    folder::String = "cache",
    fileformat::String = ".txt")
    path = joinpath(basepath, folder, filename*fileformat)
    open(path) do f
        lines = readlines(f)
        #Size of array will be first entry on last line
        dimen = Meta.parse(split(lines[end])[1])
        #The desired result is a square array
        integralMatrix = zeros(Float64, dimen, dimen)
        i = j = 1
        for l in lines
            v = Meta.parse(split(l)[end])
            if j == i
                integralMatrix[i,i] = v
                i += 1
                j = 1
            else
                integralMatrix[i,j] = integralMatrix[j,i] = v
                j += 1
            end
        end
        return integralMatrix
    end
end

function triangle(n::Int64)
    return n*(n+1) ÷ 2
end

function getTriangleIndex(i_1::Int64, j_1::Int64)
    #Transform from 1-indexing to 0-indexing
    i = i_1 - 1
    j = j_1 - 1
    if i > j
        return (triangle(i) + j) + 1
    else
        return (triangle(j) + i) + 1
    end
end

#Faster indexing with a cache of triangle numbers
function getTriangleIndex(i::Int64, j::Int64, c::Array{Int64})
    if i > j
        return c[i] + j
    else
        return c[j] + i
    end
end

function getHypertriangleIndex(i::Int64, j::Int64, k::Int64, l::Int64)
    ij = getTriangleIndex(i, j)
    kl = getTriangleIndex(k, l)
    return getTriangleIndex(ij, kl)
end

function getHypertriangleIndex(i::Int64, j::Int64, k::Int64, l::Int64, c::Array{Int64})
    ij = getTriangleIndex(i, j, c)
    kl = getTriangleIndex(k, l, c)
    return getTriangleIndex(ij, kl, c)
end

function getCachedTwoElectronIntegrals(filename::String;
    basepath = @__DIR__,
    folder::String = "cache",
    fileformat::String = ".txt")
    path = joinpath(basepath, folder, filename*fileformat)
    open(path) do f
        #This time we'll store as 1D array with special access procedure
        #For some reason
        lines = readlines(f)
        n = Meta.parse(split(lines[end])[1])
        t = triangle(n)
        triangles = precomputeTriangles(t)
        twoElectronIntegrals = Vector{Float64}(zeros(Float64, triangle(t)))
        for s in lines
            μ, ν, λ, σ, value = map(Meta.parse, split(s))
            twoElectronIntegrals[getHypertriangleIndex(μ, ν, λ, σ, triangles)] = value
        end
        #the integrals get passed around alongside an indexing cache, for convenience
        return twoElectronIntegrals, triangles
    end
end

function precomputeTriangles(lim::Int64)
    #Precalculate a cache of triangular numbers for faster lookup using
    #a 1D array of integrals
    triangleCache = Vector{Int64}([0])
    for i = 1:lim
        push!(triangleCache, triangleCache[end] + i)
    end
    return triangleCache
end

function scourFP!(a, threshold=1.0e-15)
    #Remove FP-error values
    #Always work in units ~ 1!
    for (i, val) in enumerate(a)
        if abs(val) < threshold
            a[i] = 0.0
        end
    end
end

function getOccupiedOrbitals()
    #TODO improve this
    return 1:5
end

function constructDensityMatrix(MO::Array{Float64})
    density = zeros(eltype(MO), size(MO))
    for i = 1:size(density)[1]
        for j = 1:size(density)[2]
            for m in getOccupiedOrbitals()
                    density[i,j] += MO[i,m] .* MO[j,m]
            end
        end
    end
    return density
end

function getNextFock(H::Array{Float64}, D::Array{Float64}, TEI::Array{Float64},
    triangleCache::Array{Int64})
    Fnew = copy(H)
    n = size(H)[1]
    teiInd(a,b,c,d) = getHypertriangleIndex(a,b,c,d,triangleCache)
    for i = 1:n
        for j = 1:n
            for k = 1:n
                for l = 1:n
                    Fnew[i,j] += D[k,l] * (2 * TEI[teiInd(i,j,k,l)] - TEI[teiInd(i,k,j,l)])
                end
            end
        end
    end
    return Fnew
end

function checkConvergence(E1::Float64, E2::Float64,
    D1::Array{Float64}, D2::Array{Float64},
    δe::Float64, δd::Float64)
    if E2 - E1 > δe
        return false
    elseif √(sum((D2-D1)^2)) > δd
        return false
    else
        return true
    end
end

function runExample()
    nuclearRepulsion = getCachedNuclearRepulsion(fileformat=".dat")

    overlapIntegrals = getCachedOneElectronIntegrals("overlap", fileformat=".dat")
    TIntegrals = getCachedOneElectronIntegrals("kinetic", fileformat=".dat")
    nucVIntegrals = getCachedOneElectronIntegrals("nucpoten", fileformat=".dat")
    tei, tricache = getCachedTwoElectronIntegrals("eepoten", fileformat=".dat")

    Hcore = TIntegrals + nucVIntegrals
    symm = overlapIntegrals^(-1//2)
    tsymm = transpose(symm)
    unsymm = inv(symm)
    tunsymm = transpose(unsymm)

    Fprime0 = tsymm * Hcore * symm
    F = tunsymm*Fprime0*unsymm

    C0 = symm*eigvecs(Symmetric(Fprime0))
    #scourFP!(C0, 1e-14)

    D0 = constructDensityMatrix(C0)

    Einit = sum(D0 .* (Hcore + F))

    converged = false
    itermax = 50
    it = 0
    H = Hcore
    Dnew = D0
    Enew = Einit

    electronicEnergies = Vector{Float64}([])
    while !converged && it < itermax
        it += 1
        Eprev = Enew
        Dprev = Dnew
        push!(electronicEnergies, Enew)

        F = getNextFock(H, Dnew, tei, tricache)
        Fprime = Symmetric(tsymm*F*symm)
        C = symm*eigvecs(Fprime)
        #scourFP!(C, 1e-14)
        Dnew = constructDensityMatrix(C)

        Enew = sum(Dnew .* (Hcore + F))
        converged = checkConvergence(Eprev, Enew, Dprev, Dnew, 0.000000000001, 0.00000000001)

    end

    return electronicEnergies .+ nuclearRepulsion
end
