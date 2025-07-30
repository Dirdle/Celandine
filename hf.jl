#Hartree-Fock code

module rhf

using LinearAlgebra

#Should this even exist...?
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

# function scourFP!(a, threshold=1.0e-15)
#     #Remove FP-error values
#     #Always work in units ~ 1!
#     for (i, val) in enumerate(a)
#         if abs(val) < threshold
#             a[i] = 0.0
#         end
#     end
# end

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
    orth = overlapIntegrals^(-1//2)
    torth = transpose(orth)
    unorth = inv(orth)
    tunorth = transpose(unorth)

    Fprime0 = torth * Hcore * orth
    F = tunorth * Fprime0 * unorth

    C0 = orth * eigvecs(Symmetric(Fprime0))
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
        Fprime = Symmetric(torth*F*orth)
        C = orth*eigvecs(Fprime)
        #scourFP!(C, 1e-14)
        Dnew = constructDensityMatrix(C)

        Enew = sum(Dnew .* (Hcore + F))
        converged = checkConvergence(Eprev, Enew, Dprev, Dnew, 0.000000000001, 0.00000000001)

    end

    return electronicEnergies .+ nuclearRepulsion
end


export rhf

end
