#Hartree-Fock code

module hf

function rhf(atoms, maxiter)
    #Restricted/closed-shell hartree-fock routine
    #Crude hartree-fock algorithm:
    cont = true
    itercount = 0
    #Guess ϕ_m(r), m ∈ 1, 2 ... N_electrons - N eqns
    ϕ = getOrbitalInitialGuess()
    while cont

        #Build v_h(r), K and thus determine F

        ϕ_old = copy(ϕ)
        #Solve eigenvalue eqns to get new ϕ_m(r)
        ϕ = getRevisedOrbitals()

        #If ϕ_m(r)_guess ~ ϕ_m(r)_derived, stop
        if (itercount > maxiter) | (hasConverged(ϕ_old, ϕ))
            cont = false
        end
    end

    if itercount > maxiter
        println("Celandine has failed to converge")
    else
        println("Converged to solution.")
        println(generateInformation())


end

function getInitialGuessOrbitals()
end

function getV()
end

function getK()
end

function getF()
end

function getRevisedOrbitals()
end

function hasConverged(oldOrbs, newOrbs, ϵ)
end

function generateInformation()
    #?
end

export rhf

end
