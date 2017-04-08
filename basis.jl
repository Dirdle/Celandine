#Creates basis sets

module basis

bsdirpath = "celandine\\res\\basis\\"
bsfileform = ".csv"

type BasisSet
    name::String
    basis::Dict{Integer, Array{Any}}
end

function getBasis(bsname::String)
    bsfilepath = bsdirpath * bsname * bsfileform
    f = open(bsfilepath)
    basisSet = BasisSet(bsname, Dict())

    #add each line in the file
    for l in readlines(f)
        addLine!(basisSet, l)
    end
    return basisSet
end

function addLine!(b::BasisSet, line::String)
    l = map(strip, split(line, ','))
    linenum = parse(l[1])
    linearr = formArray(l[2:end])
    b.basis[linenum] = linearr
end

function formArray(line::Vector)
    #In the input list, elements alternate between letters and strings of
    #numbers separated by spaces
    A = Array{Any}(2,0)
    for i = 1:2:length(line)
        #i indexes letters, i+1 indexes blocks of numbers
        a = line[i]
        b = listPairedNumbers(line[i+1])
        A = hcat(A, [a,b])
    end
    return A
end

function listPairedNumbers(numstring)
    nlist = split(numstring, ' ')
    A = Array{Float64}(2,0)
    for i = 1:2:length(nlist)
        A = hcat(A, [parse(nlist[i]), parse(nlist[i+1])])
    end
    return A
end

export BasisSet, getBasis
end
