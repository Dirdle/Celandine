#Information that applies to whole file:
#-measurement unit (angstrom etc)
#-using MP2 etc
#-using PBC etc
#it's complicated

#Molecules defined in files by format:
#%atomic_number%, %x%, %y%, %z%

#molecules defined in program by array:
#[number, Vector]
#Distances in the program are always in nm (?)

module moleculeReader

mutable struct AtomGeom
    atomNum::Int64
    center::Array{Float64, 1}
end

struct UniversalInformation
    lengthunit::Real #Unit of length as fraction of nm
end

function readFileToArray(filepath)
    #Read the file in the given location
    #Return the contained atomic structure as an array

    #Open the file
    open(filepath) do f
        t = readstring(f)
        #Remove any \r's. Awful char.
        t = replace(t, '\r', "")
        a = separateText(t)
        #Convert the body-block to an array of numeric values
        return convertTextBlockToArray(a[2])
    end
end

function readFileToMoleculeGeom(filepath)
    open(filepath) do f
        moleGeom = Vector{AtomGeom}([])
        t = strip(replace(read(f, String), '\r' => ""))
        block = separateText(t)[2]
        for l in block
            a = AtomGeom(parseAtomicNum(l), parseCenter(l))
            push!(moleGeom, a)
        end
        return moleGeom
    end
end

function parseAtomicNum(s::SubString{String})
    return Meta.parse(split(s)[1])
end

function parseCenter(s::SubString{String})
    return map(Meta.parse, split(s)[2:end])
end

function readFileToUniversalInformation(filepath)
    #Read the file in the given location
    #Return an object containing the information set in the header
end

function convertTextBlockToArray(text::Array)
    #Given the subset of text in the file which constructs molecules,
    #convert to an array of integers
    #This creates a 1D array of vectors
    A = [lineConvert(l) for l in text]
    #This converts it to a 2D array
    A2 = permutedims(hcat(A...), [2,1])
    #Obviously there are other ways to reach this result
    return A2
end

function convertTextBlockToArray(text::String)
    textArray = split(text, '\n')
    return convertTextBlockToArray(textArray)
end

function lineConvert(line)
    #Converts a single line from string to numerics
    a = split(line, ',')
    b = map(strip, a)
    c = map(x->parse(Float64, x), b)
    return c
end

function separateText(text)
    #Returns the text separated into two arrays of strings
    blocks = split(text, "\n\n")
    if length(blocks) == 1
        #No header. Or, no body, but in that case: bigger problems
        head = Array(String, 0)
        body = split(blocks[1], '\n')
    else
        head = split(blocks[1], '\n')
        body = split(blocks[2], '\n')
    end
    return [head, body]
end



export readFileToArray, readFileToUniversalInformation, UniversalInformation, AtomGeom, MoleculeGeom, readFileToMoleculeGeom

end
