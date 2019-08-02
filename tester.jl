#Want to include all the modules in the project...
include("moleculeReader.jl")
include("moleculeDrawer.jl")
include("basis.jl")

using moleculeReader, moleculeDrawer, basis

function testMoleculeReader()
    println("Testing molecule reading...")
    #How to make this consistently be the right path??
    filepath = joinpath(@__DIR__, "res", "exampleMolecules.txt")
    println("Testing file read to array...")
    readArr = readFileToArray(filepath)
    println(readArr)
    println("Test successful.")
end

function testMoleculeDrawer()
    println("Testing molecule drawings...")
    #Create a test array of atom positions
    #Just an example. Maybe later get the numbers for something like DCM?
    exampleArray = [1.0 0.0 0.0 0.0
                    6.0 0.0 0.0 0.7414
                    6.0 0.0 0.7414 0.7414
                    8.0 0.7414 0.7414 0.7414
                    15.0 0.7414 0.7414 0.0]
    println("Testing element symbol retrieval...")
    elementSymbols = ""
    for atomicNum in exampleArray[:, 1]
        elementSymbols *= getElementSymbol(atomicNum) * ", "
    end
    println(elementSymbols)
    println("Testing plot construction...")
    drawMolecule(exampleArray)
end

function testBasis()
    println("Testing basis import...")
    testbasisname = "sto3g"
    testbasis = getBasis(testbasisname)
    println("Created basis " * testbasis.name)
    println("Checking basis contents...")
    print(testbasis.basis[1])
    print(testbasis.basis[11])
end


try
    testBasis()
finally
    #This undoes the inclusions and uses, except when it doesn't
    workspace()
end
