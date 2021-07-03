#Want to include all the modules in the project...
include("moleculeReader.jl")
include("moleculeDrawer.jl")


using .moleculeReader
using .moleculeDrawer

include("basis.jl")

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





#Need to get the expected integrals from somewhere and do comparisonss
function testOneElectronIntegrals()
    println("Testing one-electron integrals")
    testbf = OneElec.getNormalisedBasisFunction("sto3g", 1, [0,0,0], [0,0,0])
    println("The overlap of a basis fucntion with itself should be ~1.0: ")
    println(overlap(testbf, testbf))
end

testBasis()

#
# struct myStruct
#     ab::Array{Int64, 1}
#     ac::Int64
# end
#
# function myFunc!(a::myStruct)
#     a.ab = [1,2,3]
# end
#
# s = myStruct([], 3)
# myFunc!(s)
