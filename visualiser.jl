# Makes pictures given an array of atom positions

module MoleculeDrawer
export drawMolecule, getElementSymbol

using PlotlyJS
#TODO replace with Plots, unless re-doing the entire visuals overtakes

#Defines the location of the periodic table informtion file
eledata = "celandine\\res\\pt-data2.csv"

function plotMolecule(atomArray::Array)
    #create a graph displaying the atoms in the positions given by the array
    trc = scatter3d(
                    ;x=atomArray[:, 2], y=atomArray[:, 3], z=atomArray[:, 4],
                    mode="markers+text", opacity=0.5, marker_size=12,
                    text=map(x->getElementSymbol(x), atomArray[:, 1]),
                    textposition="center")
    #TODO make some of these things into inputs to this fn
    #TODO make different elements different colours :3yes
    layout = Layout(margin=attr(l=0, r=0, t=0, b=0))
    p = plot(trc, layout)
    return p
end

function drawMolecule(atomArray::Array)
    display(plotMolecule(atomArray))
end

function getElementSymbol(atomicNumber::Int32)
    #Retrieve the symbol for a given atomic number

    #The file is properly ordered, so the nth element is the nth line #EZPZ
    l = open(readlines, eledata)[atomicNumber]
    #Note that this loads the entire file into an array. May result in large
    #memory usage if the number of chemical elements increases,
    #or if using a table with very large amounts of info for each element
    elemInf = map(strip, split(l, ','))
    return elemInf[2]
end

function getElementSymbol(atomicNumber::Float64)
    #Most of the time, atomic numbers will be floats (because reasons)
    #So this is necessary
    atNumInt = Int32(round(atomicNumber))
    return getElementSymbol(atNumInt)
end

function testvisualiser()
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

end
