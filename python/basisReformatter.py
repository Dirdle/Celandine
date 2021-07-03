import os
import re

basepath = "..\\..\\..\\..\\PyQuante\\PyQuante\\Basis\\"
outpath  = "..\\res\\basis\\"

def reformatBasis(filepath):
    with open(filepath) as f:
        basisLines = getBasisLines(f)
        blocks = [b for b in divideIntoBlocks(basisLines)]
        totalOut = []
        for b in blocks:
            totalOut.append(convertBlock(b))
        return '\n'.join(totalOut)



def getBasisLines(basisfile):
    basisLines = []
    #Read until the first {
    for line in basisfile:
        if '{' in line:
            basisLines.append(line)
            break #Inelegant!
    #Continue reading where above loop left off (file obj iterator permanence)
    for line in basisfile:
        basisLines.append(line)
        if '}' in line:
            break
    return basisLines

def divideIntoBlocks(basisLines):
    #Generator for blocks
    i = j = 0
    n = len(basisLines)
    block = []
    while i < n:
        line = basisLines[i]
        block.append(line)
        for cha in line:
            if cha == '[':
                j += 1
            if cha == ']':
                j -= 1
                if j == 0:
                    yield block
                    block = []
        i += 1
    return

def convertBlock(block):
    #want to convert a pair of number and list of pairs of letters and list of pairs of numbers
    #Start by putting the number at the front
    #principal separator should be commas so final file can be .csv
    #maybe secondary can just be a space?
    #eg 1, S, 40 2 7 3 8 5, S, 4 3 87 1 4 2, etc
    #The grouped numbers always come in pairs
    #The lines will be REALLY long though

    #Step 0: the line structure of the block is unimportant
    blockstring = ''.join(block)
    #Step 1: we can split by "'" since this character only goes with letters
    quotesplit = blockstring.split("'")
    leadNumber = getLeadNumber(quotesplit[0])
    blocklist = quotesplit[1:]
    otptlist = [leadNumber, ',']
    #pattern is now:
    #[letter, stringlist of number pairs, letter, list...  letter, list]

    #Step 2: step through the list pairwise
    for i in range(0, len(blocklist), 2):
        otptlist.append(blocklist[i])
        otptlist.append(',')
        otptlist.extend(getNumbers(blocklist[i+1]))
        otptlist.append(',')
    return " ".join(otptlist[:-1])
    #Ignore the last comma


def getLeadNumber(leadstring):
    loc = leadstring.index(':')
    mtc = re.compile("[\d.-]+")
    return ''.join(mtc.findall(leadstring[:loc]))

def getNumbers(numPairString):
    mtc = re.compile("[\d.-]+")
    #This is a regex that matches a string of however-many digits and .s
    #The question of whether it's more likely to be tripped up by multiple .s
    #or by a non-decimal number if we used \d+.\d+ instead is interesting
    return mtc.findall(numPairString)

def writeOutput(name, outstring):
    with open(outpath + name + ".csv", 'w') as f:
        f.write(outstring)
    return

if __name__ == "__main__":
    print(os.getcwd())
    print("Looking in", basepath)
    pathlist = os.listdir(basepath)
    print("Found files", pathlist[:3], "...")
    for p in pathlist:
        if not p in ["Tools.py", "basis.py", "__init__.py", "lacvp.py"]:
            name = p[:-3]
            print("Converting", p)
            s = reformatBasis(basepath + p)
            writeOutput(name, s)
