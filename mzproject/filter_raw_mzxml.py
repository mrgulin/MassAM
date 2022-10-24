###############################
#  deleteduplicates reads mzxml file and saves new mzxml file without masses that are close to those in some list.
#  This could be used to reduce size of raw files and speed up their processing.
###############################


def deleteduplicated(readfile="testxml.xml", mztolerance=0.03, expfile="+"):
    dellist = getunwantedmasses()
    # conn = open("../sirius_import/Can_C_N1-NEG.xml", "r", encoding="UTF-8")
    conn = open(readfile, "r", encoding="UTF-8")
    string1 = conn.read()
    conn.close()
    print("Length of file: ", len(string1))
    i = 0
    h = 0
    inscan = False
    delbool = False
    while True:
        if string1[i:i + 5] == "<scan":
            h = i
            inscan = True
        if string1[i - 7: i] == "</scan>":
            msLevel = findbetween(string1[h: i], 'msLevel="', '"', int)
            print(h, i, msLevel, end=", ")
            if msLevel == 2:
                precursormass = findbetween(string1[h: i], '>', '</precursorMz>', float, secondstep="<precursorMz")
                for j in dellist:
                    if abs(precursormass - j) < mztolerance:
                        print("deleting mass: " + str(precursormass) + ", similar to: " + str(j))
                        delbool = True
                        if not inscan:
                            print("FATAL ERROR; Didn't find start of scan!")
                            inscan = False
                            return -1
                        break
                print(precursormass, end="\n")
            else:
                print("")
        if delbool:
            delbool = False
            string1 = string1[:h] + string1[i:]
            i = h

        i += 1
        if i == len(string1):
            break
    if expfile == "+":
        exppath = readfile[:-4] + "_" + "filtered" + "_" + str(mztolerance) + ".xml"
    else:
        exppath = expfile

    conn = open(exppath, "w", encoding="UTF-8")
    conn.write(string1)
    conn.close()


def findbetween(searchstring, start, end, typeof, secondstep=""):
    if secondstep:
        index0 = searchstring.find(secondstep)
        if index0 == -1:
            raise ValueError("Didn't find {} in string)".format(secondstep))
        else:
            searchstring = searchstring[index0 + len(start):]
    if start:
        index0 = searchstring.find(start)
        if index0 == -1:
            raise ValueError("Didn't find {} in string)".format(start))
        else:
            searchstring = searchstring[index0 + len(start):]
    index1 = searchstring.find(end)
    if index1 == -1:
        raise ValueError("Didn't find {} in string)".format(end))
    else:
        return typeof(searchstring[:index1])


def getunwantedmasses(path="../sirius_import/deleted_masses.txt", skip=0):
    conn = open(path, "r", encoding="UTF-8")
    text = conn.readlines()
    masses = [float(i) for i in text[skip:]]
    conn.close()
    print(text[:skip])
    return masses


if __name__ == "__main__":
    print(deleteduplicated(readfile="../sirius_import/Can_B_N-NEG.xml", mztolerance=0.03,
                           expfile="20181203-Can_B_N-NEG_filtered_0.03.xml"))