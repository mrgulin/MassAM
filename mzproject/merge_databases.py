import cirpy  # https://cirpy.readthedocs.io/_/downloads/en/latest/pdf/
import molmass
import xlwt


class Suspectlist:
    def __init__(self, link="", sep="¤"):
        self.suspectdict = dict()
        if link:
            conn = open(link, "r", encoding="UTF-8")
            text = conn.readlines()
            for line in text:
                line = line.strip().split(sep)
                self.suspectdict[line[0]] = line[1:]

        # CAS : [name, table number,  "iupac", "formula", "smiles", "Mw", "imgurl"]

    def addlist(self, name, sep, casind, nameind, tablenumber, enc="ANSI", byname=False):
        tot_added = 0
        tot_tot_count = 0
        duplicated = 0
        unrecognized = []
        conn = open(name, "r", encoding=enc)
        text = conn.readlines()
        conn.close()
        for i in text:
            if tot_tot_count % 15 == 0:
                self.savedictionary("tempsave.txt")
            tot_tot_count += 1
            j = i.strip().split(sep)
            if len(j) < max(casind, nameind):
                continue
            # From here i edited code and didn't check if it works (I added byname to this function)
            if byname:
                CAS = cirpy.resolve(j[casind][casind], "CAS")
                if CAS is None:
                    unrecognized.append(j[casind][casind])
                    continue
                if type(CAS) == list:
                    CAS = CAS[0]
                CAS = editCAS(CAS)
            else:
                CAS = editCAS(j[casind])
            # To here :)
            print(tot_tot_count, CAS)
            if CAS.count("-") == 2:
                tot_added += 1
                if CAS not in self.suspectdict:
                    ret = getsmiles(CAS)
                    if ret:
                        self.suspectdict[CAS] = [j[nameind], tablenumber] + ret
                    else:
                        self.suspectdict[CAS] = False
                else:
                    if self.suspectdict[CAS]:
                        self.suspectdict[CAS][0] += "|" + j[nameind]
                        self.suspectdict[CAS][1] += "|" + tablenumber
                        duplicated += 1
            else:
                unrecognized.append(CAS)
        print("Total features: {}\n of which duplicated: {}\n unrecognized characters: {}".format(tot_added, duplicated,
                                                                                                  unrecognized))
        print("Total length {}".format(tot_tot_count))

    def savedictionary(self, directory="Suspect_list_merged.csv3", sep="¤"):
        with open(directory, 'w', encoding="UTF-8") as outfile:
            for key, value in self.suspectdict.items():
                if value:
                    s = sep.join([key] + [str(i) for i in value])
                    outfile.write("%s\n" % s)

    def save_xslx(self, file_name="final_suspect_list.xlsx"):
        book = xlwt.Workbook(encoding="utf-8")
        sheet1 = book.add_sheet("Sheet 1")
        i = 0
        for key, value in self.suspectdict.items():
            if value:
                j = 0
                for element in [key] + [str(i) for i in value]:
                    sheet1.write(i, j, str(element))
                    j += 1
                i += 1

        book.save(file_name)


def editCAS(cas):
    cutstring = 0
    for i in range(len(cas)):
        if cas[i] == "0":
            cutstring = i + 1
        else:
            break
    return cas[cutstring:]


def getsmiles(cas):
    cas = editCAS(cas)
    mol = cirpy.Molecule(cas)
    if mol.iupac_name is not None:
        # namelist = ["iupac", "formula", "smiles", "Mw", "imgurl"]
        datalist = [mol.iupac_name, mol.formula, mol.smiles, molmass.Formula(mol.formula).isotope.mass, mol.image_url]
        # print(molmass.Formula(mol.formula).spectrum())
        return datalist
    return False


if __name__ == "__main__":
    # print(getsmiles("64742-56-9"))
    # print(getsmiles("0000123-86-4"))

    # suspectlist = Suspectlist()
    # suspectlist.addlist("1 tabula-CONSLEG__fixed.csv", ";", 1, 2, "1")
    # suspectlist.savedictionary("Suspect_list_merged.csv3", "¤")
    # suspectlist = Suspectlist("Suspect_list_merged_only1.csv", "¤")
    # suspectlist.addlist("2 tabula-CELEX_onlyCAS.csv", ";", 2, 3, "2")
    # suspectlist.savedictionary()
    # suspectlist = Suspectlist("Suspect_list_merged.csv3", "¤")
    # suspectlist.addlist("3 ac0c00532_si_001_fixed.csv", ";", 2, 1, "3")
    # suspectlist.savedictionary()
    # suspectlist.addlist("4 Annex_6_swis.csv", ";", 1, 0, "4")
    # suspectlist = Suspectlist("Suspect_list_merged.csv3", "¤")
    # suspectlist.addlist("5 Table 17_book can.csv", ";", 0, 1, "5")
    # suspectlist.savedictionary("Suspect_list_merged_+5.csv")
    # suspectlist.addlist("6 IndirectAdditives.csv", ",", 0, 1, "6")
    # suspectlist.savedictionary("Suspect_list_merged_+6.csv")
    # suspectlist = Suspectlist("Suspect_list_merged_1-7.csv", "¤")
    # suspectlist.addlist("11 toxins_T3DB_2.csv", ",", 2, 1, "11")
    # suspectlist.savedictionary("Suspect_list_merged_1-7+11.csv")
    # suspectlist = Suspectlist("Suspect_list_merged_1-7+11.csv", "¤")
    # suspectlist.addlist_byname("8 EFSAOutputs_KJ_2020.csv", ",", 0, 0, "8")
    # suspectlist.savedictionary("Suspect_list_merged_1-7+11+8.csv")
    suspectlist = Suspectlist("Suspect_list_merged_1-7+11+8.csv", "¤")
    suspectlist.save_xslx("python_export_table.xls")

# def addlist_byname(self, name, sep, casind, nameind, tablenumber, enc="ANSI"):
#     tot_added = 0
#     tot_tot_count = 0
#     duplicated = 0
#     unrecognized = []
#     conn = open(name, "r", encoding=enc)
#     text = conn.readlines()
#     conn.close()
#     for i in text:
#         if tot_tot_count % 15 == 0:
#             self.savedictionary("tempsave.txt")
#         tot_tot_count += 1
#         j = i.strip().split(sep)
#         if len(j) < max(casind, nameind):
#             continue
#         CAS = cirpy.resolve(j[casind][casind], "CAS")
#         if CAS is None:
#             unrecognized.append(j[casind][casind])
#             continue
#         if type(CAS) == list:
#             CAS = CAS[0]
#         CAS = editCAS(CAS)
#         print(tot_tot_count, CAS)
#         if CAS.count("-") == 2:
#             tot_added += 1
#             if CAS not in self.suspectdict:
#                 ret = getsmiles(CAS)
#                 if ret:
#                     self.suspectdict[CAS] = [j[nameind], tablenumber] + ret
#                 else:
#                     self.suspectdict[CAS] = False
#             else:
#                 if self.suspectdict[CAS]:
#                     self.suspectdict[CAS][0] += "|" + j[nameind]
#                     self.suspectdict[CAS][1] += "|" + tablenumber
#                     duplicated += 1
#         else:
#             unrecognized.append(CAS)
#     print("Total features: {}\n of which duplicated: {}\n unrecognized characters: {}".format(tot_added, duplicated,
#                                                                                               unrecognized))
#     print("Total length {}".format(tot_tot_count))
