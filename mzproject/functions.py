from os import listdir
import numpy as np
import re

from . import dependencies as dep


def get_files(folder_path=dep.input_path, list_key: str = "", limit: int = 0) -> list:
    """
    Returnes names of files as list that can be imported into add_files
    :param limit: how much elements is returned (default 0 means all elements)
    :param list_key: If changed it can be used to get one of subsets of files
    :param folder_path: absolute path to folder. If deafult doesn't work change link with
    mzproject.dependencies.change_input(input_path)--> path has to have / at the end
    :return: list of files (strings)
    """
    all_files = listdir(folder_path)
    beer_files = [i for i in all_files if not (("Eff" in i) or ("inf" in i.lower()) or ("Solvent" in i))]
    QC_files_blank = [i for i in all_files if (("QC" in i) or ("MIX" in i) or ("blank" in i.lower()))]
    QC_files = [i for i in all_files if (("MIX" in i) and ("Eff" not in i))]
    beer_files = [i for i in beer_files if ((i not in QC_files) and ("Water" not in i) and ("QC" not in i))]
    simulant_files = []
    for i in all_files:
        for j in ["Blank_EtOH", "Ball_EtOH", "Blank_MeOH", "CP_EtOH", "CP_MeOH"]:
            if j in i and "4-2" not in i:
                simulant_files.append(i)
    # simulant_files = ["Blank_EtOH_1_negxml", "Ball_EtOH_1_negxml", "Ball_EtOH_3_negxml", "Blank_EtOH_3_negxml",
    #                   "Blank_MeOH_1_negxml", "Blank_EtOH_2_negxml", "Ball_EtOH_2_negxml", "Blank_MeOH_3_negxml",
    #                   "Blank_MeOH_2_negxml", "CP_EtOH_1_negxml", "CP_EtOH_2_negxml", "CP_EtOH_4-1_negxml",
    #                   "CP_EtOH_3_negxml", "CP_MeOH_2_negxml", "CP_MeOH_1_negxml", "CP_MeOH_3B_negxml",
    #                   "CP_MeOH_3A_negxml",
    #                   "CP_MeOH_3C_negxml"]
    filter_dict = {"": all_files, "beer": beer_files, "QC+blank": QC_files_blank, "QC": QC_files,
                   "beer_simulant": simulant_files}
    if not limit:
        limit = len(all_files)
    if list_key in filter_dict:
        return filter_dict[list_key][:limit]
    else:
        ret = f"Wrong key! Current posible keys are: {list(filter_dict.keys())}"
        print(ret)
        return []


def read_simple_file_list(path, types=(), header=False, sep=",", flatten=False, encoding="UTF-8"):
    """
    Reads simple files and strips, splits and changes it to apropriate type
    :param encoding: encoding
    :param path: Absolute (relative to location of running) path to desired file for reading
    :param types: list of types e.g. [tuple, str, float]. if one type is undefinable you can use: lambda x: x
    :param header: if header then it will return header columns separately in a tuple (ret_list, columns)
    :param sep: separator in file
    :param flatten: if you have 1 column to create 1d list
    :return:
    """
    conn = open(path, encoding=encoding)
    text = conn.readlines()
    conn.close()
    ret_list = []
    header_cols = text[0].strip().split(sep)
    if header:
        text = text[1:]
    for line in text:
        line = line.strip().split(sep)
        if types:
            line = [types[j](line[j]) for j in range(len(line))]
        ret_list.append(line)
    if flatten:
        ret_list = list(np.array(ret_list).flat)
    if header:
        return ret_list, header_cols
    else:
        return ret_list


def delete_duplicated(readfile="testxml.xml", mztolerance=0.03, expfile="+"):
    """ delete_duplicates reads mzxml file and saves new mzxml file without masses that are close to those in some list.
     This could be used to reduce size of raw files and speed up their processing.
    """
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


def find_between(searchstring, start, end, typeof, secondstep=""):
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


def get_unwanted_masses(path="../sirius_import/deleted_masses.txt", skip=0):
    conn = open(path, "r", encoding="UTF-8")
    text = conn.readlines()
    masses = [float(i) for i in text[skip:]]
    conn.close()
    print(text[:skip])
    return masses


def generate_name_index_dict_regex(list_names, regex_dict, offset_indices=0):
    group_dict = dict()
    for i, name in enumerate(list_names):
        for rd_key, rd_val in regex_dict.items():
            if sum([int(bool(re.search(re_pattern, name))) for re_pattern in rd_val]) > 0:
                cut_curr_name = rd_key
                break
        else:
            raise KeyError(f"There is no group for: {name}")
        if cut_curr_name in group_dict:
            group_dict[cut_curr_name].append(i + offset_indices)
        else:
            group_dict[cut_curr_name] = [i + offset_indices]
    return group_dict


if __name__ == "__main__":
    pass
    # print(delete_duplicated(readfile="../sirius_import/Can_B_N-NEG.xml", mztolerance=0.03,
    #                         expfile="20181203-Can_B_N-NEG_filtered_0.03.xml"))
