from os import listdir
import numpy as np
import mzproject.dependencies as dep


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
