import mzproject.class_file
import mzproject.dependencies as dep
import mzproject.functions
import numpy as np
import time
import os
from datetime import datetime
import mzproject.functions as f
import mzproject.paths as paths
from subprocess import Popen, CREATE_NEW_CONSOLE
from typing import Union

types = [("d_tr", np.float64), ("d_mz", np.float64), ("n_MSMS", int), ("n", int), ("h", np.float64)]


def get_files_qc_features():
    f_list_neg = mzproject.functions.get_files()
    f_list_pos = mzproject.functions.get_files(paths.input_path_pos)
    f_list_pos = [i for i in f_list_pos if ("QC_MIX" in i) or ("Blank_EtOH" in i)]

    types_qc = [("name", "<U20"), ("mz", np.float64), ("tr", np.float64), ("adduct", "<U20"), ("mode", "<U5"),
                ("mz_ion", np.float64)]
    qc_features = f.read_simple_file_list(paths.IJS_ofline_path + "QC/QC_compounds_final.csv", header=False,
                                          encoding="UTF-8-sig")
    qc_features = np.array([tuple(i) for i in qc_features], dtype=types_qc)

    for compound in qc_features:
        compound["mz_ion"] = compound["mz"] + {**dep.delta_mass_neg_adducts, **dep.delta_mass_pos_adducts}[
            compound["adduct"]]
    return qc_features, f_list_neg, f_list_pos


def _compare_table_with_qc(peak_data_table, rel_path, python, h_table, A_table, mzproject_obj, qc_features, comment,
                           polarity, ratio_mz_tr, reduce_file, file_group_list, filenames, save_peak_graphs=False):
    output_dict = dict()
    name_dict = dict()
    for one_name in file_group_list:
        name_dict[one_name] = []
        output_dict[one_name] = []
    for i, col_name in enumerate(filenames):
        for one_name in file_group_list:
            if one_name in col_name:
                name_dict[one_name].append(i)
                break

    if python:
        d2_list = []
        d2_list_colnames = ["name", "mz1", "mz2", "tr"] + mzproject_obj.filename

    # output np.array
    out_string = f"{comment}\n{polarity} mode\n"

    for compound in qc_features:
        output_line = np.array([0], dtype=types)

        if not ((compound['mode'] == "+" and polarity == "pos") or (compound['mode'] == "-" and polarity == "neg")):
            continue
        out_string += f"--{compound['name']}------------------------\n"

        close_features = peak_data_table[
            (abs(peak_data_table["mz"] - compound["mz_ion"]) < 0.2) &
            (abs(peak_data_table["tr"] - compound["tr"]) < 0.5)
            ]
        if len(close_features) == 0:
            out_string += "\tDidn't found similar mass feature!\n"
            output_line["d_mz"] = np.nan
            output_line["d_tr"] = np.nan
            output_line["n_MSMS"] = 0
            output_line["n"] = 0
            output_line["h"] = 0
            for key, value in name_dict.items():
                output_dict[key].append(output_line[0].copy())
            continue
        best_feature = np.argmin(abs(close_features["mz"] - compound["mz_ion"]) * ratio_mz_tr + abs(
            close_features["tr"] - compound["tr"]))
        arr1 = close_features[best_feature]
        best_feature_index = arr1["index"]
        if python and save_peak_graphs:
            mzproject_obj.plot_from_index(best_feature_index, save_graph=True, show_plot=False)
        out_string += f"\tIndex: {best_feature_index}\n"
        out_string += f"\tmz difference: {compound['mz_ion'] - arr1['mz']:.6f}"
        out_string += f" ~ {(compound['mz_ion'] - arr1['mz']) / compound['mz_ion'] * 1E6:.2f}ppm\n"
        output_line["d_mz"] = compound['mz_ion'] - arr1['mz']
        out_string += f"\ttr difference: {compound['tr'] - arr1['tr']:.3f}\n"
        output_line["d_tr"] = compound['tr'] - arr1['tr']
        if python:
            spectra_number = len(arr1["scans"].split("|"))
        else:
            MSMS_dict = get_MSMS_number(dep.IJS_ofline_path + f"/{rel_path}/MS2_spectra.mgf", reduce_file)
            if int(best_feature_index) in MSMS_dict:
                spectra_number = MSMS_dict[int(best_feature_index)]
            else:
                spectra_number = 0
                print(best_feature_index)
        out_string += f'\tNumber of MSMS spectra: {spectra_number}\n\n'
        output_line["n_MSMS"] = spectra_number
        # for getting line for every feature:
        aligned_dict_index = np.argwhere(peak_data_table == arr1)[0][0]
        if python:
            d2_list.append([compound['name'], compound['mz_ion'],
                            arr1['mz'], compound["tr"]] + list(h_table[aligned_dict_index]))

        for key, value in name_dict.items():
            count = 0
            h_sum1 = []
            A_sum1 = []
            for index2 in value:
                h1 = h_table[aligned_dict_index][index2]
                if h1 > 0:
                    count += 1
                    h_sum1.append(h1)
                    A_sum1.append(A_table[best_feature][index2])
            if len(h_sum1) > 0:
                h_sum = np.average(h_sum1)
                A_sum = np.average(A_sum1)  # AVERAGE HEIGHT OF PEAK THAT IS PRESENT!!!
            else:
                h_sum = A_sum = 0
            out_string += f"\t{key}:\n"
            out_string += f"\t\tfound {count}/{len(value)} features\n"
            out_string += f"\t\tavg h {h_sum:.2f}\n"
            out_string += f"\t\tavg A {A_sum:.2f}\n"
            output_line['n'] = count
            output_line['h'] = h_sum
            output_dict[key].append(output_line[0].copy())
        # SPECIFIC FOR MIX1 AND MIX2!!!!!!!

        # out_string += f"\t\t Ratio MIX2/MIX1: A-{dict_for_ratio['MIX2'][1] / dict_for_ratio['MIX1'][1]:.2f}"
        # out_string += f" h-{dict_for_ratio['MIX2'][0] / (dict_for_ratio['MIX1'][0] + 1):.2f}\n"
        # +1 for division by 0 (no influence for ~1E4)

        out_string += "\n\n"

        # output_list.append(output_line)
    # output_list = np.array(output_list)
    for one_name in file_group_list:
        np.savetxt(dep.IJS_ofline_path + f"/{rel_path}/Experiment_results_{one_name}.txt",
                   output_dict[one_name], fmt=['%f', '%f', '%i', '%i', '%f'])
    if python:
        with open(dep.IJS_ofline_path + f"/{rel_path}/2d_table.txt", "w") as conn:
            conn.write(",".join(d2_list_colnames) + "\n")
            for line in d2_list:
                conn.write(",".join([str(i) for i in line]) + "\n")

    print(out_string)
    conn = open(dep.IJS_ofline_path + f"/{rel_path}/QC_report.txt", "w", encoding="UTF-8")
    conn.write(out_string)
    conn.close()


def run_one_experiment(file_list, name="test1", polarity="neg", input_path=None, subdirectory="test",
                       change_dict=tuple(), qc_features: Union[tuple, np.array] = tuple(),
                       from_table=False, comment="", compare_with_qc=False, file_group_list=("MIX2", "MIX1", "Blank"),
                       save_peak_graphs=True, limit_mz=tuple(), ratio_mz_tr=50):
    # Measuring time at start
    start_time = time.perf_counter()
    current_time = datetime.now().strftime("%H:%M:%S")
    if input_path is not None:
        dep.input_path = input_path
    print(f"Current Time ={current_time}, reading files from {dep.input_path}")
    if not os.path.isdir(dep.IJS_ofline_path + f"/{subdirectory}"):
        os.mkdir(dep.IJS_ofline_path + f"/{subdirectory}")
    if not os.path.isdir(dep.IJS_ofline_path + f"/{subdirectory}/{name}"):
        os.mkdir(dep.IJS_ofline_path + f"/{subdirectory}/{name}")
    dep.change_output(dep.IJS_ofline_path + f"/{subdirectory}/{name}/")
    obj = mzproject.class_file.MzProject()

    for key, value in change_dict:
        obj.parameters[key] = value

    obj.add_files_speed(file_list, limit_mass=limit_mz)
    if not from_table:
        obj.filter_constant_ions(save_graph="deleted_masses.png")
        obj.merge_features()
        obj.generate_table(save_graph=False, force=True, )
        obj.merge_duplicate_rows()
        obj.calculate_mean()
        obj.export_tables("table")
        obj.export_tables_averaged("table_averaged")
    else:
        obj.add_aligned_dict(dep.IJS_ofline_path + f"/{subdirectory}/{name}/", "table")

    # Measuring end time:
    end_time = time.perf_counter()
    current_time = datetime.now().strftime("%H:%M:%S")
    print("Current Time =", current_time)
    string1 = f"It took {end_time - start_time:.3f} s = {(end_time - start_time) / 60:.2f} min"
    print(string1)
    if not from_table:
        # Writing methods and parameters to file
        with open(dep.IJS_ofline_path + f"/{subdirectory}/{name}/report.txt", "w", encoding="UTF-8") as conn:
            conn.write(obj() + "\n\n\n" + string1)

    # Saving additional data
    out1 = [name, polarity, comment, file_list, file_group_list]
    with open(dep.IJS_ofline_path + f"/{subdirectory}/{name}/Experiment_info.txt", "w", encoding="UTF-8") as conn:
        conn.write(repr(out1))
    if compare_with_qc:
        if len(qc_features) == 0:
            print("There is no QC features!")
        _compare_table_with_qc(peak_data_table=obj.aligned_dict["peak_data"], rel_path=f"{subdirectory}/{name}",
                               h_table=obj.aligned_dict["h"], A_table=obj.aligned_dict["A"], python=True,
                               mzproject_obj=obj, qc_features=qc_features, comment=comment, polarity=polarity,
                               ratio_mz_tr=ratio_mz_tr, reduce_file=False, file_group_list=file_group_list,
                               filenames=obj.filename, save_peak_graphs=save_peak_graphs)


def get_MSMS_number(path_to_file, reduce_file=True):
    conn = open(path_to_file, "r")
    text = conn.readlines()
    conn.close()
    id_dict = dict()
    text2 = []
    for line in text:
        line = line.strip()
        if "FEATURE_ID" in line:
            key1 = int(line.split("=")[1])
            if key1 in id_dict:
                id_dict[key1] += 1
            else:
                id_dict[key1] = 1
            text2.append(line)
    if reduce_file:
        conn = open(path_to_file, "w")
        conn.write("\n".join(text2))
        conn.close()
    return id_dict


def run_one_experiment_mzmine(name, subdirectory, file_list, file_group_list, qc_features, polarity="neg", comment="",
                              protocol="MSMS_peaklist_builder", change_dict=None, calculate_table=True,
                              reduce_file=True, save_project="", ratio_mz_tr=50):
    if change_dict is None:
        change_dict = dict()
    out1 = [name, polarity, comment, file_list, file_group_list]

    if not os.path.isdir(dep.IJS_ofline_path + f"/{subdirectory}/{name}"):
        os.mkdir(dep.IJS_ofline_path + f"/{subdirectory}/{name}")
    if not os.path.isdir(dep.IJS_ofline_path + f"/{subdirectory}"):
        os.mkdir(dep.IJS_ofline_path + f"/{subdirectory}")
    if polarity == "pos":
        dep.input_path = paths.input_path_pos
    # Generate string for files
    file_list = [dep.input_path + i for i in file_list]
    file_string = "<file>" + "</file>\n\t\t<file>".join(file_list) + "</file>"

    protocol_dict = {
        "MSMS_peaklist_builder": ["MSMS_peaklist_builder_protocol_format.xml",
                                  {"filestring": "", "max_int": 50000.0, "mz_tol": 0.018, "filter_mode": "NEW AVERAGE",
                                   "tr_range_tolerance": 0.3, "mgf_path": "", "csv_path": "", "save_project": ""
                                   }],
        # mz_center_calculation: MEDIAN/AVG/AUTO
        "ADAP_builder": ["ADAP_builder_protocol_format.xml",
                         {"filestring": "", "min_n_scans": 5, "baseline_int": 5000.0, "max_int": 50000.0,
                          "mz_tol": 0.018, "SN_threshold": 5, "mz_center_calculation": "MEDIAN",
                          "MSMS_pairing_tolerance_mz": 0.05, "MSMS_pairing_tolerance_tr": 0.35,
                          "tr_range_tolerance": 0.3, "mgf_path": "", "csv_path": "", "save_project": ""}]
    }

    conn = open(dep.IJS_ofline_path + f"/Python scripts/batch_commands/{protocol_dict[protocol][0]}", "r",
                encoding="UTF-8")
    template = conn.read()
    conn.close()

    if save_project:
        save_project = f"""<batchstep method="io.github.mzmine.modules.io.projectsave.ProjectSaveAsModule">
        <parameter name="Project file">
            <current_file>{dep.IJS_ofline_path}/{subdirectory}/{name}/{save_project}</current_file>
        </parameter>
    </batchstep>"""

    kwargs = protocol_dict[protocol][1]
    kwargs["csv_path"] = dep.IJS_ofline_path + f"/{subdirectory}/{name}/quant_table.csv"
    kwargs["mgf_path"] = dep.IJS_ofline_path + f"/{subdirectory}/{name}/MS2_spectra.mgf"
    kwargs["save_project"] = save_project
    kwargs["filestring"] = file_string
    kwargs.update(change_dict)
    template = template.format(**kwargs)

    instruction_path = dep.IJS_ofline_path + f"/{subdirectory}/{name}/Batch_instructions.txt"
    conn = open(instruction_path, "w", encoding="UTF-8")
    conn.write(template)
    conn.close()
    if calculate_table:
        processlist = [paths.mzmine_path + "startMZmine-Windows.bat", instruction_path]
        print(" ".join(processlist))

        pipe = Popen(processlist, creationflags=CREATE_NEW_CONSOLE)
        pipe.wait()
        conn = open(dep.IJS_ofline_path + f"/{subdirectory}/{name}/Experiment_info.txt", "w", encoding="UTF-8")
        conn.write(repr(out1))
        conn.close()

    conn = open(dep.IJS_ofline_path + f"/{subdirectory}/{name}/quant_table.csv", "r")
    text = conn.readlines()
    conn.close()
    columns = text[0]
    text = text[1:]
    table = []
    for line in text:
        table.append(tuple([float(i) for i in line.strip().split(",")[:-1]]))
    table = np.array(table)
    columns = columns.strip().split(",")[:-1]

    if len(qc_features) > 0:
        h_table = table[:, 3:]
        temp_table = table[:, :3]
        peak_data_table = np.array([tuple(i) for i in temp_table],
                                   dtype=[('index', '<i4'), ('mz', '<f8'), ('tr', '<f8')])
        _compare_table_with_qc(peak_data_table=peak_data_table, rel_path=f"{subdirectory}/{name}",
                               h_table=h_table, A_table=np.zeros(h_table.shape), python=False, mzproject_obj=None,
                               qc_features=qc_features, comment=comment, polarity=polarity, ratio_mz_tr=ratio_mz_tr,
                               reduce_file=reduce_file, file_group_list=file_group_list, filenames=columns[3:])


def generate_3D_table(file_list, qc_features):
    """

    :param file_list: [(relative path to folder from offline_path, ...]
    :return:
    """
    main_table = []
    support_data = []
    # file list is relative to out_path!!!
    for filename in file_list:
        conn = open(dep.IJS_ofline_path + f"/{filename}/Experiment_info.txt", "r", encoding="UTF-8")
        temp1 = eval(conn.read())
        group_list = temp1[4]
        conn.close()
        for group_name in group_list:
            main_table.append(np.loadtxt(dep.IJS_ofline_path + f"/{filename}/Experiment_results_{group_name}.txt",
                                         dtype=types))
            support_data.append(temp1[:4] + [group_name])
    length_table = len(main_table[0])
    for ele in main_table:
        if len(ele) != length_table:
            print("PROBLEM! Different length of tables (different number of features)")
    main_table = np.array(main_table)
    group_table = np.vstack(([i[0] for i in support_data],
                             [i[4] for i in support_data],
                             np.nanmean(main_table["d_tr"] ** 2, axis=1) ** 0.5,
                             np.nanmax(abs(main_table["d_tr"]), axis=1),
                             np.nanmean(main_table["d_mz"] ** 2, axis=1) ** 0.5,
                             np.nanmax(abs(main_table["d_mz"]), axis=1),
                             np.nanmean(np.where(main_table["n_MSMS"] == 0, np.nan, main_table["n_MSMS"]), axis=1),
                             np.min(np.where(main_table["n_MSMS"] == 0, 100, main_table["n_MSMS"]), axis=1),
                             np.max(main_table["n_MSMS"], axis=1),
                             # np.nanmean(np.where(main_table["n"] == 0, np.nan, main_table["n"]), axis=1),
                             np.mean(main_table["n"], axis=1),
                             np.nanmean(np.where(main_table["h"] == 0, np.nan, main_table["h"]), axis=1),
                             )
                            ).transpose()
    sup_table = np.zeros((len(group_table), 7), dtype="<U50")
    for i, line in enumerate(group_table):
        if "Beer" in line[0]:
            sup_table[i][0] = "Beer"
        if "WW" in line[0]:
            sup_table[i][0] = "WW"
        if "EtOH" in line[0]:
            sup_table[i][0] = "EtOH"
        if "MZmine" in line[0]:
            sup_table[i][1] = "MZmine"
        else:
            sup_table[i][1] = "Python"
        if "EXP" in line[0]:
            ll = line[0].split("EXP")
            sup_table[i][2] = ll[-1]
        else:
            sup_table[i][2] = "-"
        if "MIX1_random" in line[0]:
            sup_table[i][3] = "MIX1_random"
        elif "random" in line[0]:
            sup_table[i][3] = "random"
        else:
            sup_table[i][3] = "normal"
        if "dmz" in line[0]:
            ll = line[0].split("dmz")
            sup_table[i][4] = ll[-1]
        else:
            sup_table[i][4] = "1"

        if "mztol" in line[0]:
            ll = line[0].split("mztol")
            sup_table[i][5] = ll[-1]
        else:
            sup_table[i][5] = "0"

        if "ADAP" in line[0]:
            sup_table[i][6] = "ADAP"
        elif "MSMSplb" in line[0]:
            sup_table[i][6] = "MSMSplb"
        elif "MZmine" not in line[0]:
            sup_table[i][6] = "Python"
        else:
            sup_table[i][6] = "0"

    np.savetxt(dep.IJS_ofline_path + f"Experiment/group_table2.csv", np.hstack((group_table, sup_table)), delimiter=",",
               fmt="%s")
    group_table = np.array([tuple(i) for i in group_table],
                           dtype=[("exp_name", "<U50"), ("group_name", "<U50"), ("s_tr", np.float64),
                                  ("max_tr", np.float64), ("s_mz", np.float64), ("max_mz", np.float64),
                                  ("avg_MSMS", np.float64), ("min_MSMS", int), ("max_MSMS", int), ("avg_n", np.float64),
                                  ("avg_h", np.float64)])

    compound_table = np.vstack((qc_features[qc_features["mode"] == "-"]["name"],
                                np.nanmean(main_table["d_tr"] ** 2, axis=0) ** 0.5,
                                np.nanmax(abs(main_table["d_tr"]), axis=0),
                                np.nanmean(main_table["d_mz"] ** 2, axis=0) ** 0.5,
                                np.nanmax(abs(main_table["d_mz"]), axis=0),
                                np.nanmean(np.where(main_table["n_MSMS"] == 0, np.nan, main_table["n_MSMS"]), axis=0),
                                np.min(np.where(main_table["n_MSMS"] == 0, 100, main_table["n_MSMS"]), axis=0),
                                np.max(main_table["n_MSMS"], axis=0),
                                np.nanmean(np.where(main_table["n"] == 0, np.nan, main_table["n"]), axis=0),
                                np.nanmean(np.where(main_table["h"] == 0, np.nan, main_table["h"]), axis=0),
                                )
                               ).transpose()
    compound_table = np.array([tuple(i) for i in compound_table],
                              dtype=[("compound_name", "<U10"), ("s_tr", np.float64),
                                     ("max_tr", np.float64), ("s_mz", np.float64), ("max_mz", np.float64),
                                     ("avg_MSMS", np.float64), ("min_MSMS", int), ("max_MSMS", int),
                                     ("avg_n", np.float64),
                                     ("avg_h", np.float64)])
    np.savetxt(dep.IJS_ofline_path + f"Experiment/group_table.csv",
               group_table, fmt=['%-20s', '%-10s', '%f', '%f', '%f', '%f', '%5.1f', '%2i', '%2i', '%.3f', '%10.1f'],
               delimiter=",")
    np.savetxt(dep.IJS_ofline_path + f"Experiment/compound_table.csv",
               compound_table, fmt=['%s', '%f', '%f', '%f', '%f', '%f', '%i', '%i', '%f', '%f'], delimiter=",")
    return main_table, support_data, group_table, compound_table


if __name__ == "__main__":
    np.seterr(all='raise')
    qc_features1, f_list_neg1, f_list_pos1 = get_files_qc_features()

    run_one_experiment(f_list_neg1[:6], 'test_python', 'neg', subdirectory='test_python', qc_features=qc_features1,
                       compare_with_qc=True, limit_mz=(200, 250), )
    # run_one_experiment_mzmine('mzmine_test', 'test_python', [i for i in f_list_neg1 if "QC_MIX" in i],
    #                           ['MIX1', "MIX2"], qc_features1)

    # table_of_paths = ["Experiment/neg/EtOH_QC", "Experiment/neg/Beer_QC", "Experiment/neg/WW_QC",
    #                   "Experiment/neg/EtOH_QC_MIX1", "Experiment/neg/EtOH_QC_MIX1_random",
    #                   "Experiment/neg/EtOH_QC_random", "Experiment/neg/MZmine_EtOH_QC",
    #                   "Experiment/neg/MZmine_WW_QC","Experiment/neg/MZmine_Beer_QC",
    #                   ]

    # table_of_paths = os.listdir(paths.IJS_ofline_path + "/Experiment/neg3/")
    # table_of_paths = ["Experiment/neg3/" + i for i in table_of_paths]
    # tup = generate_3D_table(table_of_paths, qc_features)
    # print(tup[2])
    # # MIX1 + Blank  || MIX2 + MIX1 + Blank + random samples
