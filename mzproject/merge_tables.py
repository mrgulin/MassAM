import os
import mzproject.paths as paths
import mzproject.dependencies as dep
import mzproject.class_file
import mzproject.functions as f
import progressbar
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.cross_decomposition import PLSRegression
import matplotlib.cm as cm
import matplotlib.colors
import filecmp


def color_map_color(value, cmap_name='hsv', vmin=0, vmax=2):
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(value))[:3]  # will return rgba, we take only first 3 so we get rgb
    color = matplotlib.colors.rgb2hex(rgb)
    return color


def read_mgf(path):
    print("starting to read from: " + path)
    conn = open(path, "r")
    text = conn.readlines()
    conn.close()
    data_dict = dict()
    curr_list = None
    feature_id = None
    specter = []
    time.sleep(0.3)
    bar = progressbar.ProgressBar(max_value=len(text))
    bar.update(0)
    for id1, line in enumerate(text):
        bar.update(id1)
        line = line.strip()
        if line == "BEGIN IONS":
            curr_list = [0, 0, ""]
            specter = []
        elif "SCANS" in line:
            curr_list[2] += line.split("=")[1]
        elif len(line) == 0:
            continue
        elif line[0].isnumeric():
            pair = line.split(" ")
            pair = [float(i) for i in pair]
            specter.append(pair)
        elif "FEATURE_ID" in line:
            feature_id = int(line.split("=")[1])
        elif "FILENAME" in line:
            curr_list[2] += line.split("=")[1]
        elif "END IONS" == line:
            curr_list[0] = specter
            if feature_id in data_dict:
                data_dict[feature_id].append(curr_list)
            else:
                data_dict[feature_id] = [curr_list]
    return data_dict


class Export_ms:
    def __init__(self, split_on=100):
        self.expstring = [""]
        self.split_on = split_on
        self.current_index = 0

    def add_feature(self, name, mz, ionization, tr, i_plus_1, spectra, min_h=10):
        # scan_index_list = [[i, i.split("__")[0]] for i in line[3].split("|") if i.count("__") == 1]

        if (self.current_index + 1) % self.split_on == 0:
            self.expstring.append("")
        self.current_index += 1

        self.expstring[-1] += ">compound " + f"{name}"
        self.expstring[-1] += "\n>parentmass " + f"{mz}"
        if ionization == "pos":
            ionization_str = "[M+?]+"
        elif ionization == "neg":
            ionization_str = "[M+?]-"
        else:
            raise Exception("Problem with ionization string (only pos / neg)")

        self.expstring[-1] += "\n>ionization " + ionization_str
        self.expstring[-1] += "\n>instrumentation Unknown (LCMS)"
        # self.expstring[-1] += f"\n>source file: {float(mz):.4f}_{float(tr):.2f}" #  This was the seed of problem with
        # import in sirius!!!
        self.expstring[-1] += "\n>quality UNKNOWN"
        self.expstring[-1] += "\n>rt " + f"{tr * 60}"
        # self.expstring[-1] += "\n#scan index " + " ".join([m[0] for m in scan_index_list])
        # self.expstring[-1] += "\n#filename " + " ".join((m[1] for m in scan_index_list))
        if i_plus_1:
            self.expstring[-1] += f"\n\n>ms1\n{mz} 100.0"
            self.expstring[-1] += f"\n{float(mz) + dep.m_p} {float(i_plus_1):.4f}"
        for one_spectrum, collision_energy, index in spectra:
            if collision_energy != 0:
                str1 = f">collision {collision_energy:.0f}\n"
            else:
                str1 = ">ms2\n"
            self.expstring[-1] += f"\n\n# {index}\n" + str1
            for i in one_spectrum:
                if i[1] > min_h:
                    self.expstring[-1] += "{:.5f} {:.1f} \n".format(i[0], i[1])
        self.expstring[-1] += "\n"
        # elif mode == "cluster_averaged":
        #     energy_levels = [i for i in list(set(self.scans.tempenergy)) if not np.isnan(i)]
        #     for level in energy_levels:
        #         spectrum = average_spectrum(self, line[0], level, grouping_threshold=0.05)
        #         # TODO: Check if MS2 grouping threashold is too large/small
        #         self.expstring[-1] += f"\n>collision {level:.0f}\n"
        #         for i in spectrum:
        #             if i[1] > min_h:
        #                 self.expstring[-1] += "{:.5f} {:.1f} \n".format(i[0], i[1])
        #         self.expstring[-1] += "\n"
        #     self.expstring[-1] += "\n"

    def export_files(self, filename, folder_name=dep.output_path, add_ind=False):
        if filename == '':
            pass  # We are printing empty filename which is first call in the function and we don't have anything saved
        elif len(self.expstring[0]) < 1:
            print("Empty object!", filename)
        else:
            for ind, string in enumerate(self.expstring):
                name = f"{folder_name}/{filename}{('_' + str(ind)) * add_ind}.ms"
                with open(name, "w", encoding="UTF-8") as conn:
                    conn.write(self.expstring[-1])


class Merge_tables:
    def __init__(self, folder_list, names_list, ionization, output_path):
        self.folder_list = folder_list
        self.names_list = names_list
        self.ionization = ionization
        self.table = []
        self.group_names = []
        self.output_path = output_path
        self.table_exp = None
        self.experiment_object_list = []
        self.table_s = None
        self.table_VIP = None

    def generate_table(self):
        table = []  # new index, mz, tr, project, old index, heights...

        for file in self.folder_list:
            mzmine_project = os.path.isfile(self.output_path + file + "/quant_table.csv")
            self.experiment_object_list.append(mzmine_project)
            print(self.output_path + file + "/quant_table.csv", mzmine_project)
            if mzmine_project:
                conn = open(self.output_path + file + "/quant_table.csv", "r")
                text = conn.readlines()
                conn.close()

                header = text[0].strip().split(",")
                if header[0] != "row ID":
                    raise Exception("1. column must be row ID")
                if header[1] != "row m/z":
                    raise Exception("2. column must be row m/z")
                if header[2] != "row retention time":
                    raise Exception("3. column must be row retention time")

            else:
                try:
                    conn = open(self.output_path + file + "/table-h.csv", "r")
                    text = conn.readlines()
                    conn.close()
                except:
                    print(self.output_path + file + "/table-h.csv")
                    raise Exception("Problem with names of tables (no quant_table.csv of table-h.csv)")
                header = text[0].strip().split(",")
                if header[0] != "index":
                    raise Exception("1. column must be row ID")
                if header[1] != "mz":
                    raise Exception("2. column must be row m/z")
                if header[2] != "tr":
                    raise Exception("3. column must be row retention time")

            sample_name_change_id = [-1] * len(self.names_list)
            # getting index of every column
            for id1, sample_name in enumerate(self.names_list):
                for id2, sample_name2 in enumerate(header):
                    if sample_name in sample_name2:
                        sample_name_change_id[id1] = id2
            if -1 in sample_name_change_id:
                raise Exception("Can't continue: not all names from names_list are present")

            for one_line in text[1:]:
                one_line = one_line.strip().split(",")
                exp_line = [-1, float(one_line[1]), float(one_line[2]), file, int(one_line[0])]
                exp_line += [float(one_line[sample_name_change_id[i]]) for i in range(len(self.names_list))]
                table.append(exp_line)
        self.table = table
        return table

    def add_new_indexes(self, mz_tol=0.02, tr_tol=0.25):
        """
        It creates new indexes for features that are enough far apart.
        :param mz_tol:
        :param tr_tol:
        :return:
        """
        current_id = 0
        merged_counter = 0
        self.table = sorted(self.table, key=lambda x: self.folder_list.index(x[3]) * 1000 + x[1] + x[2] * 0.01)
        for id1, line in enumerate(self.table):
            # For loop in every row
            for id2, line2 in enumerate(self.table[:id1]):
                # for loop for all previous rows that already have set up indices
                if abs(line[1] - line2[1]) < mz_tol and abs(line[2] - line2[2]) < tr_tol:
                    # This means that we have found something that matches previous rows and we can give it same index
                    self.table[id1][0] = line2[0]
                    merged_counter += 1
                    break
            else:
                # None of the previous rows had similar mz and tr so we have to give it new index
                self.table[id1][0] = current_id
                current_id += 1

        print(merged_counter)

    def export_filtered_table(self, rewrite_folder_order=None, sep=",", check_if_same=False):
        if check_if_same:
            save_path = self.output_path + "/table_merged2.csv"
        else:
            save_path = self.output_path + "/table_merged.csv"
        if rewrite_folder_order:
            self.folder_list = rewrite_folder_order
        table_reodered = sorted(self.table, key=lambda x: self.folder_list.index(x[3]) * 1000 + sum(x[5:]) * 1E-5)
        table_exp = []

        added_features = set()
        for line in table_reodered:
            if line[0] not in added_features:
                table_exp.append(line)
                added_features.add(line[0])
        table_exp = sorted(table_exp, key=lambda x: x[1] + x[2] * 0.01)
        self.table_exp = table_exp
        conn = open(save_path, "w", encoding="UTF-8")
        conn.write(sep.join(["index", "mz", "tr", "experiment", "old_index"] + self.names_list) + "\n")
        for line in table_exp:
            conn.write(sep.join([str(i) for i in line]) + "\n")
        conn.close()
        conn2 = open(save_path[:-4] + "_non_filtered.csv", "w", encoding="UTF-8")
        conn2.write(sep.join(["index", "mz", "tr", "experiment", "old_index"] + self.names_list) + "\n")
        table_reodered = sorted(self.table, key=lambda x: x[0] * 1000 + sum(x[5:]) * 1E-5)
        for line in table_reodered:
            conn2.write(sep.join([str(i) for i in line]) + "\n")
        conn2.close()
        if check_if_same:
            if filecmp.cmp(self.output_path + "/table_merged2.csv", self.output_path + "/table_merged2.csv"):
                os.remove(self.output_path + "/table_merged2.csv")
            else:
                print("merged tables do not match!! Please manually check if files are same (new version is saved as"
                      "table_merged2.csv)")

        return table_exp

    def export_simca(self, sep=","):
        save_path = self.output_path + "/table_merged_simca.csv"
        conn = open(save_path, "w", encoding="UTF-8")
        conn.write(sep.join(["identifier"] + self.names_list) + "\n")
        if not self.table_exp:
            raise Exception("For export you first need to generate table_exp!!")
        for line in self.table_exp:
            conn.write(f"V_{line[0]}_{line[1]:.4f}_{line[2]:.1f}_{line[3]}/{line[4]}" + sep)
            conn.write(sep.join([str(i) for i in line[5:]]) + "\n")
        conn.close()

    def export_ms(self, index_list=tuple(), plot_pictures=True, limit_mass=tuple()):
        # reading spectra at the start
        export_folder = self.output_path + "/sirius/"
        if not os.path.isdir(export_folder):
            os.mkdir(export_folder)
        for index, val in enumerate(self.folder_list):
            print(f"reading file: {val}")
            folder = self.output_path + self.folder_list[index]
            if self.experiment_object_list[index]:  # MZmine
                self.experiment_object_list[index] = read_mgf(folder + "/MS2_spectra.mgf")
            else:
                obj = mzproject.class_file.MzProject(self.ionization)
                conn = open(folder + "/Experiment_info.txt")
                data = eval(conn.read())
                conn.close()
                if self.ionization == "pos":
                    dep.input_path = mzproject.paths.input_path_pos
                obj.add_files_speed(data[3], limit_mass=limit_mass)
                obj.add_aligned_dict(folder, "table")
                self.experiment_object_list[index] = obj
        self.table = sorted(self.table, key=lambda x: x[0] * 1000 + sum(x[5:]) * 1E-5)
        # Export of table that is exported
        conn2 = open(self.output_path + "/table_final.csv", "w", encoding="UTF-8")
        conn2.write(
            ",".join(["index", "mz", "tr", "experiment", "old_index", "VIP", "cov", "cor"] + self.names_list) + "\n")
        for line in self.table:
            if line[0] in index_list:
                mid_list = ["-", "-", "-"]
                line_name = f"V_{line[0]}_{line[1]:.4f}_{line[2]:.1f}_{line[3]}/{line[4]}"
                if self.table_VIP is not None:
                    xx = np.argwhere(line_name == self.table_VIP["name"])
                    if len(xx) > 0:
                        mid_list[0] = self.table_VIP["VIP"][xx[0][0]]
                if self.table_s is not None:
                    xx = np.argwhere(line_name == self.table_s["name"])
                    if len(xx) > 0:
                        mid_list[1] = self.table_s["x"][xx[0][0]]
                        mid_list[2] = self.table_s["y"][xx[0][0]]
                line = line[:5] + mid_list + line[5:]
                conn2.write(",".join([str(i) for i in line]) + "\n")
        conn2.close()

        simple_plot_set = set()
        plotting_obj = None
        plotting_obj_project_name = None
        for index2, i in enumerate(self.experiment_object_list):
            if type(i) != dict:
                # Searching for python object to get to plot pictures
                plotting_obj = i
                plotting_obj_project_name = self.folder_list[index2]
                break
        if plot_pictures:
            if not self.group_names:
                raise Exception("You have to group files in order to get pictures!!!")
            if len(self.group_names) > 10:
                pallete = plt.cm.tab20
            else:
                pallete = plt.cm.tab10
            colors = iter([pallete(i) for i in range(len(self.group_names))])
            d_1 = dict()
            for group in self.group_names:
                d_1[group] = next(colors)
            rewrite_c = dict()
            for name in self.names_list:
                for key, value in d_1.items():
                    if key in name:
                        if "blank" in name.lower():
                            linestyle = "dashed"
                        else:
                            linestyle = "solid"
                        rewrite_c[name] = (linestyle, value)
                        break
                else:
                    raise Exception(f"No group for {name}")
        previous_index = None
        export_object = Export_ms(100)
        previous_name = ""
        bar = progressbar.ProgressBar(max_value=len(self.table))
        bar.update(0)
        for zz, feature in enumerate(self.table):
            bar.update(zz)
            current_index = feature[0]
            if plot_pictures and (feature[0] not in simple_plot_set) and (
                    len(index_list) == 0 or current_index in index_list):
                # See if this wor should be plotted or not
                simple_plot_set.add(feature[0])
                dep.output_path = self.output_path
                if not os.path.isdir(dep.output_path + "/feature_pictures/"):
                    os.mkdir(dep.output_path + "/feature_pictures/")
                plotting_obj.extract_SIM(
                    feature[1], filename=self.names_list, mztolerance=0.05,
                    retention_time_range=(round(feature[2] - 0.5, 1), round(feature[2] + 0.5, 1)),
                    save_graph=f"/feature_pictures/{current_index}_{feature[1]:.3f}_{feature[2]:.1f}",
                    rewrite_colors=rewrite_c)

            if index_list and current_index not in index_list:
                # If currend index is not in index list and index list is present then we should skip
                continue
            if current_index != previous_index:
                export_object.export_files(previous_name, export_folder)
                export_object = Export_ms(100)

            file_index = self.folder_list.index(feature[3])
            curr_object = self.experiment_object_list[file_index]
            spectra = []
            if type(curr_object) == dict:  # File from MZmine
                spectra_raw = curr_object.get(feature[4])
                if spectra_raw is not None:
                    for specter_line in spectra_raw:
                        specter_line[0] = np.array(specter_line[0])
                        if len(specter_line[0]) > 0:
                            specter_line[0] = specter_line[0][specter_line[0][:, 0] < feature[1] + 30]
                            if sum(specter_line[0][:, 1]) != 0:
                                # specter_line[0][:, 1] /= max(specter_line[0][:, 1]) / 100
                                spectra.append(specter_line)
                if len(spectra) > 0:
                    export_object.add_feature(
                        f"{feature[0]}-{feature[3]}-{feature[4]}",
                        feature[1], self.ionization, feature[2], 0, spectra, 50
                    )
            else:  # Project from python
                if plot_pictures:
                    dep.output_path = self.output_path
                    curr_object.plot_from_index(feature[4], save_graph=True, graph_path_folder="feature_pictures_t/",
                                                show_plot=False)
                    plt.close('all')
                peak_data_line = curr_object.aligned_dict["peak_data"][
                    curr_object.aligned_dict["peak_data"]["index"] == feature[4]]
                if len(peak_data_line) != 1:
                    raise Exception("Problem with table!!")

                scan_index_list = [[i, i.split("__")[0]] for i in peak_data_line["scans"][0].split("|") if
                                   i.count("__") == 1]
                for j in range(len(scan_index_list)):
                    filename = scan_index_list[j][1]
                    index = scan_index_list[j][0]
                    specter = curr_object.peaks[filename][index]
                    if len(specter) > 0:
                        specter = specter[specter[:, 0] < feature[1] + 30]
                        if sum(specter[:, 1]) != 0:
                            collision_energy = curr_object.scans.loc[index]["tempenergy"]
                            # specter[:, 1] /= max(specter[:, 1]) / 100
                            spectra.append([specter, collision_energy, index])
                if len(spectra) > 0:
                    export_object.add_feature(
                        f"{feature[0]}-from{feature[3]}-{feature[4]}",
                        feature[1], self.ionization, feature[2], 0, spectra, 50
                    )

            previous_name = f"feature_{feature[0]}_mz{feature[1]:.3f}_tr{feature[2]:.1f}"
            previous_index = current_index

        export_object.export_files(previous_name, export_folder)
        if plot_pictures:
            self.rename_pictures_names(plotting_obj_project_name)

    def export_averaged_table(self, check_if_same=False):
        export_path = self.output_path
        n_non_h_cols = 5
        group_dict = dict()
        if ".xml" in self.names_list[0]:
            common_suffix_length = 8
        else:
            common_suffix_length = 7

        for i in range(len(self.names_list)):
            curr_name = self.names_list[i].replace(" ", "")[:-common_suffix_length]
            split_curr_name = curr_name.split("_")
            cut_curr_name = "_".join(split_curr_name[:-1])
            if len(split_curr_name) == 1:
                group_dict[curr_name] = [i]
                continue
            elif "-" in split_curr_name[-1]:
                cut_curr_name = curr_name
            elif len(split_curr_name[-1]) == 2:
                cut_curr_name += "_3"
            if cut_curr_name in group_dict:
                group_dict[cut_curr_name].append(i + n_non_h_cols)
            else:
                group_dict[cut_curr_name] = [i + n_non_h_cols]
        print(group_dict)
        group_list = list(group_dict.keys())
        averaged_table = []
        for index, val in enumerate(self.table):
            averaged_table.append(val[:n_non_h_cols])
            for g in group_list:
                values = [val[i] for i in group_dict[g]]
                if values.count(0) > max(0, len(values) - 2):
                    mean = 0
                # TODO: see if values are behaving as as you wish
                else:
                    mean = np.average([i for i in values if i != 0])
                averaged_table[-1].append(mean)
        self.group_names = group_list
        if check_if_same:
            with open(export_path + "averaged_table2.csv", "w", encoding="UTF-8") as conn:
                conn.write(",".join(["index", "mz", "tr", "experiment", "old_index"] + group_list) + "\n")
                for line in averaged_table:
                    conn.write(",".join([str(i) for i in line]) + "\n")
            if filecmp.cmp(export_path + "averaged_table2.csv", export_path + "averaged_table.csv"):
                os.remove(export_path + "averaged_table2.csv")
            else:
                print("averaged tables do not match!! Please manually check if files are same (new version is saved as"
                      "averaged_table2.csv)")
        else:
            with open(export_path + "averaged_table.csv", "w", encoding="UTF-8") as conn:
                conn.write(",".join(["index", "mz", "tr", "experiment", "old_index"] + group_list) + "\n")
                for line in averaged_table:
                    conn.write(",".join([str(i) for i in line]) + "\n")

    def rename_pictures_names(self, project_name):
        file_list = os.listdir(self.output_path + "/feature_pictures_t")
        print(f"project name: {project_name}")
        for filename in file_list:
            filename_splitted = filename.split("_")
            old_index = filename_splitted[0]
            mz = filename_splitted[1].split("=")[1]
            tr = filename_splitted[2].split("=")[1]
            suffix = filename_splitted[3]
            l1 = [i for i in self.table if (i[4] == int(old_index)) and (i[3] == project_name)]
            if len(l1) == 1:
                new_name = f"{l1[0][0]}__{float(mz):.3f}_{float(tr):.1f}_{suffix}"
                os.rename(self.output_path + "/feature_pictures_t/" + filename,
                          self.output_path + "/feature_pictures/" + new_name)
            else:
                print("fffuuuuu", filename, filename_splitted)
                # os.rename(self.output_path + "/feature_pictures_t/" + filename,
                #           self.output_path + "/feature_pictures/" + filename)

    def get_VIP(self, scale=True, log10=True):
        df = pd.DataFrame(self.table, columns=["index", "mz", "tr", "experiment", "old_index"] + self.names_list)
        filter_vals2 = [i for i in df.columns if "xml" in i]
        vals_df = df[filter_vals2]
        Y = np.array([not "blank" in i.lower() for i in vals_df.columns], dtype=int)
        X = vals_df.values.T
        # X = X - X.mean(axis=1)[None].T
        plsr = PLSRegression(2, scale=scale)  # --> scale=False zgleda mogoče celo bolje!
        if log10:
            plsr.fit(np.log10(1 + X), Y)
        else:
            plsr.fit(X, Y)

        colormap = {
            0: '#ff0000',  # Red
            1: '#0000ff',  # Blue
        }
        colorlist = [colormap[c] for c in Y]

        scores = pd.DataFrame(plsr.x_scores_)
        scores.index = vals_df.columns
        ax = scores.plot(x=0, y=1, kind='scatter', s=50, alpha=0.7,
                         c=colorlist, figsize=(6, 6))
        ax.set_xlabel('Scores on LV 1')
        ax.set_ylabel('Scores on LV 2')
        ax.set_title(f"{scale * 'scaled '}{log10 * 'log10'}")
        for n, (x, y) in enumerate(scores.values):
            label = scores.index.values[n]
            ax.text(x, y, label)
        plt.savefig(self.output_path + "/PLS-DA.png")

        vips = vip(vals_df.values.T, Y, plsr)
        vips = vips[np.newaxis].T
        df["VIP"] = vips
        df = df.sort_values("VIP", ascending=False)
        df.to_csv(self.output_path + "/VIP_table.csv")

        return df

    def read_simca_output(self, folder, name_VIP, min_VIP=0, max_features=-1, name_s_plot="", limit_mz=tuple()):
        table, header = f.read_simple_file_list(
            mzproject.paths.IJS_ofline_path + folder + "/" + name_VIP,
            types=[str, float, float], header=True, sep="\t")
        table = [tuple(i) for i in table]
        table = np.array(table, dtype=[("name", "<U100"), ("VIP", float), ("V2", float)])
        if len(limit_mz) == 2:
            mz_array = np.array([float(i.split("_")[2]) for i in table['name']])
            table = table[np.logical_and(limit_mz[0] < mz_array, mz_array < limit_mz[1])]
        plt.close()
        plt.hist(table["VIP"], 30, density=False, histtype='step', cumulative=-1)
        plt.vlines(min_VIP, 0, len(table))
        table2 = table
        if max_features != -1:
            plt.hlines(max_features, 0, max(table["VIP"]))
            table2 = table[:max_features]
        plt.savefig(self.output_path + "/Histogram_VIP.png")
        plt.show()
        table2 = table2[table2["VIP"] > min_VIP]
        indexes = [int(i.split("_")[1]) for i in table2["name"]]
        table_VIP = table
        if name_s_plot:
            table, header = f.read_simple_file_list(
                mzproject.paths.IJS_ofline_path + folder + "/" + name_s_plot,
                types=[str, str, str], header=True, sep="\t")
            table = [tuple(i) for i in table if i[1] != "-0"]
            table = np.array(table, dtype=[("name", "<U100"), ("x", float), ("y", float)])
            if len(limit_mz) == 2:
                mz_array = np.array([float(i.split("_")[2]) for i in table['name']])
                table = table[np.logical_and(limit_mz[0] < mz_array, mz_array < limit_mz[1])]
            plt.scatter(x=table["x"], y=table["y"], marker="x", c=[
                color_map_color(-table_VIP["VIP"][np.argwhere(i["name"] == table_VIP["name"])[0][0]], "viridis", -2, 0)
                for
                i in table], s=1)
            plt.savefig(self.output_path + "/S-plot_colored.png")
            plt.show()
            self.table_s = table
        self.table_VIP = table_VIP

        return indexes


def vip(x, y, model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_

    m, p = x.shape
    _, h = t.shape

    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)

    return vips

mzproject.merge_tables.do_all(r"C:/Users/tinc9/Documents/IJS-offline/Experiment/simulant_neg_low_mz/", file_list,
              ["python_beer_simulant_neg_lowmz"], "neg", True,
              tuple(), "Experiment/simca_results/simca_simulant_neg_lowmz",
                           "General-List_VIP_opls-da_mzLow.txt", "General-List_s-plot_opls-da_mzLow.txt")
def generate_ensemble_list(folder_project, file_list, project_list, polarity, limit_mass=tuple()):
    """
    This function gets folder of a project and information about projects and returns ensemble table that
    merges all methods that generate feature lists as well as 2d table that is suitable as simca input.
    :param folder_project: Path to the folder that keeps project folders and at the same time place where new
    files will be generated
    :param file_list: list of raw files (sample analysis files)
    :param project_list: list of names of folders of projects that we want to merge
    :param polarity: Pos/Neg
    :limit_mass: If you generated files in python with limit_mass you also need to put same tuple here!
    :return: None
    """
    obj = Merge_tables(project_list, file_list, polarity, folder_project)
    obj.generate_table()
    obj.add_new_indexes()
    obj.export_averaged_table()
    obj.export_filtered_table()
    obj.export_simca(",")

def export_ms(folder_project, file_list, project_list, polarity, from_simca=False, interesting_indexes=np.array([]),
              simca_files_tuple=tuple(), limit_mass=tuple()):
    """
    After we exported tables and obtained
    :param folder_project: Path to the folder that keeps project folders and at the same time place where new
    files will be generated
    :param file_list: list of raw files (sample analysis files)
    :param project_list: list of names of folders of projects that we want to merge
    :param polarity: Pos/Neg
    :return: None
    """
    obj = Merge_tables(project_list, file_list, polarity, folder_project)
    obj.generate_table()
    obj.add_new_indexes()
    obj.export_averaged_table(check_if_same=True)
    obj.export_filtered_table(check_if_same=True)

    if from_simca:
        folder_simca, file_vip_simca, file_s_simca = simca_files_tuple
        interesting_indexes = obj.read_simca_output(folder_simca, file_vip_simca, 1, 100, file_s_simca, limit_mass)
        obj.export_ms(interesting_indexes, limit_mass=limit_mass)
    else:
        obj.export_ms(interesting_indexes, limit_mass=limit_mass)
    return obj



def do_all(path1, file_list, project_list, polarity, export_ms=False, interesting_indexes=tuple(),
           folder_simca="", file_vip_simca="", file_s_simca="", limit_mass=tuple()):
    obj = Merge_tables(project_list, file_list, polarity, path1)
    obj.generate_table()
    obj.add_new_indexes()
    obj.export_averaged_table()
    obj.export_filtered_table()
    obj.export_simca(",")

    if export_ms:
        if not interesting_indexes:
            interesting_indexes = obj.read_simca_output(folder_simca, file_vip_simca, 1, 100, file_s_simca, limit_mass)
            if limit_mass:
                pass
        else:
            df = obj.get_VIP(True, True)
            v1 = df[["index", "VIP"]].values
            interesting_indexes = v1
        obj.export_ms(interesting_indexes, limit_mass=limit_mass)
    return obj



if __name__ == "__main__":
    pass
