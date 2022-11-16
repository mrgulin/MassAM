import numpy as np
import matplotlib.pyplot as plt
import mzproject.dependencies as dep
import pandas as pd
from pyteomics import mzxml
import os
from lxml import etree as etree_lxml
import base64
import zlib
from scipy.signal import savgol_filter
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.cluster import AgglomerativeClustering
import typing
import numba
import warnings
from dataclasses import dataclass, field
import mzproject.logger_file

m_p = 1.00727647  # Da, g/mol, amu
np.seterr(all='raise')
parameters = {
    # find_peak_function for splitting SIM into peaks
    "fpf_ascending": 0.75,
    "fpf_descending": 0.95,

    # Isotopic ratio parameters
    "isotopic_ratio_mz_tol": 0.3,

    # generate_table/parameters for splitting peaks  in 2 separate peaks
    "peak_split_interval_diff": 0.15,
    "peak_split_max_h_diff": 0.37,

    # find_peak2
    "time_interval": 0.5,  # Algorithm is only looking for data in this range (peak must end in this interval)
    "delta_mz1": 0.05,  # tolerance to interpret peaks as given m/z
    # Peaks become less jagged
    "rt_tolerance": 1.1,  # how much can mass be away from detected peak to assign it to this peak. In theory it could
    # also be less than 1 to make algorithm stricter. It is expressed as t - t-avg / w
    "fp2_noise": 1.8,  # To find out if specter is noise

    # merge_features
    "merge_features_mz_tol": 0.05,
    "merge_features_rt_tol": 1,

    # filter_constant_ions
    "mz_tolerance": 0.015,  # Maximum mz that 2 peaks can be apart to be treated as constant interference
    "min_len_peaks_per_file": 15,  # How many peaks per file must be inside mz_tolerance to be treated as same peak

    # merge_duplicate_rows

    # merge_duplicate_rows
    "duplicates_tr_tolerance": 0.1,
    "duplicates_mz_tolerance": 0.03,

    "mz_weight_exponent": 1.5
}


# @numba.experimental.jitclass([('scans', numba.types.string), ('noise', numba.types.string),
#                               ('far', numba.types.string),
#                               ('noise_ratio1', numba.types.float64), ('noise_ratio2', numba.types.int64),
#                               ('comment', numba.types.string)])
@dataclass
class RowPeakDataClass:
    def __init__(self):
        self.scans: str = ""
        self.noise: str = ""
        self.far: str = ""
        self.noise_ratio1: float = 1e10
        self.noise_ratio2: float = 0
        self.comment: str = ""


class FindPeakReturnClass:
    def __init__(self):
        self.avg_mz = np.nan
        self.avg_mz_s = np.nan
        self.tr_min = np.nan
        self.tr_high = np.nan
        self.tr_max = np.nan
        self.peak_h = 0
        self.peak_area = 0
        self.avg_mz_list = np.nan
        self.time_list = np.nan
        self.intensity_list = np.nan
        self.comment = np.nan
        self.start_index = 0
        self.end_index = 1
        self.specter_index = []
        self.isotopes = np.nan
        self.nan_object = True

    def return_aligned_list_parameters(self, med_tr, med_mz):
        return [self.peak_area, self.peak_h, self.tr_high - med_tr, self.avg_mz - med_mz, self.tr_max - self.tr_min]

    def return_empty(self, time_list, intensity_list, comment):
        self.time_list = time_list
        self.intensity_list = intensity_list
        self.comment = comment
        self.nan_object = True
        return self

    def __getitem__(self, item):
        warning.warn("THIS HAS TO BE DEPRECATED!!!!!" + str(item))
        return [
            (self.avg_mz, self.avg_mz_s),
            (self.tr_min, self.tr_high, self.tr_max),
            (self.peak_h, self.peak_area),
            self.avg_mz_list,
            self.time_list,
            self.intensity_list,
            self.comment,
            self.start_index,
            self.end_index,
            self.specter_index,
            self.isotopes
        ][item]

    def generate_tr_tuple(self):
        return self.tr_min, self.tr_high, self.tr_max


@dataclass
class SplitFeatureDataClass:
    raw_data: typing.Dict[typing.Tuple[str, int], FindPeakReturnClass] = field(default_factory=dict)
    feature_num: int = 0
    list_retention_time_split: typing.List[typing.Tuple[float]] = field(default_factory=list)
    # Info about MS2 spectra for plotting
    ms2_precursor_mz_dict: typing.Dict[typing.Tuple[str, int], FindPeakReturnClass] = field(default_factory=dict)
    ms2_tr_dict: typing.Dict[typing.Tuple[str, int], FindPeakReturnClass] = field(default_factory=dict)
    ms2_num: typing.Dict[typing.Tuple[str, int], FindPeakReturnClass] = field(default_factory=dict)

    med_mz: typing.List[float] = field(default_factory=list)
    med_tr: typing.List[float] = field(default_factory=list)

    def add_one(self, file_name, peak_index, peak_obj, precursor_mz, tr, num):
        self.raw_data[(file_name, peak_index)] = peak_obj
        self.ms2_precursor_mz_dict[(file_name, peak_index)] = precursor_mz
        self.ms2_tr_dict[(file_name, peak_index)] = tr
        self.ms2_num[(file_name, peak_index)] = num


class MzProject:
    def __init__(self, stream_level=25, file_level=20):
        self.logger = mzproject.logger_file.general_logger  # Generate logger for the class
        mzproject.logger_file.logging.captureWarnings(True)
        mzproject.logger_file.stream_handler.setLevel(stream_level)
        mzproject.logger_file.general_handler.setLevel(file_level)

        self.logger.log(20, f"Initializing object for feature table generation.")
        # this is with pytheomics
        # https://gitlab.isas.de/lifs-public/lipidxplorer/-/
        # blob/af7e8f280c145763426e6a04f7ca26cc9c073e05/notebooks/0.1-jm-MS_reader.ipynb
        self.filename = []
        self.peaks = dict()
        self.mergedMS2scans = []  # Type: list
        self.scans = None
        self.scans2 = None
        self.int_st = {"16dBPA": {"mz": 244.215458, "tr": 5}, "carbamazepine-d10": {"mz": 246.15773, "tr": 5}}
        # https://pubchem.ncbi.nlm.nih.gov/compound/16212886#section=Molecular-Formula
        # https://pubchem.ncbi.nlm.nih.gov/compound/Carbamazepine-D10#section=Synonyms
        self.parameters = parameters
        # "peak_data": dep.get_dtype((0, 0, 0))
        self.aligned_dict: typing.Dict[str, typing.Union[np.array, list, dict]] = {
            "h": [], "A": [], "peak_data": [], "delta_time": [], "delta_mz": [], "peak_length": [], "M05": [], "M1": [],
            "M15": [], "M2": []}
        self.dict_plot_index = dict()
        self.header = None  # This is header of peak_data array
        self.matched_list = []  # List reserved for matching with suspect_list
        self.averaged_aligned_dict = dict()  # same as aligned_dict but for averaged columns. Here goes calculation of
        self.header_averaged = []  # similar to averaged_aligned_dict

    def __call__(self, *args, **kwargs):
        end = self.write_parameters()
        return "This is object that is used to generate aligned feature table. This object is using next files:\n" \
               f"{', '.join(self.filename)}" + end

    def write_parameters(self):
        self.logger.log(20, f"Exporting parameters")
        log_string = "\nParameters:\n"
        for key, value in self.parameters.items():
            log_string += f"\t\t\t{key} = {value}\n"
        log_string += "\n"
        return log_string

    def set_parameter(self, name, value):
        self.logger.log(20, f"Setting parameter'{name}' from {self.parameters[name]}  to {value}.")
        self.parameters[name] = value

    def __str__(self):
        return "mzproject object"

    def __repr__(self):
        return "mzproject object"

    # region opening_modules
    def add_files(self, filename_list: list, mz_tr_tuple: tuple = (), limit_mass=tuple()):
        """
        This method takes filename_list and generates scans dataframe, peaks dictionary, filename list, ...
        :param mz_tr_tuple: tuple in form of ((mz1, tr1, ANYTHING), ...)
        :param self:
        :param filename_list: list of strings (e.g. output from f.get_files)
        :param limit_mass: (min_mz, max_mz), else there is no limit
        """
        old_filename = []
        if self.filename:
            old_filename = self.filename
            self.filename = []
        first = True
        self.logger.log(20, f'| Starting to add {len(filename_list)} files')
        # It reads data trough mzxml.read and it transforms it into dataframe
        file_counter = 0
        for filename in filename_list:  # TODO: parallelize
            self.filename.append(filename)
            scans = []
            self.peaks[filename] = dict()
            self.logger.log(18, "|| Reading ", dep.input_path + filename, f" ({file_counter}/{len(filename_list)})")
            file_counter += 1
            with mzxml.read(dep.input_path + filename) as reader:

                for item in reader:

                    if "collisionEnergy" in item:  # This is a MS2 spectrum
                        precursor_scan_num = item['precursorMz'][0]['precursorScanNum']
                        precursor_intensity = item['precursorMz'][0]['precursorIntensity']
                        activation_method = item['precursorMz'][0]['activationMethod']
                        precursor_mz = item['precursorMz'][0]['precursorMz']
                        temp_energy = item['collisionEnergy']
                    else:
                        temp_energy = activation_method = precursor_scan_num = np.nan
                        precursor_mz = precursor_intensity = np.nan

                    peak_dict_name = filename + "__" + item['index']

                    if mz_tr_tuple and not np.isnan(precursor_mz):
                        close = False
                        for line2 in mz_tr_tuple:
                            mz_j = line2[0]
                            tr_j = line2[1]
                            if abs(mz_j - precursor_mz) < 0.5 and abs(item['retentionTime'] - tr_j) < 1:
                                close = True
                        if not close:
                            continue

                    if limit_mass and not np.isnan(precursor_mz):
                        if not (limit_mass[0] < precursor_mz < limit_mass[1]):
                            continue

                    row = (
                        peak_dict_name, filename + "__" + item['num'], item['msLevel'], item['peaksCount'],
                        item['retentionTime'], item.get('lowMz'), item.get('highMz'),
                        item['basePeakMz'], item['basePeakIntensity'], item['totIonCurrent'],
                        temp_energy, activation_method, precursor_scan_num, precursor_mz, precursor_intensity)
                    scans.append(row)
                    i = item['intensity array']
                    m = item['m/z array']
                    if len(i) == 0:
                        i = [0]
                        m = [-1]

                    self.peaks[filename][peak_dict_name] = np.array([m, i]).transpose()
                scans_df = pd.DataFrame(scans, columns=['index', 'num', 'msLevel', 'peaksCount', 'retentionTime',
                                                        'lowMz', 'highMz', 'basePeakMz', 'basePeakIntensity',
                                                        'totIonCurrent', 'tempenergy', 'activationMethod',
                                                        'precursorScanNum', 'precursorMz', 'precursorIntensity'])
                scans_df.set_index('index', inplace=True)
                scans_df["filename"] = filename

                if first:
                    self.scans = scans_df  # data about scans
                    first = False
                else:
                    self.scans = self.scans.append(scans_df)
        self.scans2 = self.scans[self.scans.msLevel == 2].copy()
        self.scans2.loc[:, "keep"] = True
        self.scans2.reindex(copy=False)
        if old_filename and old_filename != self.filename:
            str1 = "Problem! Files that are read are not same as those that were in imported aligned_dict!"
            str1 += f"files from aligned_dict: {old_filename}\n imported files: {self.filename}"
            self.logger.log(40, "|| " + str1)
        self.logger.log(18, "|| files read")

    def add_files_speed(self, filename_list: list, mz_tr_tuple: tuple = (), limit_mass=tuple()):
        """
        This method takes filename_list and generates scans dataframe, peaks dictionary, filename list, ...
        :param mz_tr_tuple: tuple in form of ((mz1, tr1, ANYTHING), ...)
        :param self:
        :param filename_list: list of strings (e.g. output from f.get_files)
        :param limit_mass: (min_mz, max_mz); if left there is no limit
        """
        self.logger.log(20,
                        f"Adding files with add_files_speed; filename_list={filename_list}, mz_tr_tuple]{mz_tr_tuple},"
                        f" limit_mass={limit_mass}")
        old_filename = []
        if self.filename:
            old_filename = self.filename
            self.filename = []
        first = True
        # It reads data trough mzxml.read and it transforms it into dataframe
        file_counter = 0
        self.logger.log(20, f'| Starting to add {len(filename_list)} files with FAST algorithm')
        for filename in filename_list:  # TODO: parallelize
            self.filename.append(filename)
            scans = []
            self.peaks[filename] = dict()
            self.logger.log(18, f"|| Reading {dep.input_path + filename} ({file_counter}/{len(filename_list)})")
            file_counter += 1
            with open(dep.input_path + filename, "rb") as xml:
                xml_as_bytes = xml.read()
                tree = etree_lxml.fromstring(xml_as_bytes)
                for one_scan in tree[0].getchildren():
                    if "scan" not in one_scan.tag:
                        continue

                    # It actually is a scan
                    attributes = one_scan.attrib
                    if attributes["msLevel"] == "2":
                        precursor_mz_scan = one_scan[0]
                        specter = decode_zlib(one_scan[1].text)
                        precursor_scan_num = float(precursor_mz_scan.attrib['precursorScanNum'])
                        precursor_intensity = float(precursor_mz_scan.attrib['precursorIntensity'])
                        activation_method = precursor_mz_scan.attrib['activationMethod']
                        precursor_mz = float(precursor_mz_scan.text)
                        temp_energy = float(attributes['collisionEnergy'])

                    elif attributes["msLevel"] == "1":
                        specter = decode_zlib(one_scan[0].text)
                        temp_energy = activation_method = precursor_scan_num = np.nan
                        precursor_mz = precursor_intensity = np.nan

                    peak_dict_name = filename + "__" + attributes['num']
                    retention_time = float(attributes["retentionTime"][2:-1]) / 60.
                    if mz_tr_tuple and np.isnan(precursor_mz):
                        close = False
                        for line2 in mz_tr_tuple:
                            mz_j = line2[0]
                            tr_j = line2[1]
                            if abs(mz_j - precursor_mz) < 0.5 and abs(retention_time - tr_j) < 1:
                                close = True
                        if not close:
                            continue
                    if limit_mass and not np.isnan(precursor_mz):
                        if not (limit_mass[0] < precursor_mz < limit_mass[1]):
                            continue

                    row = (
                        peak_dict_name, filename + "__" + attributes['num'], attributes['msLevel'],
                        attributes['peaksCount'], retention_time, attributes.get('lowMz'),
                        attributes.get('highMz'),
                        attributes['basePeakMz'], attributes['basePeakIntensity'], attributes['totIonCurrent'],
                        temp_energy, activation_method, precursor_scan_num, precursor_mz, precursor_intensity)
                    scans.append(row)
                    self.peaks[filename][peak_dict_name] = specter

                type_scan = [('index', '<U50'), ('num', '<U50'), ('msLevel', np.int8), ('peaksCount', np.float64),
                             ('retentionTime', np.float), ('lowMz', np.float64), ('highMz', np.float64),
                             ('basePeakMz', np.float64), ('basePeakIntensity', float), ('totIonCurrent', float),
                             ('tempenergy', np.float64), ('activationMethod', '<U10'), ('precursorScanNum', np.float64),
                             ('precursorMz', np.float64), ('precursorIntensity', np.float64)]
                scans_df = pd.DataFrame(np.array(scans, dtype=type_scan))
                scans_df.set_index('index', inplace=True)
                scans_df["filename"] = filename

                if first:
                    self.scans = scans_df  # data about scans
                    first = False
                else:
                    self.scans = self.scans.append(scans_df)
        self.scans2 = self.scans[self.scans.msLevel == 2].copy()
        self.scans2.loc[:, "keep"] = True
        self.scans2.reindex(copy=False)
        if old_filename and old_filename != self.filename:
            str1 = "Problem! Files that are read are not same as those that were in imported aligned_dict!"
            str1 += f"files from aligned_dict: {old_filename}\n imported files: {self.filename}"
            self.logger.log(40, "|| " + str1)
        self.logger.log(18, "|| files read")

    def add_aligned_dict(self, source_folder: str, root_name: str, extension_list=tuple(), sep: str = ",",
                         common_columns: int = 10):
        """
        Method for importing aligned_dict instead of calculating one. I have to also
        :param self: mzproject object
        :param source_folder: !!Absolute path to folder where aligned dict are located
        :param root_name: Common name of table names
        :param extension_list: list of string of names of tables (e.g. ["A", "delta_time", "h", "M15", ...]. Leave empty
        if you want to import everything
        :param sep: separator in tables
        :param common_columns: columns that belong to peak_data
        """
        self.logger.log(20, f"Importing aligned dict with add_aligned_dict: source_folder={source_folder},"
                            f" root_name={root_name}, table_types={extension_list}, sep='{sep}',"
                            f" common columns={common_columns}")
        self.header = None  # to reset old header if it has been there
        old_filename = []
        if self.filename:
            old_filename = self.filename
            self.filename = []

        if len(extension_list) == 0:
            extension_list = [i for i in list(self.aligned_dict) if "peak_data" not in i]
        self.aligned_dict = dict()

        # self.dict_plot_index:
        path1 = f"{source_folder}/{root_name}-index_dict.csv"
        if os.path.isfile(path1):
            conn = open(path1)
            lines = conn.readlines()
            conn.close()
            for line in lines:
                line = line.strip().split(" ")
                if line == "":
                    continue
                self.dict_plot_index[int(line[0])] = (int(line[1]), int(line[2]))

        # aligned_dict
        for extension in extension_list:
            self.logger.log(16, f"|| Reading file {source_folder}/{root_name}-{extension}.csv")
            conn = open(f"{source_folder}/{root_name}-{extension}.csv")
            lines = conn.readlines()
            conn.close()
            header = lines[0]
            temp_aligned_list = []
            for line in lines[1:]:
                line_list = np.array(line.strip().split(sep)[common_columns:])
                temp_aligned_list.append(line_list)
            self.aligned_dict[extension] = np.array(temp_aligned_list, dtype=float)

            if not ((header == self.header) or (self.header is None)):
                self.logger.log(40, f"|| Problem: Header do not match \nheader 1: {self.header}\nheader 2: {header}")
            self.header = header
            # For peak_data
            if extension_list[0] == extension:
                temp_aligned_list = []
                m_string = [0, 0, 0, 0]
                m_index = [3, 4, 5, 8]
                for line in lines[1:]:
                    line_list = tuple(line.strip().split(sep)[:common_columns])
                    # type_list = [int, float, float, str, str, str, float, int, str, float]
                    # line_list = (type_list[i](line_list[i]) for i in range(len(type_list)))
                    temp_aligned_list.append(line_list)
                    for i in range(len(m_index)):
                        if len(line_list[m_index[i]]) > m_string[i]:
                            m_string[i] = len(line_list[m_index[i]])
                dtype = dep.get_dtype(m_string)
                # index,mz,tr,scans,noise,too_far,noise_ratio1,noise_ratio2,comment,M_plus_1
                self.aligned_dict["peak_data"] = np.array(temp_aligned_list, dtype=dtype)
        header1 = self.header.strip().replace(" ", "").split(sep)
        self.header = header1[:common_columns]
        new_filename = header1[common_columns:]
        new_filename = [i.replace(" ", "") for i in new_filename]
        if old_filename and new_filename != old_filename:
            str1 = "|| Files that are read are not same as those that were in imported raw files!"
            str1 += f"files from aligned_dict: {old_filename}\n imported files: {new_filename}"
            self.logger.log(40, str1)
        self.filename = new_filename
        self.logger.log(16, f"Finished reading table: header={self.header}, filename={self.filename}")

    # endregion

    # region table modules
    def generate_table(self, save_graph: bool = True, limit_iteration: int = 0, graph_path_folder: str = "graphs/",
                       row_comment=(), force: bool = False):
        """
        This is heart of generation of aligned table. It takes data from self.mergedMS2scans and it finds peak,
        potentially splits them in multiple peaks and calculates all parameters. It can also plot graph or save it.
        :param force: write in folder even if there is already data
        :param save_graph: saves graph in output_path + graph_path_folder
        :param limit_iteration: For testing function. If limit_iteration is 0 then there is no limit
        :param graph_path_folder: Something like "graphs/" it is relative to output_path
        :param row_comment: This gets into comments in aligned list and also in legend in graphs
        """
        self.logger.log(20, f"| Starting to generate a table with generate_table: save_graph={save_graph},"
                            f" limit_iteration={limit_iteration}, force={force}")
        self.header = ["index", "mz", "tr", "scans", "noise", "too_far", "noise_ratio1",
                       "noise_ratio2", "comment", "M_plus_1"]

        if save_graph:  # Generate folder and/or prevent overwriting it
            if os.path.isdir(dep.output_path + graph_path_folder):
                if len(os.listdir(dep.output_path + graph_path_folder)) != 0 and not force:
                    raise OSError("given folder is not empty!")
            else:
                os.makedirs(dep.output_path + graph_path_folder)
            if graph_path_folder[-1] != "/":
                graph_path_folder += "/"
        fig_gt, ax_gt = make_graph_grid()
        # Dictionary for searching index of filenames
        reverse_name_dict = dict()
        for i in range(len(self.filename)):
            reverse_name_dict[self.filename[i]] = i
        aligned_table_index = 0

        for main_group_index, mass_feature in enumerate(self.mergedMS2scans):
            # mass_feature is list of different MS2 spectra from different files that all have similar tr and mz.
            # This means that maybe some of the groups have to be additionally split
            self.logger.log(16, f"|| starting new group with {len(mass_feature)} spectra with index {main_group_index}")
            split_peak_group_obj = self.align_peak_with_ms(mass_feature)

            for subpeak_index in range(split_peak_group_obj.feature_num):
                # Maybe initial guess that all of the ms2 spectra belong to the same feature was wrong. This is why we
                # have to split the group into multiple subgroups. Therefore we get for loop. But most of the time there
                # is only one element in the list_retention_time_split.
                comment1 = ""
                if row_comment:
                    comment1 = row_comment[main_group_index]

                self.align_peak_without_ms(split_peak_group_obj, subpeak_index,
                                           comment1, aligned_table_index, reverse_name_dict)
                str1 = f'||| mz={split_peak_group_obj.med_mz[-1]:.4f}, tr={split_peak_group_obj.med_tr[-1]:.2f}, '
                str1 += f"feature_number={aligned_table_index}{('/' + str(limit_iteration)) * (limit_iteration > 0)}, "
                str1 += f"self.mergedMS2scans index = {main_group_index}/{len(self.mergedMS2scans)},"
                str1 += f" subpeak_index = {subpeak_index}"
                self.logger.log(18, str1)

                self.dict_plot_index[aligned_table_index] = (main_group_index, subpeak_index)

                if save_graph:  # or plot_this_bool:
                    plot_data = [value for key, value in split_peak_group_obj.raw_data.items() if
                                 key[1] == subpeak_index]
                    plot_names = [key[0] for key, value in split_peak_group_obj.raw_data.items() if
                                  key[1] == subpeak_index]
                    plot_graph(plot_data, plot_names, dep.output_path + graph_path_folder,
                               f"mz={split_peak_group_obj.med_mz[-1]:.4f}_tr={split_peak_group_obj.med_tr[-1]:.2f}",
                               split_peak_group_obj=split_peak_group_obj,
                               subpeak_index=subpeak_index, plt_object=(fig_gt, ax_gt),
                               start_comment=str(aligned_table_index) + "_")
                aligned_table_index += 1
                if aligned_table_index > limit_iteration and bool(limit_iteration):
                    break
            else:  # This break continue manages to break outer for loop from inner loop
                continue
            break
        keys_aligned_dict = [i for i in list(self.aligned_dict) if i != "peak_data"]
        for i in keys_aligned_dict:
            self.aligned_dict[i] = np.array(self.aligned_dict[i], dtype=float)

        # Transform normal 2d list self.aligned_dict["peak_data"] into np.array
        m_string = [max([len(j[i]) for j in self.aligned_dict["peak_data"]]) for i in [3, 4, 5, 8]]
        dtype = dep.get_dtype(
            m_string)  # get_dtype() and m_string create list of types that are needed for structured array
        self.aligned_dict["peak_data"] = np.array(self.aligned_dict["peak_data"], dtype=dtype)

    def get_correct_subgroup(self, list_retention_time_split, returned_peak_split) -> typing.Tuple[int, list]:
        """
        Helper function for align_peak_with_ms
        :param returned_peak_split:
        :param list_retention_time_split:
        :return:
        """
        if len(list_retention_time_split) == 0:  # there is no peak in list_retention_time_split yet
            list_retention_time_split.append(returned_peak_split.generate_tr_tuple())
            key = 0
            return key, list_retention_time_split
        time_interval = returned_peak_split.generate_tr_tuple()
        # I have to get index of list_retention_time_split, dict_raw_data
        best_feature = (None, 100)
        best_feature_2 = (None, 100)
        for v in range(len(list_retention_time_split)):
            distance1 = max(list_retention_time_split[v][0], time_interval[0]) - min(
                list_retention_time_split[v][2], time_interval[2])
            distance2 = abs(list_retention_time_split[v][1] - time_interval[1])
            if distance1 < best_feature[1]:
                best_feature = (v, distance1)
            if distance2 < best_feature_2[1]:
                best_feature_2 = (v, distance2)

        if best_feature[1] > self.parameters["peak_split_interval_diff"] or best_feature_2[1] > \
                self.parameters["peak_split_max_h_diff"]:
            self.logger.log(20, f"|||| Splitting feature: {round(best_feature[1], 5)} "
                                f"{self.parameters['peak_split_interval_diff']} {round(best_feature_2[1], 5)}"
                                f" {self.parameters['peak_split_max_h_diff']}")
            # Features are not the same so I need to create new one
            list_retention_time_split.append(time_interval)
            key = len(list_retention_time_split) - 1
        else:
            key = best_feature[0]
        return key, list_retention_time_split

    def align_peak_with_ms(self, mass_feature: np.ndarray) -> SplitFeatureDataClass:
        return_obj = SplitFeatureDataClass()
        list_retention_time_split = []  # [(t_min, t_highest, t_max), (...), ...] most of the time only 1 element

        # Special case when we want to set times and and retention times by ourselves! We will skip for loop
        if mass_feature[0]['filename'] == 'FROM_TABLE':
            dictionary_raw_data["not_file"] = {0: [(mass_feature[0]['precursorMz'], 's_mz'),
                                                   ('tr_min', mass_feature[0]['retentionTime'], 'tr_max')]}
            return_obj[('not_file',)] = [(mass_feature[0]['precursorMz'], 's_mz'),
                                         ('tr_min', mass_feature[0]['retentionTime'], 'tr_max')]
            warnings.warn('Did not finish this part')
            # return_obj.list_retention_time_split = ['split_time_1_element']
            return_obj.feature_num = 1
            return return_obj

        for filename_sub in np.unique(mass_feature['filename']):
            ms2_from_same_file = mass_feature[mass_feature['filename'] == filename_sub]
            # select all all mass spectra from same file and find peak from those mass spectra
            returned_peak = self.find_peak(float(np.mean(ms2_from_same_file['precursorMz'])),
                                           ms2_from_same_file['retentionTime'], filename_sub)
            # Now we get returned peak that could have more than one element. We have to go over all elements
            for returned_peak_single in returned_peak:
                if returned_peak_single.nan_object:  # Didn't find peak
                    continue
                key, list_retention_time_split = self.get_correct_subgroup(list_retention_time_split,
                                                                           returned_peak_single)
                specter_index = returned_peak_single.specter_index
                return_obj.add_one(filename_sub, key, returned_peak_single,
                                   ms2_from_same_file[specter_index]['precursorMz'],
                                   ms2_from_same_file[specter_index]['retentionTime'],
                                   ms2_from_same_file[specter_index]['num'])
        return_obj.feature_num = len(list_retention_time_split)
        return_obj.list_retention_time_split = list_retention_time_split
        return return_obj

    def align_peak_without_ms(self, split_peak_group_obj: SplitFeatureDataClass, subpeak_index,
                              row_comment, index, reverse_name_dict, save_to_aligned_list=True):
        raw_data = split_peak_group_obj.raw_data
        row_peak_data = RowPeakDataClass()
        if row_comment:
            row_peak_data.comment = row_comment
        max_h_ratio2 = [(0, False), (0, False)]  # 0-->first, 1-->second
        # median of mz so outliers don't ruin it!!
        subpeak_raw_data = {key: value for key, value in split_peak_group_obj.raw_data.items()
                            if key[1] == subpeak_index or key[0] == 'not_file'}
        med_mz = float(np.nanmedian([i.avg_mz for i in subpeak_raw_data.values()]))
        med_tr = float(np.nanmedian([i.tr_high for i in subpeak_raw_data.values()]))
        if ('not_file',) in raw_data:
            del dict_raw_data['not_file', ]
        for curr_file in self.filename:
            if (curr_file, subpeak_index) not in raw_data:
                peak_non_ms2 = self.find_peak(med_mz, [med_tr], curr_file)[0]
                # ---> it returns list[FindPeakReturnClass] so I need to take 0th element (it has only one element)
                raw_data[curr_file, subpeak_index] = peak_non_ms2

            # Adding to quantification table stuff
            curr_raw_data = raw_data[curr_file, subpeak_index]
            if (curr_file, subpeak_index) in split_peak_group_obj.ms2_num:
                to_add = "|".join(split_peak_group_obj.ms2_num[curr_file, subpeak_index])
                row_peak_data.scans += "|".join([row_peak_data.scans, to_add])
            if "Noise" in curr_raw_data.comment:
                row_peak_data.noise += "|" + curr_file
            if "TOO_FAR" in curr_raw_data.comment:
                row_peak_data.far += "|" + curr_file
            if not np.isnan(curr_raw_data.tr_max):
                if (curr_raw_data.peak_h == 0) or (curr_raw_data.tr_max - curr_raw_data.tr_min) == 0:
                    noise_ratio1 = 0
                else:
                    noise_ratio1 = curr_raw_data.peak_area / curr_raw_data.peak_h / (
                            curr_raw_data.tr_max - curr_raw_data.tr_min)
                if noise_ratio1 < row_peak_data.noise_ratio1:
                    row_peak_data.noise_ratio1 = noise_ratio1
            if curr_raw_data.peak_h > max_h_ratio2[0][0]:
                max_h_ratio2[1] = max_h_ratio2[0]
                max_h_ratio2[0] = (curr_raw_data.peak_h, "Noise" in curr_raw_data.comment)
            elif curr_raw_data.peak_h > max_h_ratio2[1][0]:
                max_h_ratio2[1] = (curr_raw_data.peak_h, "Noise" in curr_raw_data.comment)

            if type(curr_raw_data.isotopes) != tuple:
                isotopic_ratio_list = np.array([np.nan, np.nan, np.nan, np.nan])
            else:
                isotopic_ratio_list = curr_raw_data.isotopes[0][:, 0]
            if save_to_aligned_list:
                # I am adding data to the aligned_dict for each sample separately
                self.add_to_aligned_dict(index,
                                         ["A", "h", "delta_time", "delta_mz", "peak_length", "M05", "M1", "M15",
                                          "M2"],
                                         np.concatenate((curr_raw_data.return_aligned_list_parameters(med_tr, med_mz),
                                                         isotopic_ratio_list)),
                                         reverse_name_dict,
                                         curr_file)
        row_peak_data.noise_ratio2 = int(max_h_ratio2[1][1]) + int(max_h_ratio2[0][1])
        if save_to_aligned_list:
            # I am adding common peak_data to tle aligned dict
            self.add_to_aligned_dict(index, ["peak_data"],
                                     [np.array([index, med_mz, med_tr, row_peak_data.scans, row_peak_data.noise,
                                                row_peak_data.far, row_peak_data.noise_ratio1,
                                                row_peak_data.noise_ratio2, row_peak_data.comment])],
                                     dict(), "")
        split_peak_group_obj.med_mz.append(med_mz)
        split_peak_group_obj.med_tr.append(med_tr)

    def plot_from_index(self, index, index_tuple=(), save_graph=False, graph_path_folder: str = "graphs2/",
                        show_plot=True):
        warnings.warn('THIS HAS TO BE UPDATED!!')
        if not self.mergedMS2scans:
            self.filter_constant_ions(save_deleted="")
            self.merge_features()
        if index != -1:
            index_tuple = self.dict_plot_index[index]

        if save_graph:
            if not os.path.isdir(dep.output_path + graph_path_folder):
                os.makedirs(dep.output_path + graph_path_folder)
            if graph_path_folder[-1] != "/":
                graph_path_folder += "/"

        # mass_feature is list of different MS2 spectra from different files that all have similar tr and mz
        mass_feature = np.array(self.mergedMS2scans[index_tuple[0]])

        split_peak_group_obj = self.align_peak_with_ms(mass_feature)
        self.align_peak_without_ms(split_peak_group_obj, index_tuple[1], '', -1, dict(), False)

        self.logger.log(20, f'| Plotting feature with index tuple {index_tuple}: '
                            f'mz={split_peak_group_obj.med_mz[-1]:.4f}, tr={split_peak_group_obj.med_tr[-1]:.2f}')

        plot_names = [key[0] for key, value in split_peak_group_obj.raw_data.items() if
                      key[1] == index_tuple[1]]
        plot_data = [value for key, value in split_peak_group_obj.raw_data.items() if
                     key[1] == index_tuple[1]]
        plot_graph(plot_data, plot_names, dep.output_path + graph_path_folder,
                   f"mz={split_peak_group_obj.med_mz[-1]:.4f}_tr={split_peak_group_obj.med_tr[-1]:.2f}",
                   split_peak_group_obj=split_peak_group_obj, subpeak_index=index_tuple[1],
                   start_comment=str(index_tuple) + "_", save_graph=save_graph, show_plot=show_plot)

    def calculate_isotopic_ratio(self, tr, mz, current_file, show_graph=False):
        intensity_dict: typing.Dict[float, typing.List[float]] = dict()  # Dictionary for writing all intensity trends
        mz_diff_dict: typing.Dict[float, typing.List[float]] = dict()  # Dictionary for writing all mz diff trends
        ret_ratios: typing.List[typing.Tuple[float, float]] = []
        ret_mz_diff: typing.List[typing.Tuple[float, float]] = []
        suitable_df = self.scans[
            (tr[0] < self.scans["retentionTime"]) & (
                    self.scans["retentionTime"] < tr[2])
            & (self.scans["msLevel"] == 1) & (self.scans["filename"] == current_file)]
        for delta_mz in [0, 0.5, 1, 1.5, 2, 3, 4]:  # For each mass difference
            intensity_dict[delta_mz] = []
            mz_diff_dict[delta_mz] = []
            i = -1
            for row in suitable_df.itertuples():  # For each MS1 spectrum (time point)
                i += 1
                ms2_whole = self.peaks[current_file][row.num]
                if len(ms2_whole) == 0 or np.isnan(mz):
                    intensity = 0
                    mz_diff_dict[delta_mz].append(np.nan)
                else:
                    ms2_filtered = ms2_whole[(abs(mz + delta_mz * 1.003355 - ms2_whole[:, 0]) < self.parameters[
                        "isotopic_ratio_mz_tol"])]
                    intensity = sum(ms2_filtered[:, 1])
                    if intensity > 0:
                        mz_diff_dict[delta_mz].append(
                            sum(ms2_filtered[:, 1] * ms2_filtered[:, 0]) / intensity - (mz + delta_mz * 1.003355))
                    else:
                        mz_diff_dict[delta_mz].append(np.nan)

                if delta_mz != 0:
                    if intensity_dict[0][i] == 0:
                        ratio = np.nan
                    else:
                        ratio = intensity / intensity_dict[0][i]
                    intensity_dict[delta_mz].append(ratio)
                else:
                    intensity_dict[delta_mz].append(intensity)
            if delta_mz != 0:
                # calculate ratios
                ret_ratios.append(calculate_average(intensity_dict[delta_mz], intensity_dict[0], remove_zeros=True))
                ret_mz_diff.append(calculate_average(mz_diff_dict[delta_mz], intensity_dict[0], remove_zeros=False))
        if show_graph:
            suitable_df2 = self.scans[
                (tr[1] - 0.01 < self.scans["retentionTime"]) & (
                        self.scans["retentionTime"] < tr[1] + 0.01)
                & (self.scans["msLevel"] == 1) & (self.scans["filename"] == current_file)]
            fig_ms2, ax_ms2 = plt.subplots()
            ms2 = self.peaks[current_file][suitable_df2.iloc[0].num]
            ax_ms2.vlines(x=ms2[:, 0], ymin=0, ymax=ms2[:, 1])
            ax_ms2.set_xlim(mz - 0.5, mz + 4.5)
            ax_ms2.text(mz + 1, max(intensity_dict[0]),
                        "\n".join([str(np.round(i[0], 3)) + "+-" + str(np.round(i[1], 3)) for i in ret_ratios]),
                        wrap=True)
            ax_ms2.set_title(f"mz={mz:.4f}_tr={tr[1]:.2f}")
            fig_ms2.savefig(dep.output_path + f'graphs_isotopes/mz={mz:.4f}_tr={tr[1]:.2f}.svg', dpi=70)
            # figMS2.show()
            plt.close(fig_ms2)
        return np.array(ret_ratios), np.array(ret_mz_diff)

    def find_peak(self, mz: float, r_time: list, curr_file: str, show_graph=False) -> typing.List[FindPeakReturnClass]:
        """
        Method that finds peak based on mass and time. It uses find_peak_function for actually finding peak
        :param self:
        :param mz: mass where algorithm extracts TIC from
        :param r_time: list of retention time where algorithm extracts TIC from
        :param curr_file: from which file algorithms looks for peaks (MS2 data)
        :param show_graph: If not False it draws graph (of every peak in every file!)
        :return: [(avg_mz, avg_mz_s), (t_start, t_max, t_end), (peak_h, peak_area), avg_mz_list, time_list,
         intensity_list, comment (or at least place for comment)]
        """
        self.logger.log(14, f'|||| find_peak for mz={mz}, tr={r_time}, file="{curr_file}")')
        suitable_df = self.scans[
            (min(r_time) - self.parameters["time_interval"] < self.scans["retentionTime"]) & (
                    self.scans["retentionTime"] < max(r_time) + self.parameters["time_interval"])
            & (self.scans["msLevel"] == 1) & (self.scans["filename"] == curr_file)]
        time_list = []
        intensity_list = []
        avg_mz_list = []
        #  extracting tic so it can calculate where peak is located
        for row in suitable_df.itertuples():
            time_list.append(row.retentionTime)
            curr_df = self.peaks[curr_file][row.num]
            int_sum = 0
            if len(curr_df) > 0:
                curr_df = curr_df[(abs(mz - curr_df[:, 0]) < self.parameters["delta_mz1"])]
                int_sum = sum(curr_df[:, 1])
            intensity_list.append(int_sum)
            if int_sum:
                avg_mz_list.append(sum(curr_df[:, 1] * curr_df[:, 0]) / int_sum)
            else:
                avg_mz_list.append(np.nan)
        if sum(intensity_list) == 0:
            # This means that peak was not present!
            # Specter_count = 16
            return [FindPeakReturnClass().return_empty(time_list, intensity_list, comment="NO_SIGNAL")]
        intensity_list = np.array([x for y, x in sorted(zip(time_list, intensity_list))])
        avg_mz_list = np.array([x for y, x in sorted(zip(time_list, avg_mz_list))])
        time_list = sorted(time_list)

        noise_quotient = max(np.convolve(intensity_list, np.ones(5) / 5, mode='same')) / np.average(
            np.convolve(intensity_list, np.ones(5) / 5, mode='same'))

        # averaging spectra on convolve points
        # y = np.convolve(intensity_list, np.ones(self.parameters["convolve"]) \
        # / self.parameters["convolve"], mode='same')
        y = savgol_filter(intensity_list, 7, 3)

        # Calling function for finding peaks:
        peak_list = find_peak_function(y, self.parameters["fpf_ascending"], self.parameters["fpf_descending"])

        # finding right peak(s)
        peak_data_dict = find_interval(r_time, time_list, peak_list, self.parameters["rt_tolerance"])

        return_list = []
        if not len(peak_data_dict):
            return [FindPeakReturnClass().return_empty(time_list, intensity_list,
                                                       comment="NOT_FOUND" + "_Noise" * int(
                                                           noise_quotient < self.parameters["fp2_noise"]))]
        else:
            for peak_list, specter_index in peak_data_dict.items():
                ret = FindPeakReturnClass()

                start_id = peak_list[0][0]
                end_id = peak_list[1][0] + 1
                ret.start_index, ret.end_index = start_id, end_id
                ret.specter_index = specter_index
                # Saving uncut versions of the lists!
                ret.avg_mz_list, ret.time_list, ret.intensity_list = avg_mz_list, time_list, intensity_list
                # Now they can be cut:
                avg_mz_list_c = avg_mz_list[start_id: end_id]
                intensity_list_c = intensity_list[start_id: end_id]
                y_c = y[start_id: end_id]
                time_list_c = time_list[start_id: end_id]

                if len(avg_mz_list_c[~np.isnan(avg_mz_list_c)]) == 0:
                    ret.avg_mz = np.nan
                    ret.avg_mz_s = np.nan
                else:
                    weights1 = intensity_list_c[~np.isnan(avg_mz_list_c)] ** parameters[
                        "mz_weight_exponent"]
                    ret.avg_mz = np.average(avg_mz_list_c[~np.isnan(avg_mz_list_c)],
                                            weights=weights1)
                    ret.avg_mz_s = np.std(avg_mz_list_c[~np.isnan(avg_mz_list_c)])
                # not true standard deviation because it is not weighted
                index_most_intense = np.where(y_c == np.amax(y_c))[0][0]  # Index when the peak is the highest
                t_most_intense = time_list_c[index_most_intense]
                ret.peak_h = intensity_list_c[index_most_intense]
                ret.peak_area = sum(y_c)

                ret.tr_min = round(time_list_c[0], 3)
                ret.tr_high = round(t_most_intense, 3)
                ret.tr_max = round(time_list_c[-1], 3)
                tr = ret.tr_min, ret.tr_high, ret.tr_max

                if len(r_time) == 1 and abs(r_time[0] - tr[1]) > 0.15:
                    return [FindPeakReturnClass().return_empty(
                        time_list, intensity_list,
                        comment=f"TOO_FAR({abs(r_time[0] - tr[1]):.2f})" +
                                "_Noise" * int(noise_quotient < self.parameters["fp2_noise"]))]

                ret.isotopes = self.calculate_isotopic_ratio(tr, ret.avg_mz, curr_file)

                if show_graph:
                    fig, ax1 = plt.subplots()
                    ax2 = ax1.twinx()
                    ax1.set_xlabel('Retention time [min]')
                    ax1.set_ylabel('Intensity')
                    ax2.set_ylabel('delta m/z)', color='g')
                    ax2.ticklabel_format(useOffset=False, style='plain')
                    plt.title(f"Plot of m/z = {mz}, tr = {r_time} for file {curr_file}")
                    ax1.plot(time_list_c, y_c, label="convolve")
                    ax1.plot(time_list_c, intensity_list_c, label="Original list")
                    ax1.vlines(x=r_time, ymin=0, ymax=max(intensity_list_c), colors="b")
                    ax1.fill_between(time_list_c, y_c, 0, alpha=0.3)
                    ax2.plot(time_list_c, avg_mz_list_c, "g-")
                    fig.show()
                    self.logger.log(14, f"Peak data: original m/z: {mz}, original time: {r_time}"
                                        f"actual times: {tr}, h={ret.peak_h}, area={ret.peak_area},"
                                        f" mz={ret.avg_mz}+-{ret.avg_mz_s}")

                ret.comment = str(round(noise_quotient, 3))
                ret.comment += "_Noise" * int(noise_quotient < self.parameters["fp2_noise"])
                ret.nan_object = False
                return_list.append(ret)
        return return_list

    def add_to_aligned_dict(self, i, names_list, value_list, reverse_name_dict, filename1):
        """
        Function to add from find_peak2 returned list to aligned_dict
        :param self:
        :param i:
        :param names_list:
        :param value_list:
        :param reverse_name_dict:
        :param filename1:
        :return:
        """
        if len(self.aligned_dict["h"]) <= i:
            for j in names_list:
                if j != "peak_data":
                    self.aligned_dict[j].append(np.array([np.nan] * len(reverse_name_dict)))
            self.aligned_dict["peak_data"].append([])  # mz, tr, scans, noise, far, isotopic
        if names_list == ["peak_data"]:
            t_val = np.nanmedian(self.aligned_dict["M1"][i])
            self.aligned_dict["peak_data"][i] = tuple(np.concatenate((value_list[0], [t_val])))
            return 0
        for j in range(len(names_list)):
            self.aligned_dict[names_list[j]][i][reverse_name_dict[filename1]] = value_list[j]

    # Other methods ###################################
    def check_for_formulas(self, mass_list, adduct_difference=-dep.m_p, mz_tolerance=0.01,
                           time_tolerance=1) -> np.array:
        """
        based on given mass_list and adduct_difference algorithm tries to match every mass to aligned list feature(s).
        :param mass_list: list of type [[id1, mz1, tr1], [id2, mz2, tr2], ...]. Same as file list that is input for
        merged_msms_from_table
        :param adduct_difference: difference from masses in list to mz. e.g. for M-H, difference is -1.007... if masses
        are already mz values of ions than parameter equals 0
        :param mz_tolerance: self explanatory
        :param time_tolerance: If time are unknown than set to 0 but there needs to be some value in mass_list!!!
        :return: 2d array 2x n(matched features). First column is index of aligned feature and second is mass difference
        """
        table1 = mass_list
        table1 = [(i[0], i[1] + adduct_difference, i[2], []) for i in table1]

        for i in range(len(self.aligned_dict["peak_data"])):
            mz = float(self.aligned_dict["peak_data"]["mz"][i])
            for j in range(len(table1)):
                if abs(mz - table1[j][1]) < mz_tolerance:
                    if (not time_tolerance) \
                            and (table1[j][2] - float(self.aligned_dict["peak_data"]["mz"][1])) < time_tolerance:
                        table1[j][3].append((self.aligned_dict["peak_data"]["index"][i], mz - table1[j][1],
                                             self.aligned_dict["peak_data"]["tr"][i]))
        for i in range(len(table1)):
            self.logger.log(20, f"{table1[i][0]}, M{adduct_difference:+.0f}={table1[i][1]:.3f}: {table1[i][3]}")

        arr2 = np.empty([len(self.aligned_dict["A"]), 2], dtype='<U40')
        for i in table1:
            for j in i[3]:
                arr2[int(j[0]), 0] += " " + i[0]
                arr2[int(j[0]), 1] += " " + str(round(j[1], 8))

        return arr2

    def match_features(self, file_name_absolute, mz_tolerance=0.03, encoding1="UTF-8-sig"):
        """
        Finds similar masses from suspect list and self.aligned_dict. It saves matched table to self.matched_list
        :param self: mzproject object
        :param file_name_absolute: Absolute path to csv file.
        :param mz_tolerance: Max mz tolerance to match suspect and feature
        :param encoding1: Encoding of database
        :return: self.matched_list
        """
        conn = open(file_name_absolute, encoding=encoding1)
        lines = conn.readlines()
        conn.close()
        suspect_list = []
        for line in lines[1:]:
            line_list = line.strip().split(",")
            suspect_list.append(line_list)
        suspect_list_mz = np.array([i[6] for i in suspect_list], dtype=np.float)
        peaks_list_mz = np.array([i[1] for i in self.aligned_dict["peak_data"]], dtype=np.float)
        j_start = 0
        adduct_change = - dep.m_p
        match_list = []
        for i in range(len(peaks_list_mz)):
            for j in range(j_start, len(suspect_list_mz)):
                if peaks_list_mz[i] > suspect_list_mz[j] + 1:
                    j_start = j
                if peaks_list_mz[i] < suspect_list_mz[j] - 1:
                    break

                if abs(suspect_list_mz[j] + adduct_change - peaks_list_mz[i]) < mz_tolerance:
                    match_list.append(
                        suspect_list[j] + ["|", suspect_list_mz[j] + adduct_change - peaks_list_mz[i], "|"] +
                        list(self.aligned_dict["peak_data"][i])[0:4])
        self.matched_list = match_list
        return match_list

    def calculate_mean(self, suffix_length=7, dtype=float):
        """
        Algorithm averages sample parallels from aligned dict. If there are 3 samples than at least 2 must have non 0
        integral. If there is only 1 or 2 samples, "all" concentration must be non-zero.
        :param self: mzproject object
        :param suffix_length: number of character that is common to all sample name (7 for _negxml and 8 for _neg.xml)
        :param dtype: type of characters in table because aligned table can be type str
        """
        group_dict = dict()
        for i in range(len(self.filename)):
            curr_name = self.filename[i].replace(" ", "")[:-suffix_length]
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
                group_dict[cut_curr_name].append(i)
            else:
                group_dict[cut_curr_name] = [i]
        self.logger.log(20, f"| Calculate mean: group dict={group_dict}")
        for gr_name in group_dict.keys():
            self.header_averaged.append(gr_name)
        for table_name in [i for i in list(self.aligned_dict.keys()) if i != "peak_data"]:
            self.averaged_aligned_dict[table_name] = np.zeros((len(self.aligned_dict["peak_data"]), len(group_dict)))
            for i in range(len(self.aligned_dict["peak_data"])):
                k = 0
                for gr_name, j in group_dict.items():
                    values = []
                    for v in j:
                        values.append(dtype(self.aligned_dict[table_name][i][v]))
                    if values.count(0) > max(0, len(values) - 2):
                        mean = 0
                    # TODO: see if values are behaving as as you wish
                    else:
                        mean = np.average([i for i in values if i != 0])
                    self.averaged_aligned_dict[table_name][i][k] = mean
                    k += 1

    def merge_duplicate_rows(self, reindex=False):
        """
        Some files get duplicated because peaks are too long for merge_features to recognize it as one peak.
         Mass features within mz and tr tolerance are merged. If there is merge some indexes are erased.
         You can reindex them with reindex variable.
        tolerance
        :param self: mzproject object
        :param reindex: if indexes are rewritten. This causes mismatch with exported pictures that are generated in
         generate table!. Output is saved back in aligned_dict.
        """
        merged_id_list: typing.List[typing.List[int]] = []
        pd_list: typing.Union[typing.Dict, np.ndarray] = self.aligned_dict["peak_data"]
        self.logger.log(20, '| Starting to merge duplicated rows')
        for new_id in range(len(pd_list)):
            # General approach is that for each index we check if there are already features that are similar
            if len(merged_id_list) == 0:  # Only the first iteration
                merged_id_list.append([new_id])
                continue
            for group_id in range(len(merged_id_list)):
                check_id = merged_id_list[group_id][-1]  # Index of one of the features in the group

                delta_mz = abs(float(pd_list[new_id][1]) - float(pd_list[check_id][1]))
                mz_tol = self.parameters["duplicates_mz_tolerance"]
                delta_tr = abs(float(pd_list[new_id][2]) - float(pd_list[check_id][2]))
                if (delta_mz < mz_tol) and (delta_tr < self.parameters["duplicates_tr_tolerance"]):
                    self.logger.log(18, f'|| Features with indices {check_id} and {new_id} are equivalent')
                    merged_id_list[group_id].append(new_id)
                    # This means that current feature is duplicated and can be merged
                    break
            else:
                merged_id_list.append([new_id])  # Current feature is not duplicated so we add new element in list

        aligned_list_2 = []
        temp_aligned_dict = dict()
        for new_id in [i for i in list(self.aligned_dict.keys()) if i != "peak_data"]:
            temp_aligned_dict[new_id] = np.zeros((len(merged_id_list), len(self.filename)), dtype=np.float)

        for new_id in range(len(merged_id_list)):
            identical_rows = pd_list[merged_id_list[new_id]]
            # Merging data from multiple columns
            merged_row: typing.Union[typing.Dict, np.ndarray] = np.zeros(1, dtype=pd_list.dtype)
            for col_name in ['index', 'noise_ratio1', 'noise_ratio2']:
                merged_row[col_name] = min(identical_rows[col_name])
            for col_name in ['scans', 'noise', 'too_far', 'comment']:
                merged_row[col_name] = ''.join(identical_rows[col_name])
                if col_name != 'comment':  # Cleaning duplicates
                    merged_row[col_name] = '|'.join({i for i in merged_row[col_name][0].split('|') if i != ''})
            merged_row['mz'] = np.average(identical_rows['mz'])
            merged_row['tr'] = np.average(identical_rows['tr'])
            merged_row['M_plus_1'] = max(identical_rows['M_plus_1'])
            comment_add = "_merged_" + "|".join([str(i) for i in sorted(identical_rows['index'])])
            merged_row['comment'] = merged_row['comment'][0] + comment_add * (len(merged_id_list[new_id]) > 1)
            aligned_list_2.append(merged_row[0])

            for group_id in [i for i in list(self.aligned_dict.keys()) if i != "peak_data"]:
                value_list = self.aligned_dict[group_id][merged_id_list[new_id]]
                if np.all(np.isnan(value_list)):
                    temp_aligned_dict[group_id][new_id] = np.nan
                else:
                    # Maximum since all we use is actually height and area
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        temp_aligned_dict[group_id][new_id] = np.nanmax(value_list.astype(float), axis=0)
        if reindex:
            for new_id in range(len(aligned_list_2)):
                aligned_list_2[new_id]['index'] = str(new_id)
        m_string = [max([len(j[i]) for j in aligned_list_2]) for i in ['scans', 'noise', 'too_far', 'comment']]
        dtype = dep.get_dtype(m_string)
        temp_aligned_dict["peak_data"] = np.array(aligned_list_2, dtype=dtype)
        self.logger.log(18, f"|| Finished merging duplicated rows. From {len(self.aligned_dict['peak_data'])} rows to"
                            f" {len(temp_aligned_dict['peak_data'])}")
        self.aligned_dict = temp_aligned_dict
    # endregion

    # region export_modules

    def save_spectrum_picture(self, index, fragments_list, max_diff_abs=0.05):
        scan_index_string = self.aligned_dict["peak_data"]["scans"][self.aligned_dict["peak_data"]["index"] == index][0]
        mz = self.aligned_dict["peak_data"]["mz"][self.aligned_dict["peak_data"]["index"] == index][0]
        scan_index_list = [[i, i.split("__")[0], 0, [], 0, 0] for i in scan_index_string.split("|") if
                           i.count("__") == 1]
        # 0: full index
        # 1: filename
        # 2: energy
        # 3: spectrum
        # 4: num_matching_peaks
        # 5: avg_diff
        for i in range(len(scan_index_list)):
            scan_index_list[i][2] = self.scans.loc[scan_index_list[i][0]]["tempenergy"]
            spectrum = self.peaks[scan_index_list[i][1]][scan_index_list[i][0]]
            spectrum = spectrum[spectrum[:, 0] < mz + 20]
            scan_index_list[i][3] = spectrum
            # taking only peaks above 5%
            percentage_min = 5
            spectrum[:, 1] = spectrum[:, 1] * 100 / max(spectrum[:, 1])
            spectrum = spectrum[spectrum[:, 1] >= percentage_min]
            for fragment in fragments_list:
                min_dist = min(abs(fragment - spectrum[:, 0]))
                if min_dist <= max_diff_abs:
                    scan_index_list[i][4] += 1
                    scan_index_list[i][5] += min_dist
            if scan_index_list[i][4] == 0:
                scan_index_list[i][5] = 1E5
            else:
                scan_index_list[i][5] /= scan_index_list[i][4]
        scan_index_list = sorted(scan_index_list, key=lambda x: (-x[4], x[5]))
        best_feature = scan_index_list[0]
        alt_feature = [i for i in scan_index_list if i[2] != best_feature[2]][0]

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.vlines(x=best_feature[3][:, 0], ymin=0, ymax=best_feature[3][:, 1] / max(best_feature[3][:, 1]), colors="k")
        ax1.set_xlabel(xlabel="m/z")
        ax1.set_ylabel(ylabel="Relative intensity")
        plt.savefig(dep.output_path + f"{index}_best_{best_feature[2]:.0f}.svg", format="svg")

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.vlines(x=alt_feature[3][:, 0], ymin=0, ymax=alt_feature[3][:, 1] / max(alt_feature[3][:, 1]), colors="k")
        ax1.set_xlabel(xlabel="m/z")
        ax1.set_ylabel(ylabel="Relative intensity")
        plt.savefig(dep.output_path + f"{index}_alt_{int(alt_feature[2])}.svg", format="svg")

        conn = open(dep.output_path + f"{index}_data.txt", "w", encoding="UTF-8")
        conn.write(f"Best (index = {best_feature[0]}, tempenergy = {best_feature[2]:.0f}):\n")
        for line in best_feature[3]:
            conn.write(f"{line[0]}\t{line[1]}\n")
        conn.write(f"Alternative (index = {alt_feature[0]}, tempenergy = {alt_feature[2]:.0f}):\n")
        for line in alt_feature[3]:
            conn.write(f"{line[0]}\t{line[1]}\n")
        conn.close()

    def export_filtered_ms(self, index_list: list, exp_filename="sirius_1", split_on=50,
                           min_h=0, mode="non-averaged") -> None:
        """
        Based on aligned data, files and list of indexes it exports in ms file that can be imported by sirius.
        :param self: mzproject object
        :param index_list: list of indexes. It is important that type of data is same (indexes in peak_data are
         integers!) If table is in file then it can be read with f.read_simple_file_list.
        :param exp_filename: path relative to output_path without extension
        :param split_on: how many peaks is in one file
        :param min_h: how high must peak be to get into .ms file
        :param mode: non-averaged=export all spectra; cluster_averaged=Hierarchical clustering is done on mass spectrum
        """
        export_string = ""
        k = 0
        peak_data = self.aligned_dict["peak_data"]
        filtered_list = peak_data[np.isin(peak_data["index"], index_list)]
        order_list = np.concatenate(
            (np.array(index_list).reshape(-1, 1), np.array(range(len(index_list))).reshape(-1, 1)), axis=1)
        order_list = order_list[order_list[:, 0].argsort()]
        order_list = order_list[:, 1].argsort()
        filtered_list = filtered_list[order_list]
        self.logger.log(20, f"| Starting to export .ms files")
        if len(filtered_list) == 0:
            self.logger.log(30, f"|| Indexes do not match!! aligned_dict indexes: {list(peak_data[0])[:5]},"
                                f" input indexes: {index_list[:5]}")
        for line in filtered_list:
            line = list(line)
            # line[3] = line[3].decode("latin1")
            mz = line[1]
            scan_index_list = [[i, i.split("__")[0]] for i in line[3].split("|") if i.count("__") == 1]
            export_string += ">compound " + f"{line[0]}"
            export_string += "\n>parentmass " + f"{mz}"
            export_string += "\n>ionization [M+?]-"
            export_string += "\n>instrumentation Unknown (LCMS)"
            export_string += f"\n>source file: {float(mz):.4f}_{float(line[2]):.2f}"
            export_string += "\n>quality UNKNOWN"
            export_string += "\n>rt " + f"{line[2] * 60}"
            export_string += "\n#scan index " + " ".join([m[0] for m in scan_index_list])
            export_string += "\n#filename " + " ".join((m[1] for m in scan_index_list))
            export_string += f"\n\n>ms1peaks\n{mz} 100.0"
            export_string += f"\n{float(mz) + dep.m_p} {float(line[9]):.4f}"
            if mode == "non-averaged":
                for j in range(len(scan_index_list)):

                    filename = scan_index_list[j][1]
                    index = scan_index_list[j][0]
                    specter = self.peaks[filename][index]
                    specter = specter[specter[:, 0] < mz + 30]
                    if sum(specter[:, 1]) != 0:
                        collision_energy = self.scans.loc[index]["tempenergy"]
                        # specter[:, 1] /= max(specter[:, 1]) / 100
                        export_string += f"\n\n# {index}\n>collision {collision_energy:.0f}\n"
                        for i in specter:
                            if i[1] > min_h:
                                export_string += "{:.5f} {:.1f} \n".format(i[0], i[1])
                export_string += "\n"
            elif mode == "cluster_averaged":
                energy_levels = [i for i in list(set(self.scans.tempenergy)) if not np.isnan(i)]
                for level in energy_levels:
                    spectrum = self.average_spectrum(line[0], level, grouping_threshold=0.05)
                    # TODO: Check if MS2 grouping threshold is too large/small
                    export_string += f"\n>collision {level:.0f}\n"
                    for i in spectrum:
                        if i[1] > min_h:
                            export_string += "{:.5f} {:.1f} \n".format(i[0], i[1])
                    export_string += "\n"
                export_string += "\n"
            if (k + 1) % split_on == 0:
                name = dep.output_path + exp_filename + "_" + str(k + 1 - split_on) + "-" + str(k + 1) + ".ms"
                with open(name, "w", encoding="UTF-8") as conn:
                    conn.write(export_string)
                self.logger.log(18, "|| Wrote: ", exp_filename + str(k + 1 - split_on) + "-" + str(k + 1) + ".ms")

                export_string = ""
            k += 1

        with open(dep.output_path + exp_filename + f"_{(k + 1) // split_on * split_on}-end.ms", "w",
                  encoding="UTF-8") as conn:
            conn.write(export_string)
        self.logger.log(20, f"|| MS spectra are exported! in {dep.output_path + exp_filename}***-***.ms")

    def average_spectrum(self, feature_id, energy_level, grouping_threshold=0.1):
        # Getting line
        line = self.aligned_dict["peak_data"][feature_id]
        if line[0] != feature_id:
            raise ("Indexes do not match", feature_id, line[0])
        # Getting list of scans
        scans_string = line[3].replace(" ", "")
        scans_list = scans_string.split("|")
        if scans_list[0] == "":
            scans_list = scans_list[1:]
        # Getting all spectrum from indexes
        enriched_scans_list = []
        for scan in scans_list:
            line = self.scans2.loc[scan]
            spectrum = self.peaks[str(line.filename)][scan]
            spectrum = spectrum[spectrum[:, 0] < line.precursorMz + 50]
            # [index, spectrum, precursorMz, tempenergy, tot Ion current, filename]
            enriched_scans_list.append(
                (scan, spectrum, line.precursorMz, line.tempenergy, line.totIonCurrent, line.filename))

        # Creating flattened 2D array [[mz, i, index_in_original, totIonCurrent], ... ]
        merged_spectrum = np.empty((0, 4), float)
        for index1 in range(len(enriched_scans_list)):
            line = enriched_scans_list[index1]
            if line[3] != energy_level:
                continue
            add_list = np.c_[line[1], np.full((len(line[1]), 1), index1), np.full((len(line[1]), 1), line[4])]
            merged_spectrum = np.append(merged_spectrum, add_list, axis=0)

        # Creating clustering model
        cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='complete',
                                          distance_threshold=grouping_threshold)
        cluster.fit_predict(merged_spectrum[:, 0].reshape((-1, 1)))

        # Adding cluster grouping to original data
        merged_spectrum = np.concatenate((merged_spectrum, cluster.labels_.reshape(-1, 1)), axis=1)

        # Averaging groups mz and intensity based on i * totIonCurrent
        exp_spectrum = []
        for i in range(cluster.n_clusters_):
            curr_list = merged_spectrum[merged_spectrum[:, 4] == i]
            mz_i = sum(curr_list[:, 0] * curr_list[:, 1]) / sum(curr_list[:, 1])
            i_i = sum(curr_list[:, 1] * curr_list[:, 1]) / sum(curr_list[:, 1])
            exp_spectrum.append([mz_i, i_i])
        exp_spectrum = np.array(exp_spectrum)
        exp_spectrum = exp_spectrum[exp_spectrum[:, 0].argsort()]
        # exp_spectrum[:, 1] = exp_spectrum[:, 1] * 100 / max(exp_spectrum[:, 1])
        return exp_spectrum

    def export_mgf(self, index_list: list, exp_filename="mgf_1", split_on=1000, min_h=0, mode="A") -> None:
        """
        Based on aligned data, files and list of indexes it exports in ms file that can be imported by sirius.
        :param self: mzproject object
        :param index_list: list of indexes. It is important that type of data is same (indexes in 
        peak_data are integers!) If table is in file then it can be read with f.read_simple_file_list.
        :param exp_filename: path relative to output_path without extension
        :param split_on: how many peaks is in one file
        :param min_h: how high must peak be to get into .ms file
        :param mode: A: non merged, non averaged; B: merged non averaged; C: merged, averaged in energies;
        D: merged, averaged all
        """
        self.logger.log(20, "| Starting to export mgf")
        export_string = ""
        k = 0
        peak_data = self.aligned_dict["peak_data"]
        if not index_list:
            filtered_list = peak_data
        else:
            filtered_list = peak_data[np.isin(peak_data["index"], index_list)]
        order_list = np.concatenate(
            (np.array(index_list).reshape(-1, 1), np.array(range(len(index_list))).reshape(-1, 1)), axis=1)
        order_list = order_list[order_list[:, 0].argsort()]
        order_list = order_list[:, 1].argsort()
        filtered_list = filtered_list[order_list]
        if len(filtered_list) == 0:
            self.logger.log(30, f"|| Indexes do not match!! aligned_dict indexes: {list(peak_data[0])[:5]},"
                                f" input indexes: {index_list[:5]}")
        for line in filtered_list:
            scan_index_list = [[i, i.split("__")[0]] for i in line[3].split("|") if i.count("__") == 1]
            line = list(line)
            mz = line[1]
            m = 0
            if mode == "A":
                for j in range(len(scan_index_list)):
                    export_string += "\nBEGIN IONS"
                    export_string += "\nFEATURE_ID=" + f"{line[0]}"
                    export_string += "\nPEPMASS=" + f"{mz}"
                    export_string += "\nSCANS=" + f"{line[0]}"
                    export_string += "\nSOURCE_INSTRUMENT=LC-ESI-qTof"
                    export_string += f"\nFILENAME={line[0]}"
                    export_string += "\nCHARGE=1\nIONMODE=Negative"
                    export_string += "\nFILENAME=" + f"{line[0]}"
                    export_string += "\nRTINSECONDS=" + f"{line[0]}"  # + f"{line[2] * 60}"
                    export_string += "\n"
                    # export_string += "\nSCANS=" + ", ".join([m[0] for m in scan_index_list])
                    filename = scan_index_list[j][1]
                    index = scan_index_list[j][0]
                    specter = self.peaks[filename][index]
                    if sum(specter[:, 1]) != 0:
                        # specter[:, 1] /= max(specter[:, 1]) / 100
                        # export_string += f"\n\n# {index}\n>collision {collision_energy:.0f}\n"
                        for i in specter:
                            if i[1] > min_h:
                                export_string += "{:.5f} {:.1f} \n".format(i[0], i[1])
                    export_string += "END IONS\n"
            elif mode == "B":
                self.logger.log(40, "|| To be written!")
            elif mode == "C":
                energy_levels = [i for i in list(set(self.scans.tempenergy)) if not np.isnan(i)]
                for level in energy_levels:
                    spectrum = self.average_spectrum(line[0], level, grouping_threshold=0.05)
                    # TODO: Check if MS2 grouping threshold is too large/small
                    export_string += "\n"
                    export_string += "\nBEGIN IONS"
                    export_string += "\nFEATURE_ID=" + f"{line[0]}"
                    export_string += "\nPEPMASS=" + f"{mz}"
                    export_string += "\nSCANS=" + f"{line[0]}"
                    export_string += "\nSOURCE_INSTRUMENT=LC-ESI-qTof"
                    export_string += f"\nFILENAME={line[0]}"
                    export_string += "\nCHARGE=1\nIONMODE=Negative"
                    export_string += "\nFILENAME=" + f"{line[0]}_{int(level)}"
                    export_string += "\nRTINSECONDS=" + f"{line[0]}"  # + f"{line[2] * 60}"
                    export_string += "\n"
                    # export_string += "\nSCANS=" + ", ".join([m[0] for m in scan_index_list])
                    for i in spectrum:
                        if i[1] > min_h:
                            export_string += "{:.5f} {:.1f} \n".format(i[0], i[1])
                    export_string += "END IONS\n"
                export_string += "\n"
            m += 1
            if (k + 1) % split_on == 0:
                name = dep.output_path + exp_filename + "_" + str(k + 1 - split_on) + "-" + str(k + 1) + ".mgf"
                with open(name, "w", encoding="UTF-8") as conn:
                    conn.write(export_string)
                self.logger.log(18, f"|| Wrote: {exp_filename + str(k + 1 - split_on)}{str(k + 1)}.mgf")

                export_string = ""
            k += 1

        with open(dep.output_path + exp_filename + f"_{(k + 1) // 50 * 50}-end.mgf", "w",
                  encoding="UTF-8") as conn:
            conn.write(export_string)
        self.logger.log(20, f"|| MS spectra are exported! in {dep.output_path + exp_filename}***-***.mgf")

    def extract_sim(self, mz_list: list or float, filename: list or float,
                    mz_tolerance: float = 0.1, retention_time_range: tuple = (), show_graph: str = "",
                    comment: str = "", save_graph: str = "", rewrite_colors: dict = None) -> tuple:
        """
        Get SIM. Enabled for list of masses and different files
        :param rewrite_colors: Custom list of colors that is accepted by plt.
        :param mz_list: float or list of floats to extract SIM
        :param filename: files where we want to extract SIM; often self.filename
        :param mz_tolerance: max mz deviation to be interpreted as same peak
        :param retention_time_range: tuple of (tr_start, tr_end)
        :param show_graph:
        :param comment: Comment is added in title name
        :param save_graph:
        :return:
        """
        if type(filename) != list:
            filename = [filename]
        if type(mz_list) != list:
            mz_list = [mz_list]
        time_list = []
        intensity_list = []
        fig_tic, ax_tic = plt.subplots(figsize=(20, 10))
        names = []
        colors = np.concatenate((plt.cm.get_cmap("hsv")(np.linspace(0, 1, len(mz_list) * len(filename) // 2 + 1)),
                                 plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(mz_list) * len(filename) // 2 + 1))))
        for mz in mz_list:
            for filename1 in filename:
                names.append(filename1 + str(round(mz, 3)))
                time_list.append([])
                intensity_list.append([])
                if retention_time_range:
                    suitable_df = self.scans[
                        (retention_time_range[0] <= self.scans["retentionTime"]) & (
                                self.scans["retentionTime"] < retention_time_range[1])
                        & (self.scans["msLevel"] == 1) & (self.scans["filename"] == filename1)]
                else:
                    suitable_df = self.scans[(self.scans["msLevel"] == 1) & (self.scans["filename"] == filename1)]

                #  extracting tic so it can calculate where peak is located
                for row in suitable_df.itertuples():
                    time_list[-1].append(row.retentionTime)
                    curr_df = self.peaks[filename1][row.num]
                    if len(curr_df) > 0:
                        curr_df = curr_df[(abs(mz - curr_df[:, 0]) < mz_tolerance)]
                        intensity = sum(curr_df[:, 1])
                    else:
                        intensity = 0
                    intensity_list[-1].append(intensity)
        if show_graph or save_graph:
            for i, name in enumerate(names):
                if rewrite_colors:
                    name = filename[i]
                    ax_tic.plot(time_list[i], intensity_list[i], label=names[i], color=rewrite_colors[name][1],
                                linewidth=3, linestyle=rewrite_colors[name][0])
                else:
                    ax_tic.plot(time_list[i], intensity_list[i], label=names[i], color=colors[i], linewidth=3)
            handles, labels = ax_tic.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            fig_tic.legend(by_label.values(), by_label.keys(), title="File", loc='upper right')  # , fontsize='x-small')
            ax_tic.set_title(f"{show_graph}, {comment}: TIC +-{mz_tolerance}")
            limits = ax_tic.get_xlim()
            ax_tic.grid(True)
            ax_tic.set_xlim(limits[0], limits[1])
            if save_graph:
                self.logger.log(18, f'|| Saving plot in {dep.output_path}{save_graph}_'
                                    f'tr={retention_time_range}_{comment}.svg')
                fig_tic.savefig(
                    dep.output_path + f'{save_graph}_tr={retention_time_range}_{comment}.svg',
                    dpi=70)
            if show_graph:
                fig_tic.show()
        plt.close(fig_tic)
        return time_list, intensity_list

    def export_tables(self, file_name: str, additional_data_array=(), namelist=()):
        """
        Saves tables in dep.output_path/common name_typeOfTable.csv. First columns are common and then there is table
        n(files) x n(mass_features). There can also be additional columns between peak_data and rest of table. 
        In this case you must add list of strings to namelist and additional_data_array.
        :param self: mzproject object
        :param file_name: Common name for export of tables
        :param namelist:  Only if additional data: dimensions: 1 x m
        :param additional_data_array: Only if additional data: dimensions n(mass_features) x m
        """
        header = ",".join(self.header + list(namelist) + self.filename)
        self.logger.log(20, f"| Starting to export tables in {dep.output_path}")
        # self.dict_plot_index:
        path1 = f"{dep.output_path}/{file_name}-index_dict.csv"
        conn = open(path1, "w", encoding="UTF-8")
        for key, value in self.dict_plot_index.items():
            conn.write(" ".join([str(i) for i in (key,) + value]) + "\n")
        conn.close()

        for i in [i for i in list(self.aligned_dict.keys()) if i != "peak_data"]:
            self.logger.log(18, f"|| Saving table {i} with additional {len(additional_data_array)} columns.")
            if len(additional_data_array) != 0:
                # noinspection PyTypeChecker
                np.savetxt(dep.output_path + "/" + file_name + f"-{i}.csv",
                           np.concatenate((self.aligned_dict["peak_data"].tolist(), additional_data_array,
                                           self.aligned_dict[i]), axis=1), delimiter=',', fmt='%s',
                           header=header, comments='')
            else:
                # noinspection PyTypeChecker
                np.savetxt(dep.output_path + "/" + file_name + f"-{i}.csv",
                           np.concatenate((self.aligned_dict["peak_data"].tolist(),
                                           self.aligned_dict[i]), axis=1), delimiter=',', fmt='%s',
                           header=header, comments='')

    def export_tables_averaged(self, file_name):
        """
        For exporting averaged_aligned_dict. It uses peak_data from aligned dict!
        :param self:
        :param file_name: common file name for tables relative to dep.output_path
        """
        header = ",".join(self.header + self.header_averaged)
        self.logger.log(20, f"| Saving averaged aligned dict to: {dep.output_path + file_name}...")
        for i in [i for i in list(self.averaged_aligned_dict.keys()) if i != "peak_data"]:
            self.logger.log(18, f"|| Saving table: {dep.output_path + file_name}-{i}.csv")
            # noinspection PyTypeChecker
            np.savetxt(dep.output_path + file_name + f"-{i}.csv",
                       np.concatenate((self.aligned_dict["peak_data"].tolist(),
                                       self.averaged_aligned_dict[i]), axis=1), delimiter=',', fmt='%s',
                       header=header, comments='')

    def get_ms2_spectrum(self, filename: str, key1: str, show_graph: bool = True, save_plot: str = "",
                         title: str = "test_specter", most_intense: int = -1, min_h: int = -1, round_mz: bool = False):
        """
        Returns MS2 spectrum and can also plot image. It needs filename and key of spectrum that must be known 
        in advanced
        :param self: mzproject object
        :param filename: string name of file that is read in mzproject
        :param key1: key of dictionary, that is also written in scan dataframe
        :param show_graph: self explanatory
        :param save_plot: if changed it will save file with name = save_plot
        :param title: Title in plt figure
        :param most_intense: for filtering. If changed it filter only most_intense number of peaks
        :param min_h: for filtering. If changed it filters of peaks with h<min_h
        :param round_mz: If True then all mz will be rounded
        :return: np.array([[mz, h], ...)
        """
        array = self.peaks[filename][key1]
        if min_h > 0:
            array = array[array[:, 1] > min_h]
        if most_intense > 0:
            most_intense = min(most_intense, len(array))
            array = array[array[:, 1].argsort()[::-1][:most_intense]]
        if round_mz:
            array[:, 0] = np.round(array[:, 0])
        if show_graph or save_plot:
            fig, ax = plt.subplots(1, 1)
            ax.vlines(array[:, 0], ymin=0, ymax=array[:, 1], linestyles="solid", colors="k", label=key1)
            fig.suptitle(title)
            ax.legend(loc=1)
            ax.set_xlabel('mz')
            ax.set_ylabel('h')
            if show_graph:
                fig.show()
            if save_plot:
                fig.savefig(save_plot)
        return array

    # endregion3
    # region msms_finding_modules

    def filter_constant_ions(self, show_graph: bool = False, save_graph: typing.Union[bool, str] = "",
                             save_deleted: str = "deleted_masses.txt", abs_path: bool = False) -> np.array:
        """
        method that sorts self.scans2 per mz value and figures if MS2 spectra are occurring more than
        min_len_peaks_per_file times
        :param abs_path: If true paths are absolute
        :param save_deleted: if != "" then it saves duplicated masses to file
        :param save_graph: self explanatory, relative path
        :param self: mzproject object
        :param show_graph: plots scatter plot
        :return np.array of unique duplicated masses
        """
        min_len_peaks_per_file = len(self.filename) * self.parameters["min_len_peaks_per_file"]
        self.logger.log(20, f"| Starting to filter recurring masses; minimal number of MS2 spectra ="
                            f" {min_len_peaks_per_file} ({self.parameters['min_len_peaks_per_file']} per file).")
        self.scans2.sort_values(by=['precursorMz'], inplace=True)
        self.scans2.loc[:, "keep"] = True
        mz1 = 0
        indexes = []
        index_start = 0
        count = 0
        for row in self.scans2.itertuples():
            indexes.append(row.num)
            if row.msLevel != 2:
                continue
            if row.precursorMz - mz1 > self.parameters["mz_tolerance"]:
                if count - index_start < min_len_peaks_per_file:
                    mz1 = row.precursorMz
                    index_start = count
                else:
                    self.scans2.loc[indexes[index_start + 2]: indexes[-2], 'keep'] = False
                    mz1 = row.precursorMz
                    index_start = count
            count += 1
        df_keep = self.scans2[self.scans2.keep]
        df_not_keep = self.scans2[~self.scans2.keep]
        if show_graph or save_graph:
            fig_1, ax_1 = plt.subplots()
            ax_1.scatter(df_keep["retentionTime"],
                         df_keep["precursorMz"],
                         c="g", s=1, marker=".", zorder=1)
            ax_1.scatter(df_not_keep["retentionTime"],
                         df_not_keep["precursorMz"],
                         c="r", s=1, marker=".", zorder=2)
            if show_graph:
                fig_1.show()
            if save_graph:
                fig_1.savefig(dep.output_path * (not abs_path) + save_graph)
            plt.close(fig_1)
        deleted = self.scans2[~self.scans2["keep"]]['precursorMz']
        deleted = np.array(deleted.unique())
        if save_deleted != "":
            with open(str(dep.output_path * (not abs_path)) + save_deleted, "w", encoding="UTF-8") as conn:
                conn.write(
                    "\n".join(("{:.6f}".format(i) for i in deleted)))
        self.logger.log(18, f"|| Discarded {len(df_not_keep.index)} MS2 spectra and kept {len(df_keep.index)}")
        return deleted

    def merge_features(self):
        """
        Groups MS2 spectra that have similar tr and similar mz. Output gets saved in self.mergedMS2scans.
        self.mergedMS2scans looks like list[list[np.array['num', 'retentionTime', 'basePeakMz', 'tempenergy',
         'precursorMz', "precursorIntensity", "filename"], ...(similar MS2 scans)], ... (other MS2 scans)]
        :param self: mzproject object
        """
        # to filter out constantly present masses I need to get them:
        deleted_precursor_mz_list = self.scans2[~self.scans2["keep"]]['precursorMz']
        deleted_precursor_mz_list = np.array(deleted_precursor_mz_list.unique())
        self.logger.log(20, '| Starting to merge MS2 spectra to get groups.')
        col_names = ['num', 'retentionTime', 'basePeakMz', 'tempenergy', 'precursorMz', "precursorIntensity",
                     "filename"]
        array = self.scans2[self.scans2["keep"]][col_names].to_records(False)
        included = set()
        merged_list = []
        for i, line in enumerate(array):
            # set included makes sure that features that were already added are not processed multiple times
            # (in this case the for loop needs to continue)
            if array[i].num in included:
                continue
            included.add(array[i].num)
            merged_list.append([array[i]])
            # This for loop checks all MSMS scans after feature i and adds them to the group if they are close enough
            for j in range(i + 1, len(array)):
                if array[j].precursorMz > array[i].precursorMz + 1:
                    # mass is more than 1 mz larger than precursor mass in feature i. because scans2 is ordered by
                    # precursorMz we can break the loop
                    break
                mz_diff = abs(array[i].precursorMz - array[j].precursorMz)
                tr_diff = abs(array[i].retentionTime - array[j].retentionTime)
                if mz_diff < self.parameters["merge_features_mz_tol"] and \
                        tr_diff < self.parameters["merge_features_rt_tol"]:
                    included.add(array[j].num)  # Same group as feature i so we don't need to check it again
                    merged_list[-1].append(array[j])
        new_merged_list = []  # second pass to filter out recurring masses.
        for i in merged_list:
            if len(i) > 1:  # The way I am getting rid of always present ions leaves behind group with single feature
                new_merged_list.append(np.array(i))
            else:
                for deleted_mz in deleted_precursor_mz_list:
                    if abs(i[0].precursorMz - deleted_mz) < self.parameters["merge_features_mz_tol"]:
                        break
                else:
                    pass
                    self.logger.log(40, f"|| I FORGET WHAT IS THIS didn't find in deleted: {i[0]}")
        self.mergedMS2scans = new_merged_list
        self.logger.log(18, f"|| Merged all MSMS spectra in {len(new_merged_list)} features")

    def find_scans(self, mz, tr, dtr=0.1, dmz=0.03):  # returns mergedMS2scans[i] alike structure
        df2 = self.scans[not np.isnan(self.scans.precursorMz)]
        df3 = df2[(abs(df2.precursorMz - mz) < dmz) & (abs(df2.retentionTime - tr) < dtr)]
        if len(df3) == 0:
            return []
        col_names = ['num', 'retentionTime', 'basePeakMz', 'tempenergy', 'precursorMz', "precursorIntensity",
                     "filename"]
        return np.array(df3[col_names])

    def merged_msms_from_table(self, input_array: list, dtr: float = 0.5, dmz: float = 0.03):
        """
        It takes input_array and tries to find MS2 scans that are close enough. Calculation are saved in 
        self.mergedMS2scans (same as merge_features).
        :param self:
        :param input_array: list that looks like [[index, mz, tr], [index2, mz2, tr2], ...]. If table in csv file use
        f.read_simple_file_list
        :param dtr: max tr difference
        :param dmz: max mz difference
        :return: In order to keep old indexes we have to save indexes in return and call pass in generate table!
        """
        # index, mz, tr--> mz, tr have to be float!!!
        merged_msms2 = []
        old_index_list = []
        for i in input_array:
            t1 = self.find_scans(i[1], i[2], dtr, dmz)
            merged_msms2.append(t1)
            old_index_list.append(i[0])
        self.mergedMS2scans = merged_msms2
        return old_index_list

    # endregion

    def get_index_by_mz_rt(self, mz, tr):
        ret_list = []
        for id1 in range(len(self.aligned_dict["peak_data"])):
            line = self.aligned_dict["peak_data"][id1]
            if abs(mz - line[1]) < 0.5 and abs(tr - line[2]) < 0.5:
                ret_list.append((id1, line[1] - mz, line[2] - tr, 10 * (line[1] - mz) ** 2 + (line[2] - tr) ** 2))

        if len(ret_list) == 0:
            return -1
        ret_list = sorted(ret_list, key=lambda l: l[3], reverse=False)
        self.logger.log(20, f"| with mz={mz} and tr={tr} we obtain first 2 indices: {ret_list[:2]}")
        return ret_list[0][0]

    def plot_by_index(self, index, exp_file_name="picture", ha="h"):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_axes([0.1, 0.3, 0.85, 0.7])
        for i in range(len(self.filename)):
            name = self.filename[i]
            if not (("Eff" not in name) and ("inf" not in name.lower()) and ("Solvent" not in name)):
                continue
            if ha == "h":
                ax.bar(self.filename[i][:-8], self.aligned_dict["h"][index][i])
            elif ha == "A":
                ax.bar(self.filename[i][:-8], self.aligned_dict["A"][index][i])
            else:
                raise "wrong value for index!!"

        plt.xticks(rotation=90)
        plt.savefig(dep.output_path + exp_file_name + ".svg")
        fig.show()


# Functions #########################################

def empty_ret(time_list, intensity_list, comment="NOT_FOUND"):
    """
    works with find_peak2. When there is no peak it has to return empty list that is similar to ful list and this
    function does that. But still it returns time and intensity for plot
    :param time_list: list of times for plot
    :param intensity_list: list of intensity for plot
    :param comment: Comment will be shown in table and in picture.
    :return: list
    """
    return [[(np.nan, np.nan),  # (avg_mz, avg_mz_s),
             (np.nan, np.nan, np.nan),  # tr,
             (0, 0),  # (peak_h, peak_area),
             np.nan,  # avg_mz_list,
             time_list,  # time_list,
             intensity_list,  # intensity_list,
             comment,  # Comment for peak!
             0,  # start_index,
             1,  # end_index,
             [],  # specter_index
             np.nan  # isotopic ratio, first element are ratios and second are mass differences
             ]]


def find_interval(rt, time_list, peak_list, rt_tolerance):
    found_peaks = dict()
    for i, ti in enumerate(rt):
        z_scores = []
        for u in peak_list:
            t_start = time_list[u[0][0]]
            t_end = time_list[u[1][0]]
            if u[0][0] == u[1][0]:
                z_scores.append(100)
            else:
                z_scores.append(abs((2 * ti - (t_start + t_end)) / (t_end - t_start)))
        ind_low = z_scores.index(min(z_scores))
        if z_scores[ind_low] < rt_tolerance:
            peak_data_tuple = tuple(peak_list[ind_low])
            if peak_data_tuple not in found_peaks:
                found_peaks[peak_data_tuple] = [i]
            else:
                found_peaks[peak_data_tuple].append(i)
    return found_peaks


def find_peak_function(y, threshold, descending_threshold=0.95):
    """
    Algorithm work in 2 steps. Firstly it only takes maximum and minimum of intensity list and it generates a new list
    that is used in next step. For 2. step there are 2 modes: If descending_peak is False than program calculates as if
    top of peak is yet to come. It "creates" now peak if min between two maximum is below threshold (1). If new maximum
    is smaller than previous it has to be smaller than descending_threshold to be interpreted as descending of peak (2).
    If peak is descending (descending_peak == False) than it is split when intensity_i is much bigger than current
    minimum.
    :param y:
    :param threshold:
    :param descending_threshold:
    :return:
    """
    minmax = []
    y = [i - min(y) for i in y]
    ascending = False
    for i in range(len(y) - 1):
        if ((y[i + 1] - y[i]) > 0) != ascending:
            minmax.append((ascending, i, y[i]))
            ascending = not ascending
        elif y[i] == 0:
            ascending = False
        elif y[i] > 0 and y[i + 1] == 0:
            minmax.append((False, i + 1, y[i + 1]))
    if minmax[-1][0]:
        minmax.append((False, len(y) - 1, y[len(y) - 1]))
    peak_start = (minmax[0][1], minmax[0][2])
    previous_maximum = False
    previous_minimum = False
    peak_list = []
    descending_peak = False
    for i, row in enumerate(minmax):
        up_or_down, index, intensity_i = row
        if up_or_down:
            # "row" is maximum
            if previous_maximum:
                if intensity_i > previous_maximum[2] and not descending_peak:
                    # maximum when ascending, where intensity is larger that current maximum
                    if previous_minimum[2] / previous_maximum[2] <= threshold:
                        # (1)
                        temp1 = (previous_minimum[1], previous_minimum[2])
                        peak_list.append([peak_start, temp1])
                        peak_start = temp1
                        previous_minimum = False
                else:
                    if intensity_i / previous_maximum[2] < descending_threshold:
                        # (2)
                        descending_peak = True
                    if previous_minimum[2] / intensity_i <= threshold:
                        temp1 = (previous_minimum[1], previous_minimum[2])
                        peak_list.append([peak_start, temp1])
                        peak_start = temp1
                        previous_minimum = False
                        descending_peak = False

            previous_maximum = row
        else:
            if not previous_minimum:
                previous_minimum = row
            if not descending_peak or intensity_i < previous_minimum[2]:
                # this first minimum or we are ascending or it is the smallest minimum
                previous_minimum = row
    peak_list.append([peak_start, (previous_minimum[1], previous_minimum[2])])
    mzproject.logger_file.general_logger.log(10, f'||||| Obtained peak list: {peak_list}')
    return peak_list


def calculate_average(values, weights, remove_zeros=True, squared=True):
    values = np.array(values)
    weights = np.array(weights)
    if squared:
        weights = weights ** 2
    mask = np.isfinite(values) & np.isfinite(weights)
    if remove_zeros:
        mask = mask & (values != 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return np.nan, np.nan
    if sum(weights) == 0:
        return np.nan, np.nan
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return average, np.sqrt(variance)


def decode_zlib(string1, type1=np.float64):
    if string1 is None:
        return np.array([])
    decoded_source = base64.b64decode(string1.encode('ascii'))
    decoded_source = zlib.decompress(decoded_source)
    return structured_to_unstructured(np.frombuffer(bytearray(decoded_source),
                                                    dtype=np.dtype([('m/z array', type1),
                                                                    ('intensity array', type1)]).newbyteorder('>')))


# plt stuff!
def make_graph_grid():
    fig_original, ax_original = plt.subplots(2, 3, figsize=(15, 10), gridspec_kw={'width_ratios': [3, 3, 1]})
    fig_original.set_size_inches(15, 10, forward=True)
    ax_original[1, 0].ticklabel_format(useOffset=False, style='plain')
    ax_original[1, 1].ticklabel_format(useOffset=False, style='plain')
    ax_original[0, 2].axis('off')
    # ax_original[1, 2].axis('off')
    ax_original[0, 0].set_title("Chromatogram")
    ax_original[0, 1].set_title("Areas")
    ax_original[1, 1].set_title("m/z distribution")
    ax_original[1, 0].set_title("m/z time trend")
    ax_original[1, 2].set_title("Isotopic ratio")
    ax_original[1, 2].set_ylim(0, 1)
    ax_original[1, 2].set_xlim(0.0, 2.5)
    ax_original[1, 2].set_xticks([0, 1, 2])
    ax_original[1, 2].set_yticks(np.arange(0, 1.5, 1))
    ax_original[1, 2].grid()
    plt.setp(ax_original[0, 0], ylabel='h')
    plt.setp(ax_original[1, 0], ylabel='delta m/z')
    plt.setp(ax_original[1, 0], xlabel='retention time [min]')
    return fig_original, ax_original


def plot_graph(plot_list: typing.List[FindPeakReturnClass], names, folder, export_name, max_curves=18, show_plot=False,
               plt_object=(None, None),
               split_peak_group_obj: typing.Union[SplitFeatureDataClass, None] = None, subpeak_index=0,
               start_comment="", save_graph=True):
    if plt_object != (None, None):
        fig, ax = plt_object
    else:
        fig, ax = make_graph_grid()
    plot_objects = []
    colors = plt.cm.get_cmap("hsv")(np.linspace(0, 1, min(max_curves, len(plot_list)) + 1))
    max_h = max([m.peak_h for m in plot_list])
    for i in range(len(plot_list)):
        data_i = plot_list[i]
        label = str(i) + "_" + names[i] + "_" + data_i.comment
        if ("Noise" in data_i.comment) or ("TOO_FAR" in data_i.comment):
            plot_objects.append(ax[0, 0].plot(data_i.time_list, data_i.intensity_list, label=label,
                                              color=colors[i % max_curves], linestyle='--')[0])
        else:
            plot_objects.append(ax[0, 0].plot(data_i.time_list, data_i.intensity_list, label=label,
                                              color=colors[i % max_curves])[0])
        plot_objects.append(ax[0, 1].bar([i], data_i.peak_area, color=colors[i % max_curves]))
        if not np.isnan(data_i.avg_mz):
            if split_peak_group_obj is not None:
                plot_objects.append(
                    ax[0, 0].vlines(x=split_peak_group_obj.ms2_tr_dict.get((names[i], subpeak_index), []),
                                    ymin=0, ymax=max_h, color=colors[i % max_curves]))
                plot_objects.append(ax[1, 0].hlines(y=split_peak_group_obj.ms2_precursor_mz_dict.get((names[i],
                                                                                                      subpeak_index),
                                                                                                     []),
                                                    xmin=data_i.tr_min, xmax=data_i.tr_max,
                                                    color=colors[i % max_curves]))
            plot_objects.append(ax[0, 0].fill_between(data_i.time_list[data_i.start_index: data_i.end_index],
                                                      data_i.intensity_list[data_i.start_index: data_i.end_index],
                                                      0, alpha=0.2, color=colors[i % max_curves]))
            plot_objects.append(ax[1, 0].plot(data_i.time_list, data_i.avg_mz_list, color=colors[i % max_curves])[0])
            plot_objects.append(ax[1, 1].errorbar(np.array([i]), data_i.avg_mz, yerr=data_i.avg_mz_s,
                                                  marker="o", linestyle="none", c=colors[i % max_curves]))
            isotopic_rat = data_i.isotopes[0][:, 0]
            if sum(np.isnan(isotopic_rat)) != len(isotopic_rat):
                plot_objects.append(ax[1, 2].errorbar(np.array([0.5, 1, 1.5, 2, 3, 4]), isotopic_rat,
                                                      yerr=data_i.isotopes[0][:, 1], fmt='x',
                                                      color=colors[i % max_curves]))

        if (i % max_curves == max_curves - 1) or (i == len(plot_list) - 1):
            handles, labels = ax[0, 0].get_legend_handles_labels()
            plot_objects.append(fig.legend(handles, labels, title="File", loc='upper right'))  # , fontsize='x-small')
            fig.suptitle(f'{export_name}_{str(i // 15)}')
            ax[1, 0].set_xlim(ax[0, 0].get_xlim())
            ax[1, 1].set_ylim(ax[1, 0].get_ylim())
            if save_graph:
                if not os.path.isdir(folder):
                    os.mkdir(folder)
                fig.savefig(
                    folder + f'{start_comment}{export_name}_{str(i // max_curves)}.svg',
                    dpi=100)
            if show_plot:
                fig.show()
            for z in range(len(plot_objects)):
                plot_objects[z].remove()
            ax[0, 0].relim()
            ax[0, 0].autoscale()
            ax[0, 1].relim()
            ax[0, 1].autoscale()
            ax[1, 0].relim()
            ax[1, 0].autoscale()
            ax[1, 1].relim()
            ax[1, 1].autoscale()
            plot_objects = []
        i += 1


if __name__ == "__main__":
    pass
