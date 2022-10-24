from MzProject import experiment
from multiprocessing import Process

# region generate file lists and QC_features

QC_features, f_list_neg, f_list_pos = experiment.get_files_QC_features()


# endregion


def run_python_experiments(suffix, QC_features, comment_additional, number_of_files=3, **kwargs):
    """

    :param suffix:
    :param QC_features:
    :param kwargs: specify: from_table, save_peak_graphs, change_dict
    :return:
    """
    random_files = ["Ajd_inf_2_negxml", "CP_Beer_2_negxml", "Union_CP_3_negxml", "Old_StaroP_3_negxml",
                       "Ig_Eff_2_negxml", "CP_MeOH_2_negxml"]
    files_list =[
        [i for i in f_list_neg if ("QC_MIX" in i) or ("Blank_EtOH" in i)],
        [i for i in f_list_neg if ("Beer_MIX2_SPE" in i) or ("Blank_Beer" in i) or ("Beer_SPE_MIX2" in i)],
        [i for i in f_list_neg if ("QC_MIX1" in i) or ("Blank_EtOH" in i)] + random_files,
        [i for i in f_list_neg if ("QC_MIX1" in i) or ("Blank_EtOH" in i)],
        [i for i in f_list_neg if ("QC_MIX" in i) or ("Blank_EtOH" in i)] + random_files,
        [i for i in f_list_neg if ("Eff_MIX2_SPE" in i) or ("Blank_Water" in i) or ("Eff_SPE_MIX2" in i)]
    ]
    names_list = [
        f"EtOH_QC{'_' * (len(suffix) > 0)}{suffix}",
        f"Beer_QC{'_' * (len(suffix) > 0)}{suffix}",
        f"EtOH_QC_MIX1_random{'_' * (len(suffix) > 0)}{suffix}",
        f"EtOH_QC_MIX1{'_' * (len(suffix) > 0)}{suffix}",
        f"EtOH_QC_random{'_' * (len(suffix) > 0)}{suffix}",
        f"WW_QC{'_' * (len(suffix) > 0)}{suffix}"
    ]
    comment_list = [
        f"Experiment from QC files, {comment_additional}",
        f"Experiment from Beer QC files, {comment_additional}",
        f"Experiment from QC files, with 6 random files, only lower concentration, {comment_additional}",
        f"Experiment from QC files, only lower concentration, {comment_additional}",
        f"Experiment from QC files, with 6 random files, {comment_additional}",
        f"Experiment from  Wastewater QC files, {comment_additional}"
    ]
    file_group_list_list = [
        ("MIX2", "MIX1", "Blank"),
        ("MIX2_SPE", "SPE_MIX2", "Blank"),
        ("MIX1", "Blank"),
        ("MIX1", "Blank"),
        ("MIX2", "MIX1", "Blank"),
        ("MIX2_SPE", "SPE_MIX2", "Blank")
    ]

    process_list = []
    for i in range(number_of_files):
        keywords = {"name": names_list[i], "polarity": "neg", "subdirectory":"Experiment/neg2/",
                    "QC_features":QC_features, "comment": comment_list[i], "file_group_list":file_group_list_list[i]}
        keywords.update(kwargs)
        print(keywords)
        p = Process(target=experiment.run_one_experiment, args=(files_list[i],), kwargs=keywords)
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()


# MZMINE!
def run_MZmine_experiments(suffix, QC_features, comment_additional, number_of_files=3, protocol="MSMS_peaklist_builder",
                           change_dict=None):
    if change_dict is None:
        change_dict = dict()

    if type(number_of_files) == int:
        number_of_files = range(number_of_files)

    random_files = ["Ajd_inf_2_negxml", "CP_Beer_2_negxml", "Union_CP_3_negxml", "Old_StaroP_3_negxml",
                       "Ig_Eff_2_negxml", "CP_MeOH_2_negxml"]
    files_list =[
        [i for i in f_list_neg if ("QC_MIX" in i) or ("Blank_EtOH" in i)],
        [i for i in f_list_neg if ("Beer_MIX2_SPE" in i) or ("Blank_Beer" in i) or ("Beer_SPE_MIX2" in i)],
        [i for i in f_list_neg if ("QC_MIX1" in i) or ("Blank_EtOH" in i)] + random_files,
        [i for i in f_list_neg if ("QC_MIX1" in i) or ("Blank_EtOH" in i)],
        [i for i in f_list_neg if ("QC_MIX" in i) or ("Blank_EtOH" in i)] + random_files,
        [i for i in f_list_neg if ("Eff_MIX2_SPE" in i) or ("Blank_Water" in i) or ("Eff_SPE_MIX2" in i)]
    ]
    names_list = [
        f"EtOH_QC",
        f"Beer_QC",
        f"EtOH_QC_MIX1_random",
        f"EtOH_QC_MIX1",
        f"EtOH_QC_random",
        f"WW_QC"
    ]

    suffix1 = {"MSMS_peaklist_builder":"_MSMSplb", "ADAP_builder": "_ADAP"}[protocol]
    names_list = ["MZmine_" + i + suffix1 + f"{'_' * (len(suffix) > 0)}{suffix}" for i in names_list]
    comment_additional = "MZmine from MSMS_peaklist_builder, " + comment_additional
    comment_list = [
        f"Experiment from QC files, {comment_additional}",
        f"Experiment from Beer QC files, {comment_additional}",
        f"Experiment from QC files, with 6 random files, only lower concentration, {comment_additional}",
        f"Experiment from QC files, only lower concentration, {comment_additional}",
        f"Experiment from QC files, with 6 random files, {comment_additional}",
        f"Experiment from  Wastewater QC files, {comment_additional}"
    ]
    file_group_list_list = [
        ("MIX2", "MIX1", "Blank"),
        ("MIX2_SPE", "SPE_MIX2", "Blank"),
        ("MIX1", "Blank"),
        ("MIX1", "Blank"),
        ("MIX2", "MIX1", "Blank"),
        ("MIX2_SPE", "SPE_MIX2", "Blank")
    ]

    process_list = []
    for i in number_of_files:
        p = Process(target=experiment.run_one_experiment_mzmine,
                    args=(names_list[i], "Experiment/neg2/", files_list[i], file_group_list_list[i], QC_features, "neg",
                          comment_list[i]), kwargs={"protocol":protocol, "change_dict":change_dict})
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()
    """for i in range(number_of_files):
        experiment.run_one_experiment_mzmine(names_list[i], "Experiment/neg2/", files_list[i], file_group_list_list[i],
                                             QC_features, "neg", comment_list[i], protocol=protocol,
                                             change_dict=change_dict)"""
# use of these functions went to Workbooks/parameter_opt.py