
import os.path

import Workbooks.Simulant_experiment
from MzProject import experiment
import mzproject.functions as f
import mzproject.merge_tables
QC_features, f_list_neg, f_list_pos = experiment.get_files_QC_features()
file_list = f.get_files(list_key="beer_simulant")

mzproject.merge_tables.do_all(r"C:/Users/tinc9/Documents/IJS-offline/Experiment/simulant_neg_low_mz/", file_list,
              ["python_beer_simulant_neg_lowmz"], "neg", True,
              tuple(), "Experiment/simca_results/simca_simulant_neg_lowmz",
                           "General-List_VIP_opls-da_mzLow.txt", "General-List_s-plot_opls-da_mzLow.txt")
