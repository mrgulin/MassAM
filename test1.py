import numpy as np
import mzproject.paths as paths
from mzproject.featgen import experiment
import mzproject.functions as f

f_list_neg1 = f.get_files()

qc_features1 = experiment.read_qc_features(r"C:\Users\tinc9\Documents\IJS-offline\QC\QC_compounds_final.csv")

experiment.run_experiment_python([i for i in f_list_neg1 if "QC_MIX" in i], 'test_python', 'neg',
                                 subdirectory='test_python', qc_features=qc_features1, compare_with_qc=True,
                                 limit_mz=(200, 250))
