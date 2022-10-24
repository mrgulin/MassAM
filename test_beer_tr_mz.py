import mzproject.class_file
import mzproject.functions as f
import numpy as np

files = f.get_files(list_key='beer')
# obj = mzproject.class_file.mzproject('Negative')
# obj.add_files_speed(files[:5])
# obj.filter_constant_ions(save_graph="")
# # obj.merge_features()
# obj.mergedMS2scans = np.array([[[0, 114.045, 0, 0, 1.15, 0, 0]]], dtype=float)
# obj.generate_table(force=True)

"""

['num', 'retentionTime', 'basePeakMz', 'tempenergy', 'precursorMz', "precursorIntensity", "filename"]
array(['Ball_EtOH_2_negxml__69601', 1.1598833333333334, 114.0580474807,
        10.0, 114.060989379883, 4886.215816497803, 'Ball_EtOH_2_negxml'],
       dtype=object)
['', tr, 0, 0, mz, 0, '']
"""

if __name__=="__main__":
    print(files)
    mz_tr_l = [
        [227.107754, 10.63],
        [199.0775851, 9.29],
        [199.0774954, 9.92]
]
    # screen_masses(files[:5], 'Negative', mz_tr_l)
