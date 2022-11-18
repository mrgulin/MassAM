#%%

import mzproject.featgen.class_file
import mzproject.dependencies as dep
import mzproject.functions as f
import mzproject.featgen.experiment
mzproject.featgen.class_file.set_logging_levels(10, 30)
#%%

QC_files = f.get_files(folder_path=dep.input_path)
QC_files = [i for i in QC_files if "QC_MIX2" in i or 'Blank_EtOH' in i]
print(QC_files)

#%%

# os.mkdir(r'C:\Users\tinc9\Documents\IJS-offline\start_to_end/manual_python')
dep.output_path = r'C:\Users\tinc9\Documents\IJS-offline\start_to_end/manual_python/'

obj = mzproject.featgen.class_file.MzProject()

#%%

obj.add_files_speed(QC_files)
obj.filter_constant_ions(False, 'Filtered_masses.png')
obj.merge_features()
obj.generate_table(True, 10, force=True)
obj.merge_duplicate_rows(reindex=True)


class MSProject:
    pass
