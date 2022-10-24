import mzproject.functions as f
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from pyopls import OPLS  # https://github.com/BiRG/pyopls#validation
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import r2_score, accuracy_score


def read_aligned_table(path='E:/Users/Gulin/Documents/IJS-offline/FCM_2020/beer_simulant/row_filtered_v1-h.csv'):
    df_table = pd.read_csv(path)
    feature_name = "i=" + df_table["index"].astype(str) + "_mz=" + round(df_table["mz"], 4).astype(
        str) + "_tr=" + round(df_table["tr"], 1).astype(str)
    df_table["feature_name"] = feature_name
    df_table = df_table.set_index("feature_name")
    only_vals_df = df_table[df_table.columns[10:]]
    only_vals_df.transpose()
    return only_vals_df.transpose()


def transform_data(x):
    """
    Function that transforms heights/areas into data for pls
    :param x:
    :return:
    """
    if x == 0:
        x = 1
    return np.log10(x)


def create_input(df, type_of_classification):
    types = [str, str, str, int, int, int, int]
    types2 = ["|U40", "|U40", "|U40", int, int, int, int]
    t1_data, t1_header = f.read_simple_file_list(r"E:\Users\Gulin\Documents\IJS-offline\Python scripts\names_pls.csv",
                                                 types, header=True)
    dtypes = [(t1_header[i], types2[i]) for i in range(len(t1_header))]
    t1_data = [tuple(i) for i in t1_data]
    t1 = np.array(t1_data, dtype=dtypes)

    df = df[df.index.isin(t1["filename"][t1["training_set"] != 0])]
    df = df.applymap(transform_data)  # log10
    df.apply(lambda x: x - x.mean())  # centering
    # TODO: scaling! -> mogoče niti ne ker je razlika 1 en velikostni razred, razlika 2 pa 2 velikostna razreda ?

    if type(type_of_classification) == str:
        type_of_classification = (type_of_classification,)
    master_prediction_list = dict()
    for j in type_of_classification:
        curr_prediction_list = []
        for i in df.index:
            index2 = np.where(t1["filename"] == i)[0][0]
            curr_prediction_list.append(t1[j][index2])
            print(index2)
        master_prediction_list[j] = curr_prediction_list
    print(master_prediction_list)
    print(df.index)
    return df, master_prediction_list


class Model_OPLS:
    def __init__(self):
        pass

    def create_model_and_train(self):
        pass


spectra, target = create_input(read_aligned_table(), "Sample_blank")
target = target["Sample_blank"]
print(spectra[0:2])
print(target[0:2], type(target))
spectra = [[1.375, -1.375, -0.65],
           [0.375, -0.875, -0.15],
           [-0.625, 0.625, -0.05],
           [-1.125, 1.625, 0.85]]

spectra = pd.DataFrame(data=np.array(spectra), index=list(range(4)), columns=["a", "b", "c"])
target = [
    [-0.985, -0.25],
    [0.015, 1.75],
    [1.015, -1.25],
    [-0.045, -0.25]]
target = [0, 0, 1, 1]

opls = OPLS(2)
Z = opls.fit_transform(spectra, target)

pls = PLSRegression(2)
y_pred = cross_val_predict(pls, spectra, target, cv=LeaveOneOut())
q_squared = r2_score(target, y_pred)  # -0.107
dq_squared = r2_score(target, np.clip(y_pred, 0, 1))  # -0.106
accuracy = accuracy_score(target, np.sign(y_pred))  # 0.705

processed_y_pred = cross_val_predict(pls, Z, target, cv=LeaveOneOut())
processed_q_squared = r2_score(target, processed_y_pred)  # 0.981
processed_dq_squared = r2_score(target, np.clip(processed_y_pred, 0, 1))  # 0.984
processed_accuracy = accuracy_score(target, np.sign(processed_y_pred))  # 1.0

r2_X = opls.score(spectra)  # 7.8e-12 (most variance is removed)

fpr, tpr, thresholds = roc_curve(target, y_pred)
roc_auc = roc_auc_score(target, y_pred)
proc_fpr, proc_tpr, proc_thresholds = roc_curve(target, processed_y_pred)
proc_roc_auc = roc_auc_score(target, processed_y_pred)

plt.scatter(target, y_pred)
for i in range(len(processed_y_pred)):
    plt.annotate(spectra.index[i], (target[i], y_pred[i]))
plt.show()

plt.scatter(target, processed_y_pred)
for i in range(len(processed_y_pred)):
    plt.annotate(spectra.index[i], (target[i], processed_y_pred[i]))
plt.show()
