from subprocess import PIPE, Popen
import matplotlib.pyplot as plt
import numpy as np
from os import path
import mzproject.paths
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO

##################################
# There are written functions that call CFM-ID functions
# TODO: implement also metfrag!
#################################


# Location of CFM-ID folder

cfm_id_path = mzproject.paths.cfm_id_path
temporary_file_path = mzproject.paths.temporary_save_path
spectrum_folder = mzproject.paths.spectrum_folder
modeldict = {"ce+": [cfm_id_path + r"\metab_ce_cfm\param_output0.log", cfm_id_path + r"\metab_ce_cfm\param_config.txt"],
             "se+": [cfm_id_path + r"\metab_se_cfm\param_output0.log", cfm_id_path + r"\metab_se_cfm\param_config.txt"],
             "se-": [cfm_id_path + r"\negative_metab_se_cfm\param_output0.log",
                     cfm_id_path + r"\negative_metab_se_cfm\param_config.txt"]}
# CFM-ID info!
# https://sourceforge.net/p/cfm-id/wiki/Home/
#     This sourceforge provides code for CFM-ID 2.0 only. CFM -ID 3.0 has recently been released, and provides a wrapper
#     to the functionality of CFM-ID 2.0 that can be accessed at http://cfmid3.wishartlab.com/. In cases where the
#     spectrum for a molecule has been measured, it uses that, and if it is from one of the 21 classes of lipid listed
#      below then a separate rule-based fragmenter is used, otherwise CFM 2.0 is used. Source code for the rule based
#       fragmenter can be found at https://bitbucket.org/wishartlab/msrb-fragmenter/.
# Which model should I use?
#     There a several pre-trained CFM models available at
#     https://sourceforge.net/p/cfm-id/code/HEAD/tree/supplementary_material/trained_models/.Which model to use should
#     be dictated by the MS setup you want to use.
#     If you are using positive mode ESI-MS/MS data, please use either metab_ce_cfm or metab_se_cfm
#     (and select param_output0.log).
#     If you are using negative mode ESI-MS/MS data, please use negative_metab_se_cfm (param_output0.log).
#     Make sure you take BOTH the param_output and param_config file from the corresponding model.
#
# cfm-predict.exe OC(C1=CC(C(O)=O)=CC(C(O)=O)=C1)=O 0.01 <param_file> <config_file> <annotate_fragments>
# <output_file_or_dir> <apply_postproc> <suppress_exceptions>


def predictspectra(smiles, threshold=0.001, model="se+", save="", silent=True):
    """
    returnes dictionary of spectra at 3 different energy levels. It transforms smiles string to dictionary of spectra.
    Other inputs are important for CFM-ID and already preseted.
    :param silent:
    :param save: If changed to string function saves export to file with name of string. example: save="testout.txt")
    :param threshold: CFM-ID parameter
    :param smiles: string of structure in smiles
    :param model: ce+, se+, se-
    :return: dict Dictionary of spectra
    """

    # CFM-ID needs folder of trained data for prediction. There are different models for different type of ionization.
    # modeldict is there to set right model file for given model

    # When running program from Popen every parameter is element in list to processlist is creating list for Popen
    print("Smiles: ", smiles)
    processlist = [cfm_id_path + r"\cfm-predict.exe", smiles, str(threshold), modeldict[model][0],
                   modeldict[model][1]]
    print(" ".join(processlist))
    # Popen runs CFM-ID in background and it is saving its return to pipe
    pipe = Popen(processlist, stdout=PIPE, stderr=PIPE)
    # extracting returned text from pipe
    text, error = pipe.communicate()
    # Transforming text from binary to UTF-8
    text = text.decode("UTF-8")
    invalid_outputs = ["Could not parse input:", "Unsupported input molecule", "no location found for charge",
                       "Could not ionize - already charged molecule and didn't know what to do here",
                       "Could not sanitize input"]
    if any(x in text for x in invalid_outputs):
        return {'couldnt calculate': [[0], [0]]}
    # Default parameter of save is False --> if statement does not execute. If we change save when calling to string it
    # Executes beacuse boolean("non empty string") is True.
    if save:
        # Opening file and writing return in it
        outfile = open(save, "w", encoding="UTF-8")
        outfile.write(text)
        outfile.close()
    # Right now text is string. If we want to extract data we have to transform it into list
    textlist = text.split("\n")
    # Splitting it per line
    currlevel = "error"
    # currlevel is keeping track to which energy level current spectrum belongs
    specterdict = dict()
    # specterdict is dictionary which will be returned
    for i in range(len(textlist)):
        line = textlist[i].strip()
        # Getting rid of tabs and new lines
        if not silent:
            print(line)
        if "energy" in line:
            # if there is "energy" in line it means that following lines are going to be spectrum from this energy
            currlevel = model + "-" + line
            specterdict[currlevel] = [[], []]
        elif line != "":
            # code is splitting line of text with space and adding mz and intensity to list
            line = line.split(" ")
            specterdict[currlevel][0].append(float(line[0]))
            specterdict[currlevel][1].append(float(line[1]))
    return specterdict


def plot_cfm_id_msms(dictionary):
    """
    Function for plotting dictionary created by function predictspectra
    :param dictionary: return of predictspectra
    :return: plt plot
    """
    # there are usually 3 energy levels so we need 3 colours (red, green, blue)
    colors = ["r", "g", "b"]
    j = 0
    # getting trough every element in dictionary and plotting ms spcetrum
    for k, i in dictionary.items():
        plt.stem(i[0], i[1], colors[j], markerfmt=colors[j] + "o", label=k)
        j += 1
    plt.legend()
    plt.show()


def annotatespectra(smiles, filepath, model, ppmdif=20, absdif=0.01, fileout="", print_on_console=False):
    """
    In offline version of CFM-ID there is no function that could fragment compound and annotate sprectrum. So Annotating
    is done by function above. FOr input we also need save from fragmetspectra so it gets mz values
    :param print_on_console: To print text in console
    :param smiles: SMILES
    :param filepath: cfm_id_path to save of returned predicted spectra-could also be another spectrum if in right format
    :param model: ce+, se+, se-
    :param ppmdif: max ppm difference to be treated as same peak
    :param absdif: absolute mz difference to be treated as same peak
    :param fileout: cfm_id_path to export file without extension. At default it doesn't save file
    :return: returnes AnnotatedSpectrum class
    """
    processlist = [cfm_id_path + r"\cfm-annotate.exe", smiles, filepath, "doesn't matter :)", str(ppmdif), str(absdif),
                   modeldict[model][0], modeldict[model][1]]
    # processlist = [cfm_id_path + r"\cfm-annotate.exe", smiles, filepath, "AN_ID", str(ppmdif), str(absdif)]
    pipe = Popen(processlist, stdout=PIPE)
    text, sth_else = pipe.communicate()
    text = text.decode("UTF-8")
    if fileout:
        outfile = open(fileout + "_raw.txt", "w", encoding="UTF-8")
        outfile.write(text)
        outfile.close()
    if print_on_console:
        print(text)
    section_list_index = 0
    spectrum_dict = {"energy0": [], "energy1": [], "energy2": []}
    section_list = ["TARGET ID:", "energy0", "energy1", "energy2", "", "", "", ""]
    number_of_features = -1
    fragment_list = []
    losses_list = []
    id_name = ""
    for line in text.split("\n"):
        line = line.strip()
        if section_list[section_list_index + 1] == line:
            section_list_index = 1 + section_list_index
            continue
        if section_list_index == 0:
            id_name = line.split(":")[1]
        elif section_list_index in [1, 2, 3]:
            p1 = line.find("(")
            specter = line[:p1].split(" ")
            intensities = line[p1 + 1: -1].split(" ")
            for i in range(2, len(specter)):
                if specter[i]:
                    specter[i] = (int(specter[i]), float(intensities[i - 2]))
            specter[0] = float(specter[0])
            specter[1] = float(specter[1])
            specter.remove("")
            spectrum_dict[f"energy{section_list_index - 1}"].append(specter)
        elif section_list_index == 4:
            if number_of_features == -1:
                number_of_features = int(line)
            else:
                fragment = line.split(" ")
                fragment_list.append([int(fragment[0]), float(fragment[1]), " ".join(fragment[2:])])
        elif section_list_index == 5:
            loss = line.split(" ")
            losses_list.append([int(loss[0]), int(loss[1]), loss[2]])
    if len(fragment_list) != number_of_features:
        raise NameError('Length of fragment_list is not matching')
    return AnnotatedSpectrum(id_name, spectrum_dict, fragment_list, losses_list)


def draw_smiles(smiles='CCCC', x_size=20, y_size=20):
    """
    Based on SMILES it returnes png picture in binary data.
    :param smiles: SMILES
    :param x_size: width of picture in pixels
    :param y_size: height of picutre in pixels
    :return:
    """
    m = Chem.MolFromSmiles(smiles)
    d2d = Draw.MolDraw2DCairo(int(x_size), int(y_size))
    d2d.drawOptions().clearBackground = True
    Draw.PrepareAndDrawMolecule(d2d, m)
    d2d.FinishDrawing()
    png = d2d.GetDrawingText()
    bio = BytesIO(png)
    im = plt.imread(bio)
    return im


def fragmentAndAnnotate(smiles, threshold=0.001, model="se-", ppmdif=20, absdif=0.01, fileout=""):
    predictspectra(smiles, threshold, model, save=temporary_file_path + "tempfile1")
    anotate_spectra = annotatespectra(smiles, temporary_file_path + "tempfile1", model, ppmdif=ppmdif, absdif=absdif,
                                      fileout=fileout)
    anotate_spectra.smiles = smiles
    return anotate_spectra


def get_insilico(cas, smiles, mode, folder=spectrum_folder, threashold=0.001):
    path1 = folder + f"/{cas}_{mode}.txt"
    if path.isfile(path1):
        conn = open(path1, "r")
        ret = eval(conn.read())
        conn.close()
    else:
        ret = predictspectra(smiles, threashold, model=mode)
        conn = open(path1, "w")
        conn.write(repr(ret))
        conn.close()
    return ret


class AnnotatedSpectrum:
    def __init__(self, id_name, spectrum_dict, fragment_list, losses_list):
        self.smiles = None
        self.id = id_name
        self.spectrum_dict = spectrum_dict
        self.fragment_list = fragment_list
        self.losses_list = losses_list

    def plot_spectrum(self, energy_level="energy0"):
        colors = plt.cm.get_cmap("hsv")(np.linspace(0, 1, len(self.fragment_list) + 1))
        fig, ax = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'height_ratios': [5, 2], 'width_ratios': [5, 4]})
        ms_plot_ax = ax[0, 0]
        loss_plot_ax = ax[1, 0]
        ax[0, 1].axis('off')
        ax[1, 1].axis('off')

        # ms_plt_ax
        plot_list = self.spectrum_dict[energy_level]
        plot_data = []
        mz_vals = np.array([i[0] for i in plot_list])
        max_h = max([i[1] for i in plot_list])
        x_corr = np.linspace(min(mz_vals), max(mz_vals), len(mz_vals))
        for i in range(len(plot_list)):
            index_label = []
            for j in range(2, len(plot_list[i])):
                mz = plot_list[i][0]
                h = plot_list[i][j][1]
                H = plot_list[i][1]
                ind = plot_list[i][j][0]
                plot_data.append(np.array([mz, h, H, ind]))
                ms_plot_ax.vlines(mz, 0, h, linestyles="solid", colors=colors[ind])
                index_label.append(str(ind))
            # plt.text(mz, h, s=self.fragment_list[ind][2], horizontalalignment='center', verticalalignment='center', )
            ms_plot_ax.annotate(", ".join(index_label),
                                xy=(mz, h), xycoords='data',
                                xytext=(x_corr[i], max_h - np.random.random() * (max_h - h)),
                                arrowprops=dict(arrowstyle="->", color='black'), ha='center', size=20
                                )
            if self.fragment_list[ind][0] != ind:
                raise ValueError("Ne bo šlo takole :s")
        fig.suptitle(self.smiles)

        # loss_plot_ax
        plot_list = []
        for i in range(len(self.losses_list)):
            from_index = self.losses_list[i][0]
            to_index = self.losses_list[i][1]
            if from_index != self.fragment_list[from_index][0] or to_index != self.fragment_list[to_index][0]:
                print("Problem! Indexes in fragment_list are not set right!!")
            from_mass = self.fragment_list[from_index][1]
            to_mass = self.fragment_list[to_index][1]
            print(from_mass, " -> ", to_mass)
            plot_list.append((from_mass, to_mass, self.losses_list[i][2]))
        color_losses_dict = dict()
        unique_losses = list(set([i[2] for i in self.losses_list]))
        colors_losses = plt.cm.get_cmap("hsv")(np.linspace(0, 1, len(unique_losses) + 1))
        for i in range(len(unique_losses)):
            color_losses_dict[unique_losses[i]] = colors_losses[i]
        m = 0
        for line in list(set(plot_list)):
            b, c, d = line
            loss_plot_ax.hlines(y=m, xmax=b, xmin=c, label=d, color=color_losses_dict[d])
            m -= 1
        loss_plot_ax.set_xticks(list(set([i[1] for i in self.fragment_list])), minor=True)
        loss_plot_ax.xaxis.grid(True, which='minor')
        loss_plot_ax.set_xlim(ms_plot_ax.get_xlim())
        handles, labels = loss_plot_ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        loss_plot_ax.legend(*zip(*unique))

        # picture plotting
        x_size, y_size = fig.get_size_inches() * fig.dpi
        n = len(self.fragment_list)
        floor_square = np.floor(np.sqrt(n))
        ncols = floor_square
        nrows = np.ceil(n / ncols)
        for i in range(n):
            structure_image = draw_smiles(self.fragment_list[i][2], x_size=x_size * 0.4 / ncols,
                                          y_size=y_size / nrows)
            position_x = int(0.6 * x_size + x_size * 0.4 / ncols * (i % ncols))
            position_y = int(y_size / nrows * (i // ncols))
            loss_plot_ax.figure.figimage(structure_image, position_x, position_y, zorder=1)
            fig.text(position_x, position_y, s=str(i), transform=None, size=20, zorder=10)
        fig.show()
        return fig, ax


class Spectrum:
    def __init__(self, array, axis):
        if axis == 0:
            #  That means [[mz, i], [mz, i], ...]
            self.max_mz = round(max([i[0] for i in array]))
            self.min_mz = round(min([i[0] for i in array]))
            self.vector = np.zeros(self.max_mz + 1 - self.min_mz)
            for i in array:
                if i[1] != 0 and i[0] != -1:
                    self.vector[round(i[0]) - self.min_mz] += (i[1] ** 2) * (i[0] ** 0.5)  # i^1/2 * mz ^ 2
        elif axis == 1:
            # That means [[mz, mz, ...], [i, i, ...]]
            self.max_mz = round(max(array[0]))
            self.min_mz = round(min(array[0]))
            self.vector = np.zeros(self.max_mz + 1 - self.min_mz)
            for i in range(len(array[0])):
                curr_val = (array[1][i] ** 2) * (array[0][i] ** 0.5)
                if not np.isnan(curr_val):
                    self.vector[round(array[0][i]) - self.min_mz] += curr_val

    def dot_product(self, other):
        min_mz1 = max(self.min_mz, other.min_mz)
        max_mz1 = min(self.max_mz, other.max_mz)
        dot = 0
        if sum(self.vector) == 0 or sum(other.vector) == 0:
            return 0
        for mz in range(min_mz1, max_mz1 + 1):
            if (mz < self.min_mz) or (mz < other.min_mz):
                continue
            if (mz > self.max_mz) or (mz > other.max_mz):
                break
            dot += self.vector[mz - self.min_mz] * other.vector[mz - other.min_mz]
        dot /= (np.sqrt(sum(self.vector * self.vector)) * np.sqrt(sum(other.vector * other.vector)))
        return dot


if __name__ == "__main__":
    # https://moonbooks.org/Articles/How-to-insert-an-image-a-picture-or-a-photo-in-a-matplotlib-figure/
    # https://chemistry.stackexchange.com/questions/43299/is-there-a-way-to-use-free-software-to-convert-smiles-strings-to-structures
    pass
    predictspectra("CCCC")
    # a = fragmentAndAnnotate("NCCCOOCCOC")
    # a.plot_spectrum()
    # image = draw_smiles("asdasdad")
    # plt.figimage(image, 0, 0, zorder=1)
    # rsp1 = [[100, 1], [101, 2], [200, 200]]
    # rsp2 = [[105, 3], [101, 2], [200, 140]]
    # sp2 = Spectrum(rsp2, axis=0)
    # sp1 = Spectrum(rsp1, axis=0)
    # sp1.dot_product(sp2)
    # predictspectra("[Ti++].CCCC[OH-].CC(C)[OH-]", 0.001, "se-")
