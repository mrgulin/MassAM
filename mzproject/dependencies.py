import mzproject.paths
import numpy as np

input_path = mzproject.paths.input_path
output_path = mzproject.paths.output_path
IJS_ofline_path = mzproject.paths.IJS_ofline_path
silent = True


def get_dtype(m_string):
    return [("index", int), ("mz", float), ("tr", float), ("scans", f"<U{m_string[0] + 20}"),
            ("noise", f"<U{m_string[1] + 20}"), ("too_far", f"<U{m_string[2] + 20}"), ("noise_ratio1", float),
            ("noise_ratio2", int), ("comment", f"<U{m_string[3] + 50}"), ("M_plus_1", float)]


def change_input(new_path):
    global input_path
    input_path = new_path


def change_output(new_path):
    global output_path
    output_path = new_path

m_p = 1.00727647  # Da, g/mol, amu
MIX_mz = {"16dBPA": 244.215458, "10dcarbamazepin": 246.15773, "E1": 270.16198, "E2-Alpha": 272.17763,
       "E2-Beta": 272.17763, "EE2": 296.17763, "E3": 288.172545, "Ketoprofen": 254.094294,
       "Caffeine": 194.080376, "Ibuprofen": 206.13068, "Naproxen": 230.094294, "Diclofenac": 295.016684,
       "22BPF": 200.08373, "BPAF": 336.058499, "24BPF": 200.08373, "44BPF": 200.08373, "BPE": 214.09938,
       "BPA": 228.11503, "BPC": 256.14633, "BPM": 346.19328, "BPPH": 380.17763, "BPP": 346.19328,
       "BPBP": 352.14633, "BPB": 242.13068, "BP26DM": 284.17763, "BPC2": 280.005785, "BPZ": 268.14633,
       "BPFL": 350.13068, "BPAP": 290.13068, "BPS": 250.02998, "BADGE": 340.167459, "BFDGE": 312.136159,
       "BADGEx2H2O": 376.188589}
delta_mass_pos_proton = np.array([+1 * m_p])
delta_mass_pos_adducts = {"M+H": 1.007276, "M+NH4": 18.033823, "M+Na": 22.989218, "M+CH3OH+H": 33.033489,
                          "M+K": 38.963158}
delta_mass_neg_proton = np.array([-1 * m_p])
delta_mass_neg_adducts = {"M-H": -1.007276, "M-H2O-H": -19.01839, "M+Cl": +34.969402, "M+FA-H": +44.998201,
                          "M-H-CO2": -43.989829239-m_p}
