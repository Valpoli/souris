import h5py
import numpy as np
import csv

def lire_plages(mat_path, dataset_path="#refs#/K", freqs_in_khz=True, csv_out="plages.csv"):
    with h5py.File(mat_path, "r") as f:
        M = np.array(f[dataset_path])  # 4 x N

    t_start = M[0, :]                 # secondes
    f_start = M[1, :]                 # kHz (supposé)
    t_len   = M[2, :]                 # secondes
    f_len   = M[3, :]                 # kHz (supposé)

    t_end = t_start + t_len
    f_end = f_start + f_len

    # Conversion Hz/kHz
    f_start_hz = f_start * 1e3 if freqs_in_khz else f_start
    f_end_hz   = f_end   * 1e3 if freqs_in_khz else f_end

    # Sauvegarde dans un CSV
    with open(csv_out, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "t_start_s", "t_end_s", "f_start_hz", "f_end_hz"])
        for i in range(t_start.size):
            writer.writerow([i+1, t_start[i], t_end[i], f_start_hz[i], f_end_hz[i]])

    return t_start, t_end, f_start_hz, f_end_hz


# Exemple :
t0, t1, f0, f1 = lire_plages("test1.mat", dataset_path="#refs#/K", freqs_in_khz=True, csv_out="plages.csv")

print(f"✅ Sauvegardé {t0.size} détections dans plages.csv")
