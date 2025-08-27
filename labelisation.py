import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------- Paramètres ----------
in_xlsx = "FSS_test02.xlsx"
keep_groups = {"CTRL", "NMS"}
feature_cols = [
    "Score","Begin Time (s)","End Time (s)","Call Length (s)",
    "Principal Frequency (kHz)","Low Freq (kHz)","High Freq (kHz)","Delta Freq (kHz)",
    "Frequency Standard Deviation (kHz)","Slope (kHz/s)","Sinuosity",
    "Mean Power (dB/Hz)","Tonality","Peak Freq (kHz)"
]
cutoff_khz = 40.0          # seuil pour la méthode cut-off
kmeans_k = 2               # nb de clusters pour KMeans
kmeans_random_state = 42

# --------- Lecture + nettoyage (virgules décimales) ----------
df = pd.read_excel(in_xlsx, dtype=str)
df = df[df["Experimental Group"].isin(keep_groups)].copy()

# Convertir les colonnes numériques (remplacer virgule -> point)
for c in feature_cols:
    df[c] = df[c].str.replace(",", ".", regex=False).astype(float)

# --------------- 1) Méthode CUT-OFF ----------------
df_cut = df.copy()
df_cut["cluster"] = (df_cut["Principal Frequency (kHz)"] > cutoff_khz).astype(int)
df_cut["cluster"] = df_cut["cluster"].replace({1: 1, 0: 2})  # 1 si > seuil, sinon 2

# Sauvegarde par groupe
for g in sorted(keep_groups):
    out_path = f"cut_off_{g}.csv"
    df_cut[df_cut["Experimental Group"] == g].to_csv(out_path, index=False)
    print(f"✅ sauvegardé: {out_path}")

# --------------- 2) KMeans non supervisé ----------------
X = df[feature_cols].values
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=kmeans_k, n_init=10, random_state=kmeans_random_state)
labels = kmeans.fit_predict(X_scaled)

df_unsup = df.copy()
df_unsup["cluster"] = labels + 1  # 1..K (au lieu de 0..K-1)

# Sauvegarde par groupe
for g in sorted(keep_groups):
    out_path = f"unsup_{g}.csv"
    df_unsup[df_unsup["Experimental Group"] == g].to_csv(out_path, index=False)
    print(f"✅ sauvegardé: {out_path}")
