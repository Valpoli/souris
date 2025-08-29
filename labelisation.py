import os, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

FEATS = ["Score","Begin Time (s)","End Time (s)","Call Length (s)",
         "Principal Frequency (kHz)","Low Freq (kHz)","High Freq (kHz)","Delta Freq (kHz)",
         "Frequency Standard Deviation (kHz)","Slope (kHz/s)","Sinuosity",
         "Mean Power (dB/Hz)","Tonality","Peak Freq (kHz)"]

to_float = lambda s: pd.to_numeric(s.astype(str).str.replace(",",".",regex=False), errors="coerce")

def process(inp, outdir, groups=("CTRL","NMS"), cutoff=40.0, k=2, seed=42):
    df = pd.read_excel(inp, dtype=str)
    df = df[df["Experimental Group"].isin(groups)].copy()
    for c in FEATS: df[c] = to_float(df[c])
    cut = df.copy(); cut["cluster"] = (cut["Principal Frequency (kHz)"]>cutoff).astype(int).replace({0:2,1:1})
    X = StandardScaler().fit_transform(df[FEATS].fillna(0))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    unsup = df.copy(); unsup["cluster"] = km.labels_+1
    os.makedirs(outdir, exist_ok=True)
    for g in groups:
        cut[cut["Experimental Group"]==g].to_csv(f"{outdir}/cut_off_{g}.csv", index=False)
        unsup[unsup["Experimental Group"]==g].to_csv(f"{outdir}/unsup_{g}.csv", index=False)
    print(f"✅ fichiers écrits dans {outdir}")

if __name__=="__main__":
    folder = "forced_swim_csv"
    csv = "FSS_test02.xlsx"
    process(csv, folder)
