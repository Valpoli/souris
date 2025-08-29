# simplified_pipeline_by_plate.py
import os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ---------- I/O ----------
def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path, sep=None, engine="python", decimal=",")
    df.columns = [c.strip() for c in df.columns]
    return df

# ---------- Plate condition ----------
# Extraits "Hot Plate", "Room Temp", "Cold Plate" depuis la colonne File (ou met "Unknown")
PLATE_PAT = re.compile(r"Plate[\\/](Hot\s*Plate|Room\s*Temp|Cold\s*Plate)", re.IGNORECASE)
def parse_plate(file_str: str) -> str:
    s = str(file_str or "")
    m = PLATE_PAT.search(s)
    if not m: return "Unknown"
    val = m.group(1).lower().replace("  ", " ").strip()
    if "hot" in val:  return "Hot Plate"
    if "room" in val: return "Room Temp"
    if "cold" in val: return "Cold Plate"
    return "Unknown"

# ---------- Préparation ----------
NUM_CANDIDATES = [
    "Score",
    "Call Length (s)","Principal Frequency (kHz)","Low Freq (kHz)","High Freq (kHz)",
    "Delta Freq (kHz)","Frequency Standard Deviation (kHz)","Slope (kHz/s)",
    "Sinuosity","Mean Power (dB/Hz)","Tonality","Peak Freq (kHz)",
    "Begin Time (s)","End Time (s)"
]

def to_float_col(s):  # gère virgules françaises et NaN
    return pd.to_numeric(pd.Series(s, dtype="string").str.replace(",", ".", regex=False), errors="coerce")

def standardize(df):
    d = df.copy()
    if "File" in d.columns:
        d["plate"] = d["File"].map(parse_plate)
    else:
        d["plate"] = "Unknown"

    # cluster: réutilise s'il existe, sinon -1 (sera rempli par KMeans)
    if "cluster" in d.columns:
        d["cluster"] = pd.to_numeric(d["cluster"], errors="coerce").fillna(-1).astype(int)
    elif "Label" in d.columns:
        d["cluster"] = pd.Categorical(d["Label"]).codes.astype(int)
    else:
        d["cluster"] = -1

    # conversions numériques
    for c in [c for c in NUM_CANDIDATES if c in d.columns]:
        d[c] = to_float_col(d[c])

    # mid_s si temps présents (sinon laissé à NaN et on ne tracera pas les densités)
    if {"Begin Time (s)","End Time (s)"} <= set(d.columns):
        d["mid_s"] = 0.5*(d["Begin Time (s)"] + d["End Time (s)"])
    else:
        d["mid_s"] = np.nan
    return d

def build_features(d):
    use = [c for c in NUM_CANDIDATES if c in d.columns]
    if not use: raise ValueError("Aucune feature numérique trouvée (au moins 'Score' est requise).")
    X = d[use].fillna(0.0).to_numpy(dtype=float)
    return StandardScaler().fit_transform(X)

def ensure_clusters(d, n_clusters=2, seed=42):
    if (d["cluster"] >= 0).any():
        uniq = {c:i for i,c in enumerate(sorted(d.loc[d["cluster"]>=0,"cluster"].unique()))}
        d["cluster"] = d["cluster"].map(uniq).fillna(0).astype(int)
        return d, max(len(uniq), 1)
    X = build_features(d)
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    d["cluster"] = km.fit_predict(X)
    return d, n_clusters

# ---------- Couleurs & tracés ----------
def cluster_colors(K):  # inversion globale (Cl 0 = couleur la plus à droite)
    cmap = plt.get_cmap("tab10" if K <= 10 else "tab20")
    return [cmap((K-1-i) % cmap.N) for i in range(K)]

def save_scatter_pca(d, out_png, seed=42):
    X = build_features(d)
    Z = PCA(n_components=2, random_state=seed).fit_transform(X)
    K = int(d["cluster"].max()) + 1
    cols = cluster_colors(K)
    plt.figure(figsize=(6,5))
    for k in range(K):
        m = (d["cluster"]==k).to_numpy()
        plt.scatter(Z[m,0], Z[m,1], s=10, color=cols[k], label=f"Cl {k}", alpha=0.85)
    plt.title("PCA — AllPlates"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

def save_scatter_pca_by_plate(d, out_dir, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    X = build_features(d)
    Z = PCA(n_components=2, random_state=seed).fit_transform(X)
    for plate in sorted(d["plate"].unique()):
        m = (d["plate"]==plate).to_numpy()
        if not m.any(): continue
        K = int(d.loc[m,"cluster"].max()) + 1
        cols = cluster_colors(K)
        plt.figure(figsize=(6,5))
        for k in range(K):
            mk = m & (d["cluster"]==k).to_numpy()
            plt.scatter(Z[mk,0], Z[mk,1], s=10, color=cols[k], label=f"Cl {k}", alpha=0.85)
        plt.title(f"PCA — {plate}"); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"PCA_{plate.replace(' ','_')}.png"), dpi=300); plt.close()

def save_bars_fixed_scale(d, out_png):
    pivot = (d.assign(n=1).groupby(["plate","cluster"])["n"].sum()
             .unstack("cluster").fillna(0).astype(int))
    plates = list(pivot.index); clusters = list(pivot.columns)
    K = len(clusters); x = np.arange(len(plates)); width = 0.8/max(K,1)
    y_max = pivot.values.max() if pivot.size else 0
    cols = cluster_colors(K); color_by_cluster = {k: cols[i] for i,k in enumerate(clusters)}

    plt.figure(figsize=(max(6, len(plates)*0.9), 4.5))
    for xi, plate in enumerate(plates):
        row = pivot.loc[plate]
        ordre = row.sort_values(ascending=False).index  # max à gauche
        for j, k in enumerate(ordre):
            plt.bar(xi + j*width, row[k], width=width, color=color_by_cluster[k],
                    label=f"Cl {k}" if xi==0 else None)
    plt.xticks(x + (K-1)*width/2, plates, rotation=0)
    plt.ylabel("Calls"); plt.title("Calls per cluster per plate (max à gauche)")
    plt.ylim(0, y_max*1.15 if y_max else 1); plt.legend(fontsize=8, title="Cluster")
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

# ---------- MAIN ----------
def main(path, n_clusters=2, seed=42, out_dir="plots"):
    base = os.path.splitext(os.path.basename(path))[0]
    out_root = os.path.join(out_dir, base); os.makedirs(out_root, exist_ok=True)

    df = read_any(path)
    df = standardize(df)
    df, K = ensure_clusters(df, n_clusters=n_clusters, seed=seed)

    # PCA all + par condition
    save_scatter_pca(df, os.path.join(out_root, "PCA_AllPlates.png"), seed=seed)
    save_scatter_pca_by_plate(df, os.path.join(out_root, "embed_by_plate"), seed=seed)

    # Barres
    save_bars_fixed_scale(df, os.path.join(out_root, "bars_calls_per_plate.png"))

    # Densités temps: uniquement si mid_s dispo (Begin/End)
    if df["mid_s"].notna().any():
        from sklearn.neighbors import KernelDensity
        def density_kde(times, T=180.0, bins=180, bw=1.5):
            grid = np.linspace(0, T, bins); times = np.clip(times, 0, T)
            if len(times) >= 2:
                kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(times[:,None])
                d = np.exp(kde.score_samples(grid[:,None])); d *= len(times)/(np.trapezoid(d,grid)+1e-12)
            else:
                d = np.zeros_like(grid)
            return grid, d
        K = int(df["cluster"].max())+1; cols = cluster_colors(K)
        den_dir = os.path.join(out_root,"density"); os.makedirs(den_dir, exist_ok=True)

        # All plates
        ymax = 1e-9
        curves = []
        for k in range(K):
            t = df.loc[df["cluster"]==k, "mid_s"].dropna().to_numpy()
            x,y = density_kde(t); curves.append(("All",k,x,y)); ymax = max(ymax, y.max() if len(y) else 0)
        plt.figure(figsize=(8,4))
        for _,k,x,y in curves: plt.plot(x,y,color=cols[k],label=f"Cl {k}")
        plt.title("Density — AllPlates"); plt.ylim(0, ymax*1.05 if ymax>0 else 1)
        plt.xlabel("Time (s)"); plt.ylabel("≈ calls/s"); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(den_dir,"density_AllPlates.png"), dpi=300); plt.close()

        # Par plate, même échelle
        plates = sorted(df["plate"].unique())
        for p in plates:
            ymax = 1e-9; curves=[]
            sub = df[df["plate"]==p]
            for k in range(K):
                t = sub.loc[sub["cluster"]==k, "mid_s"].dropna().to_numpy()
                x,y = density_kde(t); curves.append((p,k,x,y)); ymax = max(ymax, y.max() if len(y) else 0)
            plt.figure(figsize=(8,4))
            for _,k,x,y in curves: plt.plot(x,y,color=cols[k],label=f"Cl {k}")
            plt.title(f"Density — {p}"); plt.ylim(0, ymax*1.05 if ymax>0 else 1)
            plt.xlabel("Time (s)"); plt.ylabel("≈ calls/s"); plt.legend(fontsize=8); plt.tight_layout()
            plt.savefig(os.path.join(den_dir,f"density_{p.replace(' ','_')}.png"), dpi=300); plt.close()

    print(f"✅ Figures enregistrées dans: {out_root}")

if __name__ == "__main__":
    folder = "plate_inputs_csv"
    import glob
    files = glob.glob(os.path.join(folder, "*.xlsx")) + glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        print(f"Aucun fichier trouvé dans {folder}")
    else:
        for f in files:
            print(f"\n=== Traitement de {f} ===")
            main(f, out_dir="plots/plate")
