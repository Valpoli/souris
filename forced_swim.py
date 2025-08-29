# simplified_pipeline.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
import umap

# ----------------- I/O minimal -----------------

def read_csv_simple(path):
    df = pd.read_csv(path, sep=None, engine="python", decimal=",")
    df.columns = [c.strip() for c in df.columns]
    return df

def standardize_minimal(df):
    df = df.copy()
    # temps
    df["begin_s"] = pd.to_numeric(df["Begin Time (s)"], errors="coerce")
    df["end_s"]   = pd.to_numeric(df["End Time (s)"],   errors="coerce")
    df["day"]     = df["Day"].astype(str)
    # cluster
    if "cluster" in df.columns:
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(-1).astype(int)
    elif "Label" in df.columns:
        # encode labels en ints 0..K-1
        uniq = {lab:i for i, lab in enumerate(pd.Categorical(df["Label"]).categories)}
        df["cluster"] = df["Label"].map(uniq).astype(int)
    else:
        df["cluster"] = -1
    # mid-times
    df["mid_s"] = 0.5*(df["begin_s"] + df["end_s"])
    df = df.dropna(subset=["begin_s","end_s","mid_s","day"])
    return df

# ----------------- Features & embed uniques (AllDays) -----------------

def build_features(df):
    # petit set robuste et bref
    cols = [
        "Call Length (s)",
        "Principal Frequency (kHz)",
        "Low Freq (kHz)",
        "High Freq (kHz)",
        "Delta Freq (kHz)",
        "Frequency Standard Deviation (kHz)",
        "Slope (kHz/s)",
        "Tonality",
        "Peak Freq (kHz)",
        "begin_s",
        "end_s",
    ]
    use = [c for c in cols if c in df.columns]
    X = df[use].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    X = StandardScaler().fit_transform(X)
    return X

def ensure_clusters(df, n_clusters=4, random_state=42):
    if (df["cluster"] >= 0).any():
        # normalise en 0..K-1
        uniq = {c:i for i,c in enumerate(sorted(df.loc[df["cluster"]>=0,"cluster"].unique()))}
        df["cluster"] = df["cluster"].map(uniq).fillna(0).astype(int)
        return df, len(uniq) or n_clusters
    # sinon KMeans rapide
    X = build_features(df)
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    df["cluster"] = km.fit_predict(X)
    return df, n_clusters

def compute_embeddings_all(X, random_state=42):
    pca = PCA(n_components=2, random_state=random_state)
    Zp = pca.fit_transform(X)
    # um = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=random_state)
    um = umap.UMAP(random_state=random_state)
    Zu = um.fit_transform(X)
    return Zp, Zu

# ----------------- Tracés concis -----------------

def cluster_colors(K):
    cmap = plt.get_cmap("tab10" if K <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(K)]

def save_scatter(Z, df, title, out_png):
    K = int(df["cluster"].max()) + 1
    colors = cluster_colors(K)
    plt.figure(figsize=(6,5))
    for k in range(K):
        m = (df["cluster"]==k).to_numpy()
        plt.scatter(Z[m,0], Z[m,1], s=10, color=colors[k], label=f"Cl {k}", alpha=0.85)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def density_kde(times, T=180.0, bins=180, bandwidth=1.5):
    grid = np.linspace(0, T, bins)
    times = np.clip(times, 0, T)
    if len(times) >= 2:
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(times[:,None])
        d = np.exp(kde.score_samples(grid[:,None]))
        d *= len(times) / (np.trapezoid(d, grid) + 1e-12)
    else:
        d = np.zeros_like(grid)
        for t in times:
            d += np.exp(-0.5*((grid - t)/bandwidth)**2)
        if d.sum() > 0:
            d *= len(times) / (np.trapezoid(d, grid) + 1e-12)
    return grid, d

def save_densities_fixed_scale(df, out_dir, T=180.0, bins=180, bandwidth=1.5):
    os.makedirs(out_dir, exist_ok=True)
    K = int(df["cluster"].max()) + 1
    colors = cluster_colors(K)
    days = sorted(df["day"].unique())

    # AllDays + per-day, même Y
    curves = []

    # AllDays
    all_curves = []
    for k in range(K):
        t = df.loc[df["cluster"]==k, "mid_s"].to_numpy()
        x, y = density_kde(t, T, bins, bandwidth)
        all_curves.append((k, x, y))
        curves.append(("AllDays", k, x, y))
    y_max = max(y.max() for _,_,_,y in curves)  # init avec AllDays
    # per-day
    for day in days:
        d = df[df["day"]==day]
        for k in range(K):
            t = d.loc[d["cluster"]==k, "mid_s"].to_numpy()
            x, y = density_kde(t, T, bins, bandwidth)
            curves.append((day, k, x, y))
            y_max = max(y_max, y.max())

    # plot AllDays
    plt.figure(figsize=(8,4))
    for k, x, y in all_curves:
        plt.plot(x, y, color=colors[k], label=f"Cl {k}")
    plt.title("Density — AllDays")
    plt.xlabel("Time (s)"); plt.ylabel("≈ calls/s"); plt.ylim(0, y_max*1.05)
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "density_AllDays.png"), dpi=300); plt.close()

    # plot per-day, même Y
    for day in days:
        plt.figure(figsize=(8,4))
        for (_day, k, x, y) in curves:
            if _day != day: continue
            plt.plot(x, y, color=colors[k], label=f"Cl {k}")
        plt.title(f"Density — {day}")
        plt.xlabel("Time (s)"); plt.ylabel("≈ calls/s"); plt.ylim(0, y_max*1.05)
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"density_{day}.png"), dpi=300); plt.close()

def save_bars_fixed_scale(df, out_png):
    pivot = (df.assign(n=1).groupby(["day","cluster"])["n"].sum()
             .unstack("cluster").fillna(0).astype(int))
    days = list(pivot.index)
    K = pivot.shape[1]
    colors = cluster_colors(K)
    x = np.arange(len(days))
    width = 0.8 / max(K,1)
    y_max = pivot.values.max()

    plt.figure(figsize=(max(6, len(days)*0.9), 4.5))
    for i, k in enumerate(pivot.columns):
        plt.bar(x + i*width, pivot[k].values, width=width, color=colors[i], label=f"Cl {k}")
    plt.xticks(x + (K-1)*width/2, days)
    plt.ylabel("Calls"); plt.title("Calls per cluster per day")
    plt.ylim(0, y_max * 1.15)
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)

def save_pairwise_plots(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    K = int(df["cluster"].max()) + 1
    colors = cluster_colors(K)
    cols = ["Call Length (s)","Principal Frequency (kHz)","Low Freq (kHz)",
            "High Freq (kHz)","Delta Freq (kHz)","Frequency Standard Deviation (kHz)",
            "Slope (kHz/s)","Tonality","Peak Freq (kHz)","begin_s","end_s"]
    cols = [c for c in cols if c in df.columns]
    D = df.copy()
    for c in cols: D[c] = pd.to_numeric(D[c], errors="coerce")
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            x, y = cols[i], cols[j]
            plt.figure(figsize=(5,4))
            for k in range(K):
                m = (df["cluster"]==k) & D[[x,y]].notna().all(1)
                plt.scatter(D.loc[m,x], D.loc[m,y], s=8, alpha=0.7, color=colors[k], label=f"Cl {k}")
            plt.xlabel(x); plt.ylabel(y); plt.legend(fontsize=7)
            plt.tight_layout()
            fname = f"{sanitize(y)}_vs_{sanitize(x)}.png"
            plt.savefig(os.path.join(out_dir, fname), dpi=300)
            plt.close()

# ----------------- MAIN (une seule entrée) -----------------

def main(csv_path, n_clusters=4, T=180.0, bins=180, bandwidth=1.5, random_state=42, out_dir="plots"):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_root = os.path.join(out_dir, base)
    os.makedirs(out_root, exist_ok=True)

    df = read_csv_simple(csv_path)
    df = standardize_minimal(df)
    df, K = ensure_clusters(df, n_clusters=n_clusters, random_state=random_state)
    save_pairwise_plots(df, os.path.join(out_root, "comparaison_plot"))

    # Embeddings sur ALL DAYS, puis filtres par jour avec mêmes coords
    X = build_features(df)
    Zp, Zu = compute_embeddings_all(X, random_state=random_state)

    emb_dir = os.path.join(out_root, "embed"); os.makedirs(emb_dir, exist_ok=True)
    save_scatter(Zp, df, f"PCA — {base} — AllDays", os.path.join(emb_dir, "PCA_AllDays.png"))
    save_scatter(Zu, df, f"UMAP — {base} — AllDays", os.path.join(emb_dir, "UMAP_AllDays.png"))

    for day in sorted(df["day"].unique()):
        m = (df["day"]==day).to_numpy()
        save_scatter(Zp[m], df.loc[m], f"PCA — {base} — {day}", os.path.join(emb_dir, f"PCA_{day}.png"))
        save_scatter(Zu[m], df.loc[m], f"UMAP — {base} — {day}", os.path.join(emb_dir, f"UMAP_{day}.png"))

    # Barres (échelle fixe)
    save_bars_fixed_scale(df, os.path.join(out_root, "bars_calls_per_day.png"))

    # Densités (AllDays + per-day, même échelle)
    save_densities_fixed_scale(df, os.path.join(out_root, "density"), T=T, bins=bins, bandwidth=bandwidth)

    print(f"✅ Figures enregistrées dans: {out_root}")

if __name__ == "__main__":
    folder = "forced_swim_csv"
    import glob

    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        print(f"Aucun fichier .csv trouvé dans {folder}")
    else:
        for f in csv_files:
            print(f"\n=== Traitement de {f} ===")
            main(f, out_dir="plots/forced_swim")
