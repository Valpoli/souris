# simplified_pipeline_by_session.py
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
import umap

# ----------------- I/O minimal -----------------

def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        # CSV avec éventuelles virgules décimales
        df = pd.read_csv(path, sep=None, engine="python", decimal=",")
    df.columns = [c.strip() for c in df.columns]
    return df

# S1/S2… depuis le nom de fichier ou le chemin (fallback: Session1/Session2)
SESSION_PATTERNS = [
    re.compile(r"[\\/_\s]S(\d+)\b", re.IGNORECASE),        # "... F1 S1.mat"
    re.compile(r"[\\/_]Session\s*(\d+)\b", re.IGNORECASE), # ".../Session1/..."
]

def parse_session(file_str: str) -> str:
    s = str(file_str)
    for pat in SESSION_PATTERNS:
        m = pat.search(s)
        if m:
            return f"S{m.group(1)}"
    return "S?"  # inconnu

def standardize_minimal(df):
    df = df.copy()

    # temps
    df["begin_s"] = pd.to_numeric(df.get("Begin Time (s)"), errors="coerce")
    df["end_s"]   = pd.to_numeric(df.get("End Time (s)"),   errors="coerce")

    # session (remplace 'day')
    if "File" in df.columns:
        df["session"] = df["File"].map(parse_session).astype(str)
    else:
        df["session"] = "S?"

    # cluster (réutilise si présent, sinon map Label → ints)
    if "cluster" in df.columns:
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(-1).astype(int)
    elif "Label" in df.columns:
        cats = pd.Categorical(df["Label"])
        df["cluster"] = pd.Series(cats.codes).astype(int)  # 0..K-1
    else:
        df["cluster"] = -1

    # mid-times
    df["mid_s"] = 0.5 * (df["begin_s"] + df["end_s"])

    df = df.dropna(subset=["begin_s", "end_s", "mid_s"])
    return df

# ----------------- Features & embed uniques (AllSessions) -----------------

FEATURES = [
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

def build_features(df):
    use = [c for c in FEATURES if c in df.columns]
    X = df[use].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    X = StandardScaler().fit_transform(X)
    return X

def ensure_clusters(df, n_clusters=4, random_state=42):
    if (df["cluster"] >= 0).any():
        # normalise en 0..K-1
        uniq = {c:i for i,c in enumerate(sorted(df.loc[df["cluster"]>=0,"cluster"].unique()))}
        df["cluster"] = df["cluster"].map(uniq).fillna(0).astype(int)
        return df, max(len(uniq), 1)
    # sinon KMeans rapide
    X = build_features(df)
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    df["cluster"] = km.fit_predict(X)
    return df, n_clusters

def compute_embeddings_all(X, random_state=42):
    pca = PCA(n_components=2, random_state=random_state)
    Zp = pca.fit_transform(X)
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
    sessions = sorted(df["session"].astype(str).unique())

    # AllSessions + per-session, même Y
    curves, all_curves = [], []
    for k in range(K):
        t = df.loc[df["cluster"]==k, "mid_s"].to_numpy()
        x, y = density_kde(t, T, bins, bandwidth)
        all_curves.append((k, x, y)); curves.append(("AllSessions", k, x, y))
    y_max = max(y.max() for _,_,_,y in curves)

    for sess in sessions:
        d = df[df["session"]==sess]
        for k in range(K):
            t = d.loc[d["cluster"]==k, "mid_s"].to_numpy()
            x, y = density_kde(t, T, bins, bandwidth)
            curves.append((sess, k, x, y))
            y_max = max(y_max, y.max())

    # plot AllSessions
    plt.figure(figsize=(8,4))
    for k, x, y in all_curves:
        plt.plot(x, y, color=colors[k], label=f"Cl {k}")
    plt.title("Density — AllSessions")
    plt.xlabel("Time (s)"); plt.ylabel("≈ calls/s"); plt.ylim(0, y_max*1.05)
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "density_AllSessions.png"), dpi=300); plt.close()

    # plot per-session, même Y
    for sess in sessions:
        plt.figure(figsize=(8,4))
        for (_sess, k, x, y) in curves:
            if _sess != sess: continue
            plt.plot(x, y, color=colors[k], label=f"Cl {k}")
        plt.title(f"Density — {sess}")
        plt.xlabel("Time (s)"); plt.ylabel("≈ calls/s"); plt.ylim(0, y_max*1.05)
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"density_{sess}.png"), dpi=300); plt.close()

def save_bars_fixed_scale(df, out_png):
    # tableau sessions x clusters (comptes)
    pivot = (df.assign(n=1)
               .groupby(["session","cluster"])["n"].sum()
               .unstack("cluster").fillna(0).astype(int))

    sessions = list(pivot.index)
    clusters = list(pivot.columns)            # pour la légende
    K = len(clusters)
    color_list = cluster_colors(K)
    color_by_cluster = {k: color_list[i] for i, k in enumerate(clusters)}

    x = np.arange(len(sessions))
    width = 0.8 / max(K, 1)
    y_max = pivot.values.max() if pivot.size else 0

    plt.figure(figsize=(max(6, len(sessions)*0.9), 4.5))

    # on trace session par session, en triant les clusters par valeur décroissante
    for xi, sess in enumerate(sessions):
        row = pivot.loc[sess]
        ordre = row.sort_values(ascending=False).index  # plus grand à gauche
        for j, k in enumerate(ordre):
            plt.bar(xi + j*width,
                    row[k],
                    width=width,
                    color=color_by_cluster[k],
                    label=f"Cl {k}" if xi == 0 else None)  # légende une seule fois

    # axe & légende
    plt.xticks(x + (K-1)*width/2, sessions)
    plt.ylabel("Calls")
    plt.title("Calls per cluster per session (trié: max à gauche)")
    plt.ylim(0, y_max * 1.15 if y_max else 1)
    plt.legend(fontsize=8, title="Cluster")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)

def save_pairwise_plots(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    K = int(df["cluster"].max()) + 1
    colors = cluster_colors(K)
    cols = [c for c in [
        "Call Length (s)","Principal Frequency (kHz)","Low Freq (kHz)",
        "High Freq (kHz)","Delta Freq (kHz)","Frequency Standard Deviation (kHz)",
        "Slope (kHz/s)","Tonality","Peak Freq (kHz)","begin_s","end_s"
    ] if c in df.columns]

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

def main(path, n_clusters=4, T=180.0, bins=180, bandwidth=1.5, random_state=42, out_dir="plots"):
    base = os.path.splitext(os.path.basename(path))[0]
    out_root = os.path.join(out_dir, base)
    os.makedirs(out_root, exist_ok=True)

    df = read_any(path)
    df = standardize_minimal(df)
    df, K = ensure_clusters(df, n_clusters=n_clusters, random_state=random_state)
    save_pairwise_plots(df, os.path.join(out_root, "comparaison_plot"))

    # Embeddings sur ALL SESSIONS, puis filtres par session avec mêmes coords
    X = build_features(df)
    Zp, Zu = compute_embeddings_all(X, random_state=random_state)

    emb_dir = os.path.join(out_root, "embed"); os.makedirs(emb_dir, exist_ok=True)
    save_scatter(Zp, df, f"PCA — {base} — AllSessions", os.path.join(emb_dir, "PCA_AllSessions.png"))
    save_scatter(Zu, df, f"UMAP — {base} — AllSessions", os.path.join(emb_dir, "UMAP_AllSessions.png"))

    for sess in sorted(df["session"].astype(str).unique()):
        m = (df["session"]==sess).to_numpy()
        save_scatter(Zp[m], df.loc[m], f"PCA — {base} — {sess}", os.path.join(emb_dir, f"PCA_{sess}.png"))
        save_scatter(Zu[m], df.loc[m], f"UMAP — {base} — {sess}", os.path.join(emb_dir, f"UMAP_{sess}.png"))

    # Barres (échelle fixe)
    save_bars_fixed_scale(df, os.path.join(out_root, "bars_calls_per_session.png"))

    # Densités (AllSessions + per-session, même échelle)
    save_densities_fixed_scale(df, os.path.join(out_root, "density"), T=T, bins=bins, bandwidth=bandwidth)

    print(f"✅ Figures enregistrées dans: {out_root}")

if __name__ == "__main__":
    folder = "double_aversion_csv"
    import glob
    files = glob.glob(os.path.join(folder, "*.csv")) + glob.glob(os.path.join(folder, "*.xlsx"))
    if not files:
        print(f"Aucun fichier trouvé dans {folder}")
    else:
        for f in files:
            print(f"\n=== Traitement de {f} ===")
            main(f, out_dir="plots/double_aversion")
