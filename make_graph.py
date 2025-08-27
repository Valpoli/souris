import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

# UMAP est optionnel : si absent, on saute proprement
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# ---------- Utils I/O & standardisation ----------

def _read_csv_smart(path):
    """
    Lecture robuste : séparateur auto, support du décimal ','
    """
    try:
        df = pd.read_csv(path, sep=None, engine="python", decimal=",")
    except Exception:
        df = pd.read_csv(path)  # fallback
    df.columns = [c.strip() for c in df.columns]
    return df

def _infer_group_from_path(path):
    name = os.path.basename(path).lower()
    if "nms" in name:
        return "NMS"
    if "ctrl" in name or "control" in name or "controle" in name:
        return "CTRL"
    return "GROUP"

def _infer_day_from_df_or_name(df, path):
    # colonne explicite ?
    for cand in ["day", "Day", "jour", "Jour"]:
        if cand in df.columns:
            return df[cand].astype(str)
    # sinon: parse du nom de fichier
    m = re.search(r"day\s*([0-9]+)", os.path.basename(path), flags=re.I)
    if m:
        return pd.Series([f"Day{m.group(1)}"] * len(df))
    # défaut: Day1
    return pd.Series(["Day1"] * len(df))

def _standardize_columns(df, path):
    """
    Retourne un DataFrame avec colonnes standard :
      begin_s, end_s, f_start_hz, f_end_hz, cluster (optionnel), day
    """
    df = df.copy()

    # Temps début/fin
    time_pairs = [
        ("Begin Time (s)", "End Time (s)"),
        ("Begin time (s)", "End time (s)"),
        ("begin_s", "end_s"),
        ("t_start_s", "t_end_s"),
        ("t_start", "t_end"),
        ("start", "end"),
    ]
    begin_col = end_col = None
    for a, b in time_pairs:
        if a in df.columns and b in df.columns:
            begin_col, end_col = a, b
            break
    if begin_col is None:
        # parfois on a durée ; on tente "start" et une "duration"
        if "start" in df.columns and "duration" in df.columns:
            df["begin_s"] = df["start"].astype(float)
            df["end_s"]   = df["start"].astype(float) + df["duration"].astype(float)
        else:
            raise ValueError("Colonnes de temps introuvables (ex: 'Begin Time (s)'/'End Time (s)')")
    else:
        df["begin_s"] = df[begin_col].astype(str).str.replace(",", ".", regex=False).astype(float)
        df["end_s"]   = df[end_col].astype(str).str.replace(",", ".", regex=False).astype(float)

    # Fréquences
    fstart = fend = None
    for a, b in [("f_start_hz","f_end_hz"), ("f_start","f_end")]:
        if a in df.columns and b in df.columns:
            fstart, fend = a, b
            break
    if fstart is None:
        # essayer kHz
        for a, b in [("f_start_khz","f_end_khz"), ("f_min","f_max")]:
            if a in df.columns and b in df.columns:
                df["f_start_hz"] = pd.to_numeric(df[a], errors="coerce") * 1000.0
                df["f_end_hz"]   = pd.to_numeric(df[b], errors="coerce") * 1000.0
                break
    else:
        df["f_start_hz"] = pd.to_numeric(df[fstart], errors="coerce")
        df["f_end_hz"]   = pd.to_numeric(df[fend], errors="coerce")

    # jour
    df["day"] = _infer_day_from_df_or_name(df, path)

    # cluster si présent
    for c in ["cluster","Cluster","label","Label"]:
        if c in df.columns:
            df["cluster"] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            break

    # drop lignes invalides
    df = df.dropna(subset=["begin_s","end_s"])
    return df

def _build_features(df):
    """
    Construit un petit set de features robustes pour PCA/UMAP/Clustering.
    Si f_start_hz/f_end_hz manquent, on se rabat sur les features temporelles.
    """
    dur = (df["end_s"] - df["begin_s"]).to_numpy()
    start = df["begin_s"].to_numpy()
    feats = [dur, start]

    if "f_start_hz" in df.columns and "f_end_hz" in df.columns:
        f0 = df["f_start_hz"].to_numpy()
        f1 = df["f_end_hz"].to_numpy()
        f_center = 0.5 * (f0 + f1)
        f_bw = (f1 - f0)
        feats += [f0, f1, f_center, f_bw]

    X = np.vstack(feats).T
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def _ensure_clusters(df, n_clusters=4, random_state=0):
    if "cluster" in df.columns and df["cluster"].notna().any():
        # normalise en int 0..K-1
        uniq = {c:i for i,c in enumerate(sorted(df["cluster"].dropna().unique()))}
        df["cluster"] = df["cluster"].map(uniq).astype("Int64")
        return df, len(uniq)
    # sinon, KMeans
    X = _build_features(df)
    Xs = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    cl = kmeans.fit_predict(Xs)
    df["cluster"] = cl
    return df, n_clusters


# ---------- Tracés ----------

def _cluster_colors(n):
    N = max(int(n), 3)
    base = "tab20" if n > 10 else "tab10"
    cmap = mpl.colormaps.get_cmap(base).resampled(N)
    return [cmap(i / (N - 1)) for i in range(n)]

def plot_umap_and_pca_per_day(df, group, out_dir, n_neighbors=15, min_dist=0.1):
    os.makedirs(out_dir, exist_ok=True)
    days = sorted(df["day"].unique())
    n_clusters = int(df["cluster"].max()) + 1
    colors = _cluster_colors(n_clusters)

    for day in days:
        d = df[df["day"] == day]
        X = _build_features(d)
        Xs = StandardScaler().fit_transform(X)

        # PCA
        pca = PCA(n_components=2, random_state=0)
        Zp = pca.fit_transform(Xs)

        plt.figure(figsize=(6,5))
        for k in range(n_clusters):
            mask = (d["cluster"] == k).to_numpy()
            plt.scatter(Zp[mask,0], Zp[mask,1], s=15, label=f"Cluster {k}", color=colors[k], alpha=0.8)
        plt.title(f"PCA — {group} {day}")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{day}_PCA.png"), dpi=300)
        plt.close()

        # UMAP (si dispo)
        if HAS_UMAP:
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=0)
            Zu = reducer.fit_transform(Xs)
            plt.figure(figsize=(6,5))
            for k in range(n_clusters):
                mask = (d["cluster"] == k).to_numpy()
                plt.scatter(Zu[mask,0], Zu[mask,1], s=15, label=f"Cluster {k}", color=colors[k], alpha=0.8)
            plt.title(f"UMAP — {group} {day}")
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{day}_UMAP.png"), dpi=300)
            plt.close()
        else:
            warnings.warn("umap-learn non installé : UMAP ignoré pour " + f"{group} {day}")

def _density_series(times_s, T=180.0, bins=180, bandwidth=1.5):
    """
    Retourne (grid, density) sur [0,T].
    - Par défaut KDE (KernelDensity) si >= 2 points, sinon histogramme lissé léger.
    - bandwidth ~ échelle secondes.
    """
    grid = np.linspace(0, T, bins)
    times_s = np.clip(times_s, 0, T)

    if len(times_s) >= 2:
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(times_s[:, None])
        log_d = kde.score_samples(grid[:, None])
        d = np.exp(log_d)
        # normalisation pour intégrale ~ nb d'appels
        d *= len(times_s) / (np.trapezoid(d, grid) + 1e-12)
    else:
        # fallback: delta(s) -> petit bump gaussien
        d = np.zeros_like(grid)
        for t in times_s:
            d += np.exp(-0.5*((grid - t)/bandwidth)**2)
        if d.sum() > 0:
            d *= len(times_s) / (np.trapezoid(d, grid) + 1e-12)
    return grid, d

def plot_density_per_day(df, group, out_dir, T=180.0, bins=180, bandwidth=1.5):
    os.makedirs(out_dir, exist_ok=True)
    days = sorted(df["day"].unique())
    n_clusters = int(df["cluster"].max()) + 1
    colors = _cluster_colors(n_clusters)

    # mid-times des appels
    df = df.copy()
    df["mid_s"] = 0.5*(df["begin_s"] + df["end_s"])

    for day in days:
        d = df[df["day"] == day]

        # --------- version superposée
        plt.figure(figsize=(8,4))
        for k in range(n_clusters):
            t = d.loc[d["cluster"]==k, "mid_s"].to_numpy()
            x, y = _density_series(t, T=T, bins=bins, bandwidth=bandwidth)
            plt.plot(x, y, label=f"Cluster {k}", color=colors[k])
        plt.title(f"Densité des appels (0–{int(T)} s) — {group} {day}")
        plt.xlabel("Temps (s)")
        plt.ylabel("Densité (≈ appels/s)")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{day}_density_overlay.png"), dpi=300)
        plt.close()

        # --------- version facettes (une ligne par cluster)
        rows = n_clusters
        fig, axes = plt.subplots(rows, 1, figsize=(8, 1.8*rows), sharex=True)
        if rows == 1:
            axes = [axes]
        for k, ax in enumerate(axes):
            t = d.loc[d["cluster"]==k, "mid_s"].to_numpy()
            x, y = _density_series(t, T=T, bins=bins, bandwidth=bandwidth)
            ax.plot(x, y, color=colors[k])
            ax.set_ylabel(f"Cl {k}")
            ax.grid(alpha=0.2)
            ax.set_xlim(0, T)
        axes[0].set_title(f"Densité des appels par cluster — {group} {day}")
        axes[-1].set_xlabel("Temps (s)")
        fig.tight_layout(h_pad=0.2)
        fig.savefig(os.path.join(out_dir, f"{day}_density_facets.png"), dpi=300)
        plt.close(fig)

def plot_cluster_evolution_bars(df, group, out_dir):
    """
    Barres : nb d'appels par cluster et par jour.
    """
    os.makedirs(out_dir, exist_ok=True)
    pivot = (
        df.assign(count=1)
          .groupby(["day","cluster"])["count"].sum()
          .unstack("cluster")
          .fillna(0)
          .astype(int)
    )
    days = list(pivot.index)
    clusters = list(pivot.columns)
    n_clusters = len(clusters)
    colors = _cluster_colors(n_clusters)

    x = np.arange(len(days))
    width = 0.8 / max(n_clusters, 1)

    plt.figure(figsize=(max(6, len(days)*0.9), 4.5))
    for i, k in enumerate(clusters):
        plt.bar(x + i*width, pivot[k].values, width=width, label=f"Cluster {k}", color=colors[i])
    plt.title(f"Évolution des appels par cluster — {group}")
    plt.xticks(x + (n_clusters-1)*width/2, days, rotation=0)
    plt.ylabel("Nombre d'appels")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"summary_evolution_clusters.png"), dpi=300)
    plt.close()


# ---------- Pipeline par groupe & pipeline global ----------

def process_group_csv(csv_path, out_root="plots", group_label=None, n_clusters=4,
                      T=180.0, bins=180, bandwidth=1.5):
    """
    Traite un CSV (un groupe) -> figures par jour + résumé groupe.
    """
    df_raw = _read_csv_smart(csv_path)
    df = _standardize_columns(df_raw, csv_path)

    group = group_label or _infer_group_from_path(csv_path)
    df["day"] = _infer_day_from_df_or_name(df, csv_path)  # (garantie)
    df, k = _ensure_clusters(df, n_clusters=n_clusters)

    out_dir = os.path.join(out_root, group)
    # Figures par jour (UMAP & PCA)
    plot_umap_and_pca_per_day(df, group, os.path.join(out_dir, "embed"))
    # Densités par jour
    plot_density_per_day(df, group, os.path.join(out_dir, "density"), T=T, bins=bins, bandwidth=bandwidth)
    # Barres d'évolution par groupe
    plot_cluster_evolution_bars(df, group, os.path.join(out_dir, "summary"))

    print(f"✅ {group}: figures enregistrées sous '{out_dir}/'")

def process_two_csvs(nms_csv_path, ctrl_csv_path, out_root="plots",
                     n_clusters=4, T=180.0, bins=180, bandwidth=1.5):
    """
    Appelle cette fonction avec tes 2 CSV (NMS, CTRL).
    Chaque groupe est traité de la même façon et les sorties vont dans :
      out_root/NMS/... et out_root/CTRL/...
    """
    process_group_csv(nms_csv_path, out_root=out_root, group_label="NMS",
                      n_clusters=n_clusters, T=T, bins=bins, bandwidth=bandwidth)
    process_group_csv(ctrl_csv_path, out_root=out_root, group_label="CTRL",
                      n_clusters=n_clusters, T=T, bins=bins, bandwidth=bandwidth)

# Exemple :
process_two_csvs(
    nms_csv_path="cut_off_NMS.csv",
    ctrl_csv_path="cut_off_CTRL.csv",
    out_root="plots/cut_off",   # <- dossier de sortie pour la méthode cut-off
    n_clusters=2,
    T=180.0,
    bins=180,
    bandwidth=1.5
)

process_two_csvs(
    nms_csv_path="unsup_NMS.csv",
    ctrl_csv_path="unsup_CTRL.csv",
    out_root="plots/unsup",     # <- dossier de sortie pour la méthode KMeans
    n_clusters=2,
    T=180.0,
    bins=180,
    bandwidth=1.5
)
