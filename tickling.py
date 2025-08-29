# tickling_pipeline_k2.py
import os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

# --------- parsing depuis le nom de fichier ---------
META = re.compile(r"_(CTRL|NMS)_Day(\d+)", re.IGNORECASE)
def parse_meta(path:str):
    m = META.search(os.path.basename(path))
    group = m.group(1).upper() if m else ""
    day = f"Day{m.group(2)}" if m else "Day?"
    phase = Path(path).parent.name  # "2minutes Tickling" / "30 sec before Tickling"
    return group, day, phase

# --------- I/O + numériques ---------
FEATS = ["Score","Begin Time (s)","End Time (s)","Call Length (s)",
         "Principal Frequency (kHz)","Low Freq (kHz)","High Freq (kHz)","Delta Freq (kHz)",
         "Frequency Standard Deviation (kHz)","Slope (kHz/s)","Sinuosity",
         "Mean Power (dB/Hz)","Tonality","Peak Freq (kHz)"]

def read_any(p):
    ext = p.lower().split(".")[-1]
    df = pd.read_excel(p, dtype=str) if ext in ("xlsx","xls") else pd.read_csv(p, sep=None, engine="python", decimal=",", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

def to_float(s): return pd.to_numeric(pd.Series(s, dtype="string").str.replace(",", ".", regex=False), errors="coerce")

def prep_df(df, group, day, phase):
    d = df.copy()
    d["group"], d["day"], d["phase"] = group, day, phase
    for c in [c for c in FEATS if c in d.columns]: d[c] = to_float(d[c])
    if {"Begin Time (s)","End Time (s)"} <= set(d.columns):
        d["mid_s"] = 0.5*(d["Begin Time (s)"] + d["End Time (s)"])
    else:
        d["mid_s"] = np.nan
    return d

def build_X(d):
    use = [c for c in FEATS if c in d.columns]
    if not use: raise ValueError("Aucune feature numérique trouvée (au moins 'Score').")
    return StandardScaler().fit_transform(d[use].fillna(0.0).to_numpy(float))

# --------- clustering FORCÉ à 2 clusters + mapping stable ---------
def cluster_k2(d, seed=42):
    X = build_X(d)
    km = KMeans(n_clusters=2, n_init="auto", random_state=seed).fit(X)
    lab = km.labels_.astype(int)
    # mapping stable: cluster 1 = plus haute fréquence (ou score si freq absente)
    crit_col = "Principal Frequency (kHz)" if "Principal Frequency (kHz)" in d.columns else "Score"
    crit = d[crit_col].fillna(d[crit_col].median()).to_numpy(float)
    m0 = crit[lab==0].mean() if (lab==0).any() else -np.inf
    m1 = crit[lab==1].mean() if (lab==1).any() else -np.inf
    if m1 < m0:  # on veut 1 = high, 0 = low
        lab = 1 - lab
    d = d.copy(); d["cluster"] = lab
    return d

# --------- couleurs (inversées) ---------
def cluster_colors(K):
    cmap = plt.get_cmap("tab10" if K<=10 else "tab20")
    return [cmap((K-1-i) % cmap.N) for i in range(K)]

# --------- plots ---------
def save_pca_all_and_by_day(d, out_root, seed=42):
    X = build_X(d); Z = PCA(n_components=2, random_state=seed).fit_transform(X)
    K = int(d["cluster"].max())+1; cols = cluster_colors(K)
    # All
    plt.figure(figsize=(6,5))
    for k in range(K):
        m = (d["cluster"]==k).to_numpy()
        plt.scatter(Z[m,0], Z[m,1], s=10, color=cols[k], alpha=0.9, label=f"Cl {k}")
    plt.title("PCA — AllDays"); plt.legend(fontsize=8); plt.tight_layout()
    (out_root/"embed").mkdir(parents=True, exist_ok=True)
    plt.savefig(out_root/"embed"/"PCA_AllDays.png", dpi=300); plt.close()
    # By day (mêmes coords)
    for day in sorted(d["day"].unique()):
        m = (d["day"]==day).to_numpy()
        if not m.any(): continue
        plt.figure(figsize=(6,5))
        for k in range(K):
            mk = m & (d["cluster"]==k).to_numpy()
            plt.scatter(Z[mk,0], Z[mk,1], s=10, color=cols[k], alpha=0.9, label=f"Cl {k}")
        plt.title(f"PCA — {day}"); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(out_root/"embed"/f"PCA_{day}.png", dpi=300); plt.close()

def save_bars_max_left(d, out_png):
    pv = (d.assign(n=1).groupby(["day","cluster"])["n"].sum().unstack("cluster").fillna(0).astype(int))
    days = list(pv.index); clusters = list(pv.columns)
    K = len(clusters); cols = cluster_colors(K); cmap = {k: cols[i] for i,k in enumerate(clusters)}
    x = np.arange(len(days)); width = 0.8/max(1,K); y_max = pv.values.max() if pv.size else 0
    plt.figure(figsize=(max(6, len(days)*0.9), 4.5))
    for xi, day in enumerate(days):
        row = pv.loc[day]; order = row.sort_values(ascending=False).index
        for j,k in enumerate(order):
            plt.bar(xi + j*width, row[k], width=width, color=cmap[k], label=f"Cl {k}" if xi==0 else None)
    plt.xticks(x + (K-1)*width/2, days); plt.ylabel("Calls"); plt.title("Calls per cluster per day")
    plt.ylim(0, y_max*1.15 if y_max else 1); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

def save_densities(d, out_dir, T=180.0, bins=180, bw=1.5):
    if not d["mid_s"].notna().any(): return
    out_dir.mkdir(parents=True, exist_ok=True)
    K = int(d["cluster"].max())+1; cols = cluster_colors(K); grid = np.linspace(0,T,bins)
    def kde(ts):
        ts = np.clip(ts,0,T)
        if len(ts)>=2:
            kd = KernelDensity(kernel="gaussian", bandwidth=bw).fit(ts[:,None])
            y = np.exp(kd.score_samples(grid[:,None])); y *= len(ts)/(np.trapezoid(y,grid)+1e-12); return y
        return np.zeros_like(grid)
    # All
    ymax=0; curves=[]
    for k in range(K):
        t = d.loc[d["cluster"]==k, "mid_s"].dropna().to_numpy(float)
        y=kde(t); curves.append((k,y)); ymax=max(ymax,y.max() if y.size else 0)
    plt.figure(figsize=(8,4))
    for k,y in curves: plt.plot(grid,y,color=cols[k],label=f"Cl {k}")
    plt.title("Density — AllDays"); plt.ylim(0, ymax*1.05 if ymax else 1)
    plt.xlabel("Time (s)"); plt.ylabel("≈ calls/s"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(out_dir/"density_AllDays.png", dpi=300); plt.close()
    # Per day
    for day in sorted(d["day"].unique()):
        sub = d[d["day"]==day]; ymax=0; curves=[]
        for k in range(K):
            t = sub.loc[sub["cluster"]==k,"mid_s"].dropna().to_numpy(float)
            y=kde(t); curves.append((k,y)); ymax=max(ymax,y.max() if y.size else 0)
        plt.figure(figsize=(8,4))
        for k,y in curves: plt.plot(grid,y,color=cols[k],label=f"Cl {k}")
        plt.title(f"Density — {day}"); plt.ylim(0, ymax*1.05 if ymax else 1)
        plt.xlabel("Time (s)"); plt.ylabel("≈ calls/s"); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(out_dir/f"density_{day}.png", dpi=300); plt.close()

# --------- exécution sur un dossier (phase) ---------
def run_phase(phase_dir: str, seed=42):
    files = sorted([str(p) for p in Path(phase_dir).glob("*.xlsx")] + [str(p) for p in Path(phase_dir).glob("*.csv")])
    if not files:
        print(f"⚠️ Aucun fichier dans {phase_dir}"); return
    dfs=[]
    for f in files:
        group, day, phase = parse_meta(f)
        df = prep_df(read_any(f), group, day, phase)
        dfs.append(df)
    d = pd.concat(dfs, ignore_index=True)
    d = cluster_k2(d, seed=seed)  # <-- K=2 garanti + mapping stable

    out_root = Path("plots")/"Tickling"/Path(phase_dir).name
    (out_root/"embed").mkdir(parents=True, exist_ok=True)
    (out_root/"density").mkdir(parents=True, exist_ok=True)

    save_pca_all_and_by_day(d, out_root, seed=seed)
    save_bars_max_left(d, out_root/"bars_calls_per_day.png")
    save_densities(d, out_root/"density")

    print(f"✅ Plots écrits dans: {out_root}")

# --------- main ---------

if __name__ == "__main__":
    run_phase("exps/Tickling/2minutes Tickling", seed=42)
    run_phase("exps/Tickling/30 sec before Tickling", seed=42)
