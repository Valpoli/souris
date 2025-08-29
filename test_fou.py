# tickling_mat_clustering.py
import os, re, sys, csv, glob, warnings, joblib, h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import librosa, librosa.display
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# ---------- config courte ----------
DATASET_PATH = "#refs#/K"   # HDF5 dataset dans le .mat (4×N)
FREQS_IN_KHZ = True         # le .mat fournit les fréquences en kHz
FEATS = ["Score","Begin Time (s)","End Time (s)","Call Length (s)",
         "Principal Frequency (kHz)","Low Freq (kHz)","High Freq (kHz)",
         "Delta Freq (kHz)","Frequency Standard Deviation (kHz)",
         "Slope (kHz/s)","Sinuosity","Mean Power (dB/Hz)","Tonality","Peak Freq (kHz)"]

# ---------- utils I/O ----------
def read_table(path):
    ext = Path(path).suffix.lower()
    df = pd.read_excel(path, dtype=str) if ext in {".xlsx",".xls"} else \
         pd.read_csv(path, sep=None, engine="python", decimal=",", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

def to_float(s):
    return pd.to_numeric(pd.Series(s, dtype="string").str.replace(",",".",regex=False), errors="coerce")

def base_name(p):  # nom fichier .mat sans dossier
    return os.path.basename(str(p)).strip()

# ---------- 1) .mat -> DataFrame détections ----------
def mat_to_df(mat_path, dataset_path=DATASET_PATH, freqs_in_khz=FREQS_IN_KHZ):
    with h5py.File(mat_path, "r") as f:
        M = np.array(f[dataset_path])  # 4×N
    t0, f0, dt, df = M
    t1 = t0 + dt; f1 = f0 + df
    det = pd.DataFrame({
        "FileBase": base_name(mat_path),
        "ID": np.arange(1, t0.size+1, dtype=int),
        "Begin Time (s)": t0, "End Time (s)": t1,
        "Call Length (s)": dt,
        "Low Freq (kHz)": (f0 if freqs_in_khz else f0/1000.0),
        "High Freq (kHz)": (f1 if freqs_in_khz else f1/1000.0),
    })
    det["Delta Freq (kHz)"] = det["High Freq (kHz)"] - det["Low Freq (kHz)"]
    return det

# ---------- 2) jointure avec le tableau, on garde que les utiles ----------
def join_keep_useful(det_df, table_df):
    T = table_df.copy()
    T["FileBase"] = T["File"].map(lambda s: base_name(s))
    T["ID"] = pd.to_numeric(T["ID"], errors="coerce").astype("Int64")
    # remet au format numérique les features pertinentes
    for c in [c for c in FEATS if c in T.columns]:
        T[c] = to_float(T[c])
    keep = det_df.merge(T, on=["FileBase","ID"], how="inner", suffixes=("", "_tab"))
    # aligne les features manquantes depuis le .mat si pas dans le tableau
    for c in ["Begin Time (s)","End Time (s)","Call Length (s)","Low Freq (kHz)","High Freq (kHz)","Delta Freq (kHz)"]:
        if c in keep.columns and f"{c}_tab" in keep.columns:
            keep[c] = keep[f"{c}_tab"].combine_first(keep[c])
    # sélection finale de colonnes
    meta_cols = [c for c in ["Experimental Group","Day","Sex","File","FileBase","ID","Label","Accepted","phase"] if c in keep.columns]
    feat_cols = [c for c in FEATS if c in keep.columns]
    return keep[meta_cols + feat_cols].reset_index(drop=True)

# ---------- 3) features & clustering ----------
def build_X(df):
    use = [c for c in FEATS if c in df.columns]
    X = df[use].fillna(0.0).to_numpy(float)
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler, use

def kmeans_cluster(df, k=2, seed=42):
    X, scaler, used = build_X(df)
    km = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit(X)
    return km.labels_.astype(int), km, scaler, used

# ---------- 4) images par détection + plaquettes ----------
def audio_for_matpath(file_col_value):
    # tente .wav de même base
    p = str(file_col_value)
    maybe = p.replace(".mat",".wav")
    if os.path.isfile(maybe): return maybe
    b = base_name(p).replace(".mat",".wav")
    for up in [".","..","../.."]:
        cand = os.path.join(up, b)
        if os.path.isfile(cand): return cand
    return None

def render_detection_png(audio_path, row, out_png, pad_t=0.3, pad_f_khz=5.0, n_fft=4096, hop=512):
    if not audio_path: return False
    t0, t1 = float(row["Begin Time (s)"]), float(row["End Time (s)"])
    f0, f1 = float(row["Low Freq (kHz)"]), float(row["High Freq (kHz)"])
    offset = max(t0 - pad_t, 0.0); duration = (t1 - t0) + 2*pad_t
    y, sr = librosa.load(audio_path, sr=None, mono=True, offset=offset, duration=duration)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(4,3))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop, x_axis='time', y_axis='hz')
    ax = plt.gca()
    x0 = t0 - offset; w = t1 - t0
    y0 = f0*1000; h = (f1-f0)*1000; padf = pad_f_khz*1000
    ax.add_patch(plt.Rectangle((x0,y0), w,h, fill=False, color="red", lw=1.5))
    plt.xlim(max(0,x0-pad_t/2), x0+w+pad_t/2); plt.ylim(max(0,y0-padf), y0+h+padf)
    plt.axis("off"); os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(pad=0); plt.savefig(out_png, dpi=200); plt.close(); return True

def make_cluster_panels(df, out_dir, per_panel=25):
    os.makedirs(out_dir, exist_ok=True)
    audio_hint = df["File"].iloc[0] if "File" in df.columns else df["FileBase"].iloc[0]
    for k in sorted(df["cluster"].unique()):
        sub = df[df["cluster"]==k].copy()
        thumbs = []
        for _, r in sub.iterrows():
            a = audio_for_matpath(r.get("File", r.get("FileBase","")))
            out_png = os.path.join(out_dir, f"cl{k}_{base_name(r.get('FileBase', r.get('File','')))}_{int(r['ID'])}.png")
            ok = render_detection_png(a, r, out_png)
            if ok: thumbs.append(out_png)
        # panel
        if not thumbs: continue
        thumbs = thumbs[:per_panel]
        cols = int(np.ceil(np.sqrt(len(thumbs)))); rows = int(np.ceil(len(thumbs)/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.0))
        axes = np.array(axes).reshape(-1)
        for ax, img in zip(axes, thumbs):
            ax.imshow(plt.imread(img)); ax.set_title(Path(img).stem, fontsize=7)
            ax.axis("off")
        for ax in axes[len(thumbs):]: ax.axis("off")
        fig.suptitle(f"Cluster {k} — {len(sub)} detections", fontsize=10)
        plt.tight_layout(); fig.savefig(os.path.join(out_dir, f"panel_cluster_{k}.png"), dpi=200); plt.close(fig)

# ---------- 5) boucle d’édition ----------
def interactive_loop(df, km, scaler, used, out_root):
    df = df.copy()
    print("\n=== BOUCLE INTERACTIVE ===")
    print("cmd: show | move <rowidx> <newk> | moveid <FileBase> <ID> <newk> | retrain | save | quit")
    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            cmd = "quit"
        if cmd=="": continue
        if cmd=="show":
            print(df.groupby("cluster").size())
            make_cluster_panels(df, os.path.join(out_root,"panels"))
            print(f"→ panels mis à jour dans {out_root}/panels")
        elif cmd.startswith("moveid "):
            _, fb, sid, sk = cmd.split()
            sid, sk = int(sid), int(sk)
            m = (df["FileBase"]==fb) & (df["ID"]==sid)
            n = int(m.sum()); df.loc[m,"cluster"]=sk
            print(f"→ déplacé {n} détection(s) de {fb}#{sid} → cluster {sk}")
        elif cmd.startswith("move "):
            _, sidx, sk = cmd.split(); i=int(sidx); sk=int(sk)
            if 0<=i<len(df):
                df.loc[i,"cluster"]=sk; print(f"→ row {i} → cluster {sk}")
        elif cmd=="retrain":
            # supervision simple sur labels courants
            X = scaler.transform(df[used].fillna(0.0).to_numpy(float))
            y = df["cluster"].astype(int).to_numpy()
            clf = LogisticRegression(max_iter=1000, multi_class="auto").fit(X, y)
            joblib.dump({"scaler":scaler,"features":used,"clf":clf}, os.path.join(out_root,"model.joblib"))
            print("✅ modèle supervisé ré‐entraîné et sauvegardé → model.joblib")
        elif cmd=="save":
            # sauvegarde clustering courant (labels) + KMeans
            df.to_csv(os.path.join(out_root,"labels_curated.csv"), index=False)
            joblib.dump({"kmeans":km,"scaler":scaler,"features":used}, os.path.join(out_root,"kmeans.joblib"))
            print("✅ labels_curated.csv + kmeans.joblib sauvegardés")
        elif cmd in {"quit","exit"}:
            print("bye."); break
        else:
            print("commande inconnue.")

# ---------- MAIN ----------
def main(mat_dir, table_path, out_dir="outputs", k=2, seed=42):
    out_root = Path(out_dir); out_root.mkdir(parents=True, exist_ok=True)
    table = read_table(table_path)

    # .mat → détections → jointure utile
    all_det = []
    for mp in glob.glob(os.path.join(mat_dir, "*.mat")):
        det = mat_to_df(mp); all_det.append(det)
    dets = pd.concat(all_det, ignore_index=True) if all_det else pd.DataFrame(columns=["FileBase","ID"])
    df = join_keep_useful(dets, table)
    if df.empty:
        print("⚠️ Aucune détection utile trouvée (jointure vide)."); return

    # clustering non supervisé (k choisi par l’utilisateur)
    labels, km, scaler, used = kmeans_cluster(df, k=k, seed=seed)
    df["cluster"] = labels

    # panels init
    run_dir = out_root / "run"
    (run_dir/"panels").mkdir(parents=True, exist_ok=True)
    make_cluster_panels(df, run_dir/"panels")
    print(f"✅ Clustering k={k}. Panels init → {run_dir/'panels'}")
    print("👉 Astuce: 'show' pour regénérer les panels après modifications.")

    # boucle interactive: correction manuelle + ré‐entrainement + sauvegarde
    interactive_loop(df, km, scaler, used, str(run_dir))

if __name__ == "__main__":
    # Exemples d’appel :
    # python tickling_mat_clustering.py ./mats ./table.xlsx 4
    mat_dir   = sys.argv[1] if len(sys.argv)>1 else "mats"
    table_xls = sys.argv[2] if len(sys.argv)>2 else "FSS_test02.xlsx"
    k         = int(sys.argv[3]) if len(sys.argv)>3 else 2
    main(mat_dir, table_xls, out_dir="tickling_out", k=k, seed=42)
