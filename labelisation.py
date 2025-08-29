import os, re
import pandas as pd
from pathlib import Path
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

    cut = df.copy()
    cut["cluster"] = (cut["Principal Frequency (kHz)"]>cutoff).astype(int).replace({0:2,1:1})

    X = StandardScaler().fit_transform(df[FEATS].fillna(0))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    unsup = df.copy(); unsup["cluster"] = km.labels_+1

    os.makedirs(outdir, exist_ok=True)
    for g in groups:
        cut[cut["Experimental Group"]==g].to_csv(f"{outdir}/cut_off_{g}.csv", index=False)
        unsup[unsup["Experimental Group"]==g].to_csv(f"{outdir}/unsup_{g}.csv", index=False)
    print(f"✅ fichiers écrits dans {outdir}")

# --- fusion S1/S2 (déjà présent) ---
def _parse_session_from_path(p: str) -> str:
    m = re.search(r"[\\/_]Session\s*(\d+)|\bS(\d+)\b", p, re.IGNORECASE)
    return f"S{(m.group(1) or m.group(2))}" if m else ""

def merge_excels(p1: str, p2: str, out_path: str|None=None) -> str:
    d1 = pd.read_excel(p1, dtype=str); d1["session"] = _parse_session_from_path(p1)
    d2 = pd.read_excel(p2, dtype=str); d2["session"] = _parse_session_from_path(p2)
    merged = pd.concat([d1, d2], ignore_index=True)
    if out_path is None:
        out_path = str(Path(p1).with_name("DAPP_CTRL&NMS_Sessions12_merged_Stats.xlsx"))
    merged.to_excel(out_path, index=False)
    return out_path

# --- fusion Hot/Cold/Room ---
def _parse_plate_from_path(p: str) -> str:
    s = p.lower()
    if "hot plate" in s or "hotplate" in s:   return "Hot Plate"
    if "cold plate" in s or "coldplate" in s: return "Cold Plate"
    if "room temp" in s or "roomtemp" in s:   return "Room Temp"
    return ""

def merge_excels_multi(paths: list[str], out_path: str|None=None) -> str:
    dfs = []
    for p in paths:
        d = pd.read_excel(p, dtype=str)
        d["plate"] = _parse_plate_from_path(p)  # Hot/Cold/Room (utile si tu veux comparer après)
        dfs.append(d)
    merged = pd.concat(dfs, ignore_index=True)
    if out_path is None:
        out_path = str(Path(paths[0]).with_name("HotColdRoom_CTRL&NMS_merged_AllPlates.xlsx"))
    merged.to_excel(out_path, index=False)
    return out_path

if __name__=="__main__":
    # --- cas FSS (inchangé)
    folder = "forced_swim_csv"
    csv = "exps/FSS/FSS_test02.xlsx"
    process(csv, folder)

    # --- cas DAPP : fusion S1 + S2 puis process
    folder = "double_aversion_csv"
    s1 = "exps/DAPP/DAPP_CTRL&NMS_Session1_merged_Stats.xlsx"
    s2 = "exps/DAPP/DAPP_CTRL&NMS_Session2_merged_Stats.xlsx"
    csv = merge_excels(s1, s2)
    process(csv, folder)

    # --- cas PLATE : fusion Hot + Cold + Room puis process
    folder = "plate_inputs_csv"
    cold = "exps/HotAndColdPlate/ColdPlate_CTRL&NMS_merged_Stats.xlsx"
    hot  = "exps/HotAndColdPlate/HotPlate_CTRL&NMS_merged_Stats.xlsx"
    room = "exps/HotAndColdPlate/RoomTemp_CTRL&NMS_merged_stats.xlsx"  # tel que fourni
    csv = merge_excels_multi([cold, hot, room])
    process(csv, folder)
