#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
from pathlib import Path

def read_table(path: Path) -> pd.DataFrame:
    # Détecte automatiquement CSV vs Excel
    suf = path.suffix.lower()
    if suf in {".csv", ".tsv"}:
        # essaie virgule puis point-virgule
        try:
            df = pd.read_csv(path, sep=",", encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    elif suf in {".xlsx", ".xls"}:
        # nécessite openpyxl pour .xlsx
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Extension non supportée: {suf}")
    return df

def main():
    if len(sys.argv) < 2:
        print("Usage: python count_clusters.py <chemin/vers/fichier.csv|xlsx>")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    if not in_path.exists():
        print(f"Fichier introuvable: {in_path}")
        sys.exit(1)

    df = read_table(in_path)

    # Normalise le nom de colonne (au cas où la casse varie)
    cols = {c.lower(): c for c in df.columns}
    if "cluster" not in cols:
        raise KeyError(
            "Colonne 'cluster' introuvable. Colonnes disponibles: "
            + ", ".join(df.columns.astype(str))
        )
    cluster_col = cols["cluster"]

    # Nettoyage léger : convertir en entier si possible
    cluster = pd.to_numeric(df[cluster_col], errors="coerce").astype("Int64")

    counts = cluster.value_counts(dropna=False).sort_index()
    # Prépare un DataFrame propre pour export
    out = counts.rename_axis("cluster").reset_index(name="count")

    # Affichage console
    print("\nNombre d'éléments par cluster :")
    for cl, n in out.itertuples(index=False):
        label = "NaN" if pd.isna(cl) else cl
        print(f"  cluster {label}: {n}")

    # Sauvegarde
    out_path = in_path.with_name("cluster_counts.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nRésultat sauvegardé dans: {out_path}")

if __name__ == "__main__":
    main()
