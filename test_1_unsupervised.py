import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap.umap_ as umap

# -----------------------------
# 1. Charger les données Excel
# -----------------------------
df = pd.read_excel("data.xlsx")

# -----------------------------
# 2. Nettoyage et sélection
# -----------------------------
# Colonnes à ignorer pour le clustering
colonnes_inutiles = ["File", "ID", "Accepted", "Label"]

labels_originaux = df["Label"] if "Label" in df.columns else None
print(labels_originaux)

df_num = df.drop(columns=colonnes_inutiles, errors="ignore")

# Conversion virgules -> points puis float
df_num = df_num.replace(",", ".", regex=True)
df_num = df_num.astype(float)

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)

# -----------------------------
# 3. Clustering KMeans (2 à 7 clusters)
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

for k in range(2, 8):
    # Création dossier
    dossier = f"clusters_k{k}"
    os.makedirs(dossier, exist_ok=True)

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Ajout au DataFrame pour plots
    df_plot = df_num.copy()
    df_plot["Cluster"] = clusters

    # --- PCA plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="tab10", s=50)
    plt.title(f"Clustering KMeans avec k={k} (projection PCA)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "pca_clusters.png"))
    plt.close()

    # --- Paires de variables
    variables = df_num.columns
    for var_x, var_y in itertools.combinations(variables, 2):
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=df_plot, x=var_x, y=var_y, hue="Cluster", palette="tab10", s=40)
        plt.title(f"{var_x} vs {var_y} (k={k})")
        plt.tight_layout()

        # Nettoyer les noms pour éviter caractères interdits
        safe_var_x = (var_x.replace(" ", "_")
                             .replace("/", "_")
                             .replace("(", "")
                             .replace(")", "")
                             .replace("[", "")
                             .replace("]", ""))
        safe_var_y = (var_y.replace(" ", "_")
                             .replace("/", "_")
                             .replace("(", "")
                             .replace(")", "")
                             .replace("[", "")
                             .replace("]", ""))

        plt.savefig(os.path.join(dossier, f"{safe_var_x}_vs_{safe_var_y}.png"))
        plt.close()

    # --- UMAP
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=clusters, palette="tab10", s=50)
    plt.title(f"Projection UMAP des données (k={k})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(os.path.join(dossier, "umap_clusters.png"))
    plt.close()

    # -----------------------------
    # 4. Comparaison Labels vs Clusters (si k=2 et si label dispo)
    # -----------------------------
    if k == 2 and labels_originaux is not None:
        comparaison = pd.crosstab(labels_originaux, clusters)
        print("\n📊 Comparaison clusters (k=2) vs Labels originaux :")
        print(comparaison)

        # Heatmap visuelle
        plt.figure(figsize=(6, 4))
        sns.heatmap(comparaison, annot=True, fmt="d", cmap="Blues")
        plt.title("Comparaison clusters (k=2) vs Labels originaux")
        plt.ylabel("Label original")
        plt.xlabel("Cluster KMeans")
        plt.tight_layout()
        plt.savefig(os.path.join(dossier, "comparaison_labels_vs_clusters.png"))
        plt.close()

print("✅ Graphiques générés : un dossier par cluster (2→7) avec PCA, toutes les combinaisons de variables, et UMAP.")
