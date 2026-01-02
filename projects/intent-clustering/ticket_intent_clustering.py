from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD


# ----------------------------
# Config
# ----------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DEFAULT_INPUT = r"C:\Users\Codyw\PycharmProjects\PythonProject\simulated_service_tickets.xlsx"
SHEET_NAME = "tickets"  # sheet that doesn't include true_intent

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

STOPWORDS_MINI = set("""
a an and are as at be been but by can did do does for from get got had has have i im in into is it its
me my of on or our please so that the their there this to up we were what when where will with you your
""".split())


# ----------------------------
# Text cleaning
# ----------------------------
def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""

    s = str(s).lower()
    s = re.sub(r"\n+", " ", s)
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    toks = [t for t in s.split() if t not in STOPWORDS_MINI and len(t) > 2]
    return " ".join(toks)


# ----------------------------
# Embeddings
# ----------------------------
def build_embeddings(texts: list[str]) -> Tuple[Any, str]:
    """
    Returns (X, method_name)
    - Tries SBERT first; falls back to TF-IDF if not available
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        X = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return X, "sbert(all-MiniLM-L6-v2)"
    except Exception:
        vec = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=2
        )
        X = vec.fit_transform(texts)
        return X, "tfidf(1-2grams)"


# ----------------------------
# Clustering
# ----------------------------
def cluster_embeddings(X) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """
    Returns (labels, algo_name, info)
    - Tries HDBSCAN first; falls back to KMeans and chooses k via silhouette
    """
    # HDBSCAN 
    try:
        import hdbscan  # type: ignore

        X_dense = X.toarray() if hasattr(X, "toarray") else X
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=25,
            min_samples=10,
            metric="euclidean"
        )
        labels = clusterer.fit_predict(X_dense)
        info = {"noise_ratio": float(np.mean(labels == -1))}
        return labels, "hdbscan", info
    except Exception:
        pass

    # KMeans fallback
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    ks = list(range(4, 13))
    best_k, best_score, best_labels = None, -1.0, None

    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_dense)
        if len(set(labels)) > 1:
            score = silhouette_score(X_dense, labels)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels

    return best_labels, "kmeans", {"best_k": best_k, "silhouette": best_score}


# ----------------------------
# Interpret clusters (top terms + sample tickets)
# ----------------------------
def top_terms_by_cluster(df: pd.DataFrame, label_col: str, text_col: str, top_n: int = 10) -> pd.DataFrame:
    vec = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(df[text_col].tolist())
    vocab = np.array(vec.get_feature_names_out())

    labels = df[label_col].values
    clusters = sorted([c for c in np.unique(labels) if c != -1])

    rows = []
    for c in clusters:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        mean_tfidf = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:top_n]
        rows.append({
            "cluster": int(c),
            "n_tickets": int(len(idx)),
            "top_terms": ", ".join(vocab[top_idx].tolist())
        })

    return pd.DataFrame(rows).sort_values("n_tickets", ascending=False)


def sample_tickets(df: pd.DataFrame, label_col: str, n_per_cluster: int = 3) -> pd.DataFrame:
    out = []
    for c in sorted(df[label_col].unique()):
        if c == -1:
            continue
        sub = df[df[label_col] == c].head(n_per_cluster)
        for _, r in sub.iterrows():
            out.append({
                "cluster": int(c),
                "ticket_id": r["ticket_id"],
                "title": r.get("title", ""),
                "snippet": str(r.get("ticket_text", ""))[:220]
            })
    return pd.DataFrame(out)


# ----------------------------
# KPI table tying clusters to volume/rework
# ----------------------------
def cluster_kpis(df: pd.DataFrame, label_col: str = "cluster") -> pd.DataFrame:
    grp = df.groupby(label_col).agg(
        tickets=("ticket_id", "count"),
        rework=("rework_flag", "sum"),
        rework_rate=("rework_flag", "mean"),
        avg_resolution_hours=("resolution_time_hours", "mean"),
        top_channel=("channel", lambda s: s.value_counts().index[0]),
        top_team=("assigned_team", lambda s: s.value_counts().index[0]),
    ).reset_index()

    grp["tickets_pct"] = grp["tickets"] / grp["tickets"].sum()
    grp = grp.sort_values("tickets", ascending=False)
    return grp


# ----------------------------
# 2D visualization (SVD) + optional UMAP if installed
# ----------------------------
def plot_clusters_2d(X, labels: np.ndarray, outpath: Path):
    import matplotlib.pyplot as plt

    X_dense = X.toarray() if hasattr(X, "toarray") else X

    # Try UMAP if available (better visuals), else SVD
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED)
        X2 = reducer.fit_transform(X_dense)
        title = "Ticket clusters (UMAP 2D)"
    except Exception:
        svd = TruncatedSVD(n_components=2, random_state=RANDOM_SEED)
        X2 = svd.fit_transform(X_dense)
        title = "Ticket clusters (SVD 2D)"

    plt.figure(figsize=(8, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=12)
    plt.title(title)
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ----------------------------
# Main pipeline
# ----------------------------
def main(input_path: str = DEFAULT_INPUT):
    input_path = str(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    # Load
    df = pd.read_excel(input_path, sheet_name=SHEET_NAME)

    # Basic sanity
    needed = ["ticket_id", "ticket_text", "rework_flag"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'. Columns found: {df.columns.tolist()}")

    # Clean
    df["clean_text"] = df["ticket_text"].apply(clean_text)

    # Embeddings
    X, embed_method = build_embeddings(df["clean_text"].tolist())

    # Cluster
    labels, algo, info = cluster_embeddings(X)
    df["cluster"] = labels

    # KPIs + interpretation
    kpis = cluster_kpis(df, label_col="cluster")
    terms = top_terms_by_cluster(df[df["cluster"] != -1], label_col="cluster", text_col="clean_text", top_n=12)
    samples = sample_tickets(df, label_col="cluster", n_per_cluster=3)

    # Plot
    plot_path = OUT_DIR / "clusters_2d.png"
    plot_clusters_2d(X, labels, plot_path)

    # Export to Excel (so you can review like a stakeholder would)
    out_xlsx = OUT_DIR / "cluster_results.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="tickets_with_clusters", index=False)
        kpis.to_excel(writer, sheet_name="cluster_kpis", index=False)
        terms.to_excel(writer, sheet_name="cluster_top_terms", index=False)
        samples.to_excel(writer, sheet_name="cluster_samples", index=False)

    # Console summary
    print("\n--- Run Summary ---")
    print(f"Rows: {len(df)}")
    print(f"Embedding: {embed_method}")
    print(f"Clustering: {algo} | {info}")
    print(f"Outputs:\n  {out_xlsx}\n  {plot_path}\n")

    print("Top clusters by volume/rework:")
    print(kpis.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
