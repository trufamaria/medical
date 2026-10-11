"""Step 1: candidate themes with a topic model (TF-IDF + NMF).

Usage: python 01_themes.py DATA.csv [OUTPUT_FOLDER] [K]     (K = number of themes, default 10, or "auto")
Many posts are only a link to an article. The words in the article's link are used as text.
Name the themes by hand: read themes.csv, then write results/theme_names.csv (columns: theme,name).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from common import load_messages

MIN_TOKENS = 4
TOP_N = 10


def npmi(top_idx, B):
    n = B.shape[0]; df = np.asarray(B.sum(axis=0)).ravel(); sc = []
    for w in top_idx:
        co = (B[:, w].T @ B[:, w]).toarray(); ps = []
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                pij = co[i, j] / n
                ps.append(-1.0 if pij == 0 else np.log(pij / (df[w[i]] / n * df[w[j]] / n)) / -np.log(pij))
        sc.append(np.mean(ps))
    return float(np.mean(sc))


def main():
    data_path = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results"); out.mkdir(parents=True, exist_ok=True)
    k_arg = sys.argv[3] if len(sys.argv) > 3 else "10"
    d = load_messages(data_path)
    docs = d[d["n_tokens"] >= MIN_TOKENS].copy().reset_index(drop=True)
    print(f"Messages with text: {len(d)}. Long enough for themes: {len(docs)}.")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.4, sublinear_tf=True)
    X = vec.fit_transform(docs["text_clean"]); terms = np.array(vec.get_feature_names_out())
    fit = lambda k: NMF(n_components=k, init="nndsvd", random_state=0, max_iter=400).fit(X)
    if k_arg == "auto":
        B = CountVectorizer(vocabulary=vec.vocabulary_, binary=True, ngram_range=(1, 2)).fit_transform(docs["text_clean"])
        rows = [{"k": k, "coherence_npmi": round(npmi([np.argsort(c)[::-1][:10] for c in fit(k).components_], B), 4)}
                for k in (6, 8, 10, 12, 14)]
        sel = pd.DataFrame(rows); sel.to_csv(out / "theme_model_selection.csv", index=False); print(sel)
        k = int(sel.sort_values("coherence_npmi", ascending=False).iloc[0]["k"])
    else:
        k = int(k_arg)
    m = fit(k); W = m.transform(X)
    docs["theme"] = W.argmax(axis=1).astype(str); docs["theme_strength"] = W.max(axis=1)
    names_path = out / "theme_names.csv"
    names = dict(pd.read_csv(names_path, dtype=str).values) if names_path.exists() else {}
    rows = []
    for t in range(k):
        mem = docs[docs["theme"] == str(t)].sort_values("theme_strength", ascending=False)
        rows.append({"theme": t, "name": names.get(str(t), ""), "top_words": ", ".join(terms[np.argsort(m.components_[t])[::-1][:TOP_N]]),
                     "n_messages": len(mem), "share_pct": round(100 * len(mem) / len(docs), 1),
                     "example_messages": " || ".join(s.replace("\n", " ")[:200] for s in mem["text"].head(4))})
    th = pd.DataFrame(rows); th.to_csv(out / "themes.csv", index=False, encoding="utf-8-sig")
    docs["year"] = docs["date"].dt.year
    by_year = pd.crosstab(docs["year"], docs["theme"], normalize="index").mul(100).round(1)
    by_year.columns = [names.get(c, c) for c in by_year.columns]
    by_year.to_csv(out / "themes_by_year_pct.csv")
    docs[["message_id", "date", "views", "theme", "theme_strength", "text"]].to_csv(out / "message_themes.csv", index=False, encoding="utf-8-sig")

    order = th.sort_values("n_messages")
    fig, ax = plt.subplots(figsize=(9, 0.5 * k + 1.3))
    labels = [n if n else f"Theme {t}" for t, n in zip(order["theme"], order["name"])]
    ax.barh(labels, order["share_pct"], color="#2a6f97")
    for i, v in enumerate(order["share_pct"]): ax.text(v + 0.3, i, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_xlabel("Share of messages (%)"); ax.set_title(f"Themes in the channel (topic model, n = {len(docs)} messages)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(out / "fig_themes.png", dpi=200)
    print(th[["theme", "name", "n_messages", "share_pct", "top_words"]].to_string(index=False))


if __name__ == "__main__":
    main()
