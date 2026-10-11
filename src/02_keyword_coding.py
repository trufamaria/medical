"""Step 2: transparent keyword coding and a theme co-occurrence network.

Usage: python 02_keyword_coding.py DATA.csv [OUTPUT_FOLDER]
Each theme is a list of word patterns (CODEBOOK below). A message gets a theme if any pattern
appears in its text or in the words of its links. A message can have several themes.
Two themes are linked when they appear in the same message. Edge weight = number of messages.
Outputs: keyword_theme_counts.csv, keyword_theme_by_year_pct.csv, cooccurrence_edges.csv,
         entity_mentions.csv, fig_cooccurrence_network.png, cooccurrence.gexf
"""
import itertools
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from common import URL_RE, load_messages, normalize, slug_words

CODEBOOK = {
    "COVID vaccines": r"vacin|mrna|pfizer|moderna|astrazeneca|spike|imuniz",
    "Early treatment (HCQ, ivermectin)": r"hidroxicloroquina|cloroquina|ivermectina|tratamento precoce|nitazoxanida|kit covid",
    "Masks and lockdown": r"mascara|lockdown|isolamento|distanciamento|passaporte",
    "Children": r"crianca|infantil|pediatr|adolescente|escola|bebe",
    "Health authorities": r"anvisa|\bcfm\b|ministerio da saude|\boms\b|\bwho\b|\bcdc\b|\bfda\b|conselho federal|sus\b|\bpni\b",
    "Censorship and free speech": r"censura|strike|liberdade de|bloque|cancel|derrubad|removid|banid|suspens",
    "Science and studies": r"estudo|pubmed|meta.?analise|ensaio clinico|pesquisa|artigo cientifico|evidencia|clinicaltrials",
    "Law and politics": r"audiencia publica|camara|senado|assembleia|deputad|vereador|governador|\bstf\b|projeto de lei|\bcpi\b|tribunal|justica|prefeitura",
    "Lives, congress, events": r"\blive\b|congresso|comunica|palestra|webinar|encontro|simposio",
    "Donations and membership": r"\bpix\b|\bdoe\b|doacao|associad|contribu|apoie|engaje|vendas",
    "Integrative and anthroposophic medicine": r"antroposof|homeopat|integrativ|viscum|fitoterap",
    "Life and family values": r"aborto|ventre|valor da vida|vida humana|familia|eutanasia|gravidez|gestante",
    "Global elites and the WHO treaty": r"\bgates\b|great reset|reset|tratado|soberania|globalis|agenda 2030|\bforum economico|\bwef\b",
    "Rio Grande do Sul floods": r"rio grande do sul|enchente|\bsos\b|calamidade",
}
ENTITIES = {  # people and groups named in the earlier hand-made figures, counted here to test them
    "Peter McCullough": r"mccullough", "Vladimir Zelenko": r"zelenko", "FLCCC": r"flccc", "Pierre Kory": r"\bkory\b",
    "Robert Malone": r"\bmalone\b", "Reiner Fuellmich": r"fuellmich", "Robert F. Kennedy Jr.": r"\brfk\b|kennedy",
    "José Augusto Nasser": r"nasser", "Bill Gates": r"\bgates\b", "Bolsonaro": r"bolsonaro", "Lula": r"\blula\b",
    "Pfizer": r"pfizer", "Moderna": r"moderna", "Anvisa": r"anvisa", "CFM": r"\bcfm\b", "OMS / WHO": r"\boms\b|\bwho\b",
}


def code(texts: pd.Series, book: dict) -> pd.DataFrame:
    return pd.DataFrame({k: texts.str.contains(p, regex=True) for k, p in book.items()})


def draw(g: nx.Graph, path: Path) -> None:
    pos = nx.spring_layout(g, weight="weight", seed=3, k=2.2, iterations=400)
    n_max = max(d["messages"] for _, d in g.nodes(data=True)); w_max = max(d["weight"] for *_, d in g.edges(data=True))
    fig, ax = plt.subplots(figsize=(11, 8.5))
    for u, v, d in g.edges(data=True):
        ax.plot(*zip(pos[u], pos[v]), color="#2a6f97", alpha=0.18 + 0.5 * d["weight"] / w_max, lw=0.5 + 9 * d["weight"] / w_max, zorder=1,
                solid_capstyle="round")
    for n, d in g.nodes(data=True):
        ax.scatter(*pos[n], s=250 + 2600 * d["messages"] / n_max, color="#e9c46a", edgecolor="#555", linewidth=0.8, zorder=2)
        r = ((250 + 2600 * d["messages"] / n_max) / np.pi) ** 0.5 + 5  # label sits just below the circle
        ax.annotate(f"{n} ({d['messages']})", pos[n], xytext=(0, -r), textcoords="offset points", ha="center", va="top", fontsize=8.5,
                    zorder=4, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_title("Themes that appear in the same message\n"
                 "Circle size: messages with the theme (in brackets). Line width: messages with both themes.", fontsize=11)
    ax.axis("off"); ax.margins(0.15); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main() -> None:
    data_path = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results"); out.mkdir(parents=True, exist_ok=True)
    d = load_messages(data_path)
    d["search"] = d["text"].apply(lambda t: normalize(URL_RE.sub(" ", t) + " " + " ".join(slug_words(u) for u in URL_RE.findall(t))))
    C = code(d["search"], CODEBOOK)
    print(f"Messages: {len(d)}. With at least one theme: {int(C.any(axis=1).sum())} ({100 * C.any(axis=1).mean():.0f}%)")

    counts = pd.DataFrame({"messages": C.sum(), "share_pct": (100 * C.mean()).round(1)}).sort_values("messages", ascending=False)
    counts.to_csv(out / "keyword_theme_counts.csv"); print(counts)
    by_year = C.groupby(d["year"]).mean().mul(100).round(1); by_year.insert(0, "n_messages", d.groupby("year").size())
    by_year.to_csv(out / "keyword_theme_by_year_pct.csv")

    g = nx.Graph()
    for k in C.columns: g.add_node(k, messages=int(C[k].sum()))
    rows = []
    for a, b in itertools.combinations(C.columns, 2):
        w = int((C[a] & C[b]).sum())
        if w > 0: rows.append({"theme_a": a, "theme_b": b, "messages_with_both": w})
    edges = pd.DataFrame(rows).sort_values("messages_with_both", ascending=False)
    edges.to_csv(out / "cooccurrence_edges.csv", index=False)
    keep = edges[edges["messages_with_both"] >= 25]  # figure shows pairs with at least 25 messages
    for _, r in keep.iterrows(): g.add_edge(r["theme_a"], r["theme_b"], weight=int(r["messages_with_both"]))
    g.remove_nodes_from([n for n in list(g.nodes) if g.degree(n) == 0])
    nx.write_gexf(g, out / "cooccurrence.gexf")
    print(f"Network: {g.number_of_nodes()} themes, {g.number_of_edges()} links (pairs with 25+ messages)")
    print(keep.head(12).to_string(index=False)); draw(g, out / "fig_cooccurrence_network.png")

    E = code(d["search"], ENTITIES)
    pd.DataFrame({"messages_mentioning": E.sum()}).sort_values("messages_mentioning", ascending=False).to_csv(out / "entity_mentions.csv")
    print(E.sum().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
