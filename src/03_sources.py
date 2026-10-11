"""Step 3: where does the channel send its readers? Links and @mentions, counted from the data.

Usage: python 03_sources.py DATA.csv [OUTPUT_FOLDER]
Every link is sorted by its domain into a type (DOMAIN_TYPES below). Domains not listed are "Other".
Network: the channel points to each type of destination. Line width = number of links.
Outputs: link_domains.csv, link_types.csv, link_types_by_year.csv, mentions.csv, tme_links.csv,
         fig_link_types.png, fig_sources_network.png, sources.gexf
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from common import MENTION_RE, URL_RE, load_messages

DOMAIN_TYPES = {
    "Own websites (MPV)": ["medicospelavida", "medicospelavidacovid19", "mpv"],
    "YouTube": ["youtube.com", "youtu.be"],
    "Alt video platforms": ["rumble.com", "odysee.com", "bitchute.com", "brighteon.com"],
    "Alt social (Gettr)": ["gettr.com", "getrr.com"],
    "Mainstream social": ["instagram.com", "twitter.com", "x.com", "t.co", "facebook.com", "fb.com", "tiktok.com"],
    "Telegram": ["t.me"],
    "Scientific sources": ["pubmed.ncbi.nlm.nih.gov", "clinicaltrials.gov", "sciencedirect.com", "nih.gov", "nejm.org",
                           "thelancet.com", "bmj.com", "jamanetwork.com", "medrxiv.org", "frontiersin.org", "cochrane.org", "scielo", "biomedcentral.com", "cell.com", "scientificamerican.com"],
    "Government and professional bodies": ["gov.br", "cfm.org.br", "portal.cfm.org.br", "camara.leg.br", "senado.leg.br", "jus.br", "who.int", "cdc.gov", "fda.gov"],
    "News media": ["revistaoeste.com", "nypost.com", "gazetadopovo.com.br", "folha.uol.com.br", "oglobo.globo.com", "g1.globo.com",
                   "telegraph.co.uk", "theepochtimes.com", "estadao.com.br", "cnnbrasil.com.br", "jovempan.com.br", "brasilsemmedo.com", "poder360.com.br", "nationalreview.com", "americaoutloud.com"],
    "Advocacy and health-freedom groups": ["brownstone.org", "childrenshealthdefense.org", "citizengo.org", "citizensforfreespeech.org", "chng.it"],
}


def domain_of(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url.lower()).split("/")[0].split("?")[0].rstrip(".,;:)")


def type_of(domain: str) -> str:
    for t, keys in DOMAIN_TYPES.items():
        for k in keys:
            if domain == k or domain.endswith("." + k) or (k in domain and "." not in k) or (k.startswith("medicospelavida") and k in domain):
                return t
    return "Other"


def main() -> None:
    data_path = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results"); out.mkdir(parents=True, exist_ok=True)
    d = load_messages(data_path)
    rows = [(r["message_id"], r["date"], r["year"], u, domain_of(u)) for _, r in d.iterrows() for u in URL_RE.findall(r["text"])]
    links = pd.DataFrame(rows, columns=["message_id", "date", "year", "url", "domain"])
    links["type"] = links["domain"].apply(type_of)
    print(f"Links: {len(links)} in {links['message_id'].nunique()} of {len(d)} messages. Distinct domains: {links['domain'].nunique()}")

    dom = links.groupby(["domain", "type"]).size().rename("links").reset_index().sort_values("links", ascending=False)
    dom.to_csv(out / "link_domains.csv", index=False)
    types = links["type"].value_counts().rename("links").to_frame(); types["share_pct"] = (100 * types["links"] / len(links)).round(1)
    types.to_csv(out / "link_types.csv"); print(types)
    by_year = pd.crosstab(links["year"], links["type"], normalize="index").mul(100).round(1)
    by_year.insert(0, "n_links", links.groupby("year").size()); by_year.to_csv(out / "link_types_by_year.csv")

    # Links to the group's own sites versus everywhere else
    own = links["type"].eq("Own websites (MPV)")
    print(f"Own websites: {own.mean() * 100:.0f}% of links. External: {100 - own.mean() * 100:.0f}%")
    print("Other domains (check these and add them to DOMAIN_TYPES if they fit a type):")
    print(dom[dom["type"] == "Other"].head(15).to_string(index=False))

    # Mentions (@) and Telegram links
    men = pd.Series([m.lower() for t in d["text"] for m in MENTION_RE.findall(URL_RE.sub(" ", t))]).value_counts()
    men.rename("mentions").to_frame().to_csv(out / "mentions.csv")
    tme = links[links["domain"] == "t.me"]["url"].str.extract(r"t\.me/(?:s/)?([^/?\s]+)")[0].value_counts()
    tme.rename("links").to_frame().to_csv(out / "tme_links.csv"); print("t.me targets:", dict(tme.head(10)))

    # Figure 1: bar chart of link types
    t = types.sort_values("links")
    fig, ax = plt.subplots(figsize=(8.5, 4.5)); ax.barh(t.index, t["links"], color="#2a6f97")
    for i, (v, s) in enumerate(zip(t["links"], t["share_pct"])): ax.text(v + 5, i, f"{v} ({s:.0f}%)", va="center", fontsize=9)
    ax.set_xlabel("Links in messages"); ax.set_title(f"Where the channel's links point (n = {len(links)} links)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False); ax.set_xlim(0, t["links"].max() * 1.18); fig.tight_layout()
    fig.savefig(out / "fig_link_types.png", dpi=200); plt.close(fig)

    # Figure 2: star network, channel -> destination types (and top domains inside each type)
    g = nx.DiGraph(); g.add_node("Channel", kind="channel")
    for ty, r in types.iterrows(): g.add_edge("Channel", ty, weight=int(r["links"]))
    nodes = [n for n in g.nodes if n != "Channel"]
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False) + np.pi / 2
    pos = {"Channel": (0, 0)}; pos.update({n: (np.cos(a), np.sin(a)) for n, a in zip(nodes, angles)})
    w_max = types["links"].max(); fig, ax = plt.subplots(figsize=(9, 8.5))
    for n in nodes:
        w = g["Channel"][n]["weight"]; ax.annotate("", xy=pos[n], xytext=pos["Channel"],
            arrowprops=dict(arrowstyle="-|>", lw=0.8 + 9 * w / w_max, color="#555", alpha=0.5, shrinkA=38, shrinkB=6))
    ax.scatter(*pos["Channel"], s=3200, color="#c1121f", zorder=3); ax.text(0, 0, "MPV\nchannel", color="white", ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
    for n in nodes:
        w = g["Channel"][n]["weight"]; ax.scatter(*pos[n], s=300 + 2200 * w / w_max, color="#e9c46a", edgecolor="#555", zorder=3)
        x, y = pos[n]; ha = "left" if x > 0.3 else "right" if x < -0.3 else "center"
        ax.text(x * 1.14, y * 1.14 + (0.06 if abs(x) <= 0.3 and y > 0 else -0.06 if abs(x) <= 0.3 else 0), f"{n}\n{w} links", ha=ha,
                va="bottom" if (abs(x) <= 0.3 and y > 0) else "top" if abs(x) <= 0.3 else "center", fontsize=9)
    ax.set_xlim(-2.3, 2.3); ax.set_ylim(-1.5, 1.5); ax.axis("off")
    ax.set_title("Where the channel sends its readers\nArrow width: number of links.", fontsize=11); fig.tight_layout()
    fig.savefig(out / "fig_sources_network.png", dpi=200); plt.close(fig); nx.write_gexf(g, out / "sources.gexf")


if __name__ == "__main__":
    main()
