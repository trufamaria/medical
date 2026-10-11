"""Shared helpers: load the Telegram export, clean Portuguese text, pull links and mentions."""
import re
import unicodedata
from urllib.parse import unquote

import pandas as pd

URL_RE = re.compile(r"https?://[^\s<>\"')]+|www\.[^\s<>\"')]+")
MENTION_RE = re.compile(r"(?<![\w.])@[A-Za-z]\w{3,}")
RENAME = {"message.id": "message_id", "message.date": "date", "message.text": "text",
          "message.views": "views", "message.peer_id.channel_id": "channel_id",
          "message.sender_id": "sender_id", "message.post_author": "post_author"}

STOPWORDS = set("""
a ao aos aquela aquele aquilo as ate com como da das de dela dele depois do dos e ela ele em entre era essa esse esta este eu foi foram ha isso isto ja lhe mais mas me mesmo meu minha muito na nas nem no nos nossa nosso num numa nao o os ou para pela pelas pelo pelos por qual quando que quem se sem ser seu sua so tambem te tem tinha um uma voce vai sao ter esta estao sera pode podem deve fazer faz sendo contra cada todo toda todos todas outro outra outros outras ano anos dia dias hoje agora aqui bem sim novo nova segundo primeira primeiro nesta neste nessa nesse pois porque entao assim ainda sempre nunca apenas alem antes durante apos desde sobre tanto tao quanto
https http www com br org gov html php youtube youtu instagram twitter telegram whatsapp gettr rumble odysee
medicospelavida medicospelavidacovid opiniao noticias noticia artigo video videos live canal link acesse clique confira veja assista inscreva compartilhe
""".split())


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def slug_words(url: str) -> str:
    """Words from the last part of an article link (many posts are only a link)."""
    path = re.sub(r"^https?://[^/]+", "", url)
    path = re.sub(r"[?#].*$", "", unquote(path)).strip("/")
    if not path:
        return ""
    last = path.split("/")[-1]
    if re.fullmatch(r"[\d.]+|[A-Za-z0-9_-]{9,12}", last) and "-" not in last:
        return ""  # ids (youtube, pubmed) carry no words
    return last.replace("-", " ").replace("_", " ")


def clean_for_topics(text: str) -> str:
    slugs = " ".join(slug_words(u) for u in URL_RE.findall(text))
    text = URL_RE.sub(" ", text) + " " + slugs
    text = MENTION_RE.sub(" ", text)
    text = normalize(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(t for t in text.split() if len(t) > 2 and t not in STOPWORDS)


def load_messages(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=RENAME)
    df["text"] = df["text"].fillna("").astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["channel_id", "message_id"])
    df = df[df["text"].str.strip() != ""].copy()
    df["year"] = df["date"].dt.year
    df["text_clean"] = df["text"].apply(clean_for_topics)
    df["n_tokens"] = df["text_clean"].str.split().str.len()
    return df.sort_values("date").reset_index(drop=True)
