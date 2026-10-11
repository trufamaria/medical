# Themes and Sources in a Brazilian Doctors' Telegram Channel

Python code that finds themes in the posts of a public Telegram channel, shows which themes appear together, and maps where the channel sends its readers.

I used it on the channel of Médicos Pela Vida (MPV), a Brazilian group of doctors that opposed COVID-19 measures. The data has 1,221 messages (1,123 with text) from October 2021 to January 2026. The code works with any Telegram export in the same format.

**Status:** work in progress. Themes come from a topic model and a keyword list. I read example messages to name them, and the names are mine.

## What the code does

| Step | Script | Output |
|------|--------|--------|
| 1. Themes | `src/01_themes.py` | Topic model (TF-IDF and NMF): 10 themes with top words, example messages, share by year |
| 2. Theme network | `src/02_keyword_coding.py` | Keyword coding with a visible codebook, themes that appear in the same message, mentions of named people and groups |
| 3. Sources | `src/03_sources.py` | Every link sorted by domain type, link types by year, `@` mentions, `t.me` links |

## Data format

One CSV with these columns (the names Telegram exports use): `message.id`, `message.date`, `message.text`, `message.views`, `message.peer_id.channel_id`, `message.sender_id`. Spaces around header names are removed.

The raw data and the message-level results are not in this repository. They hold the full text of posts. The `.gitignore` file keeps them out.

## How to run

```
pip install -r requirements.txt
bash run_all.sh path/to/export.csv
```

Run time is under one minute. Results go in `results/`.

## Method

**Links as text.** About half of the posts are only a link to an article on the group's website. The words in the link ("covid-vacina-criancas-obrigatoriedade") are added to the text, so these posts can be placed in a theme.

**Topic model.** Links, mentions, accents, numbers, and Portuguese stop words are removed. Words are weighted with TF-IDF (single words and pairs). A non-negative matrix factorization (NMF) model finds 10 sets of words that occur together. Each message goes to its strongest theme. Coherence (NPMI) was 0.51 at 6 themes, 0.49 at 10, and 0.39 at 12. I used 10 because 6 joined distinct themes into two large groups. Run `python src/01_themes.py DATA results auto` to see the table. Messages with fewer than 4 words after cleaning are left out (1,039 of 1,123 are used).

**Keyword coding.** I wrote 14 themes as lists of word patterns (the `CODEBOOK` in `02_keyword_coding.py`). A message gets a theme if a pattern appears in its text or link words. A message can have several themes. Two themes are linked when they appear in the same message. The figure shows pairs with at least 25 messages.

**Sources.** Each link goes to a domain. Each domain goes to a type (`DOMAIN_TYPES` in `03_sources.py`). Domains not listed count as "Other". The script prints the largest "Other" domains so you can add them.

## Main results

- **The channel mostly promotes itself.** 59% of the 1,273 links go to the group's own websites. YouTube is next (9%), then scientific sources (6%), alternative video platforms (6%), Gettr (5%), and mainstream social media (5%).
- **Programming is the largest part of the channel.** Two weekly lives ("Comunica MPV" on Tuesdays, a series with an anthroposophic medicine alliance on Wednesdays) make up about 20% of messages in the topic model. Donation and membership calls often sit in the same posts (213 messages with both themes).
- **COVID vaccines are the largest content theme.** 27% of messages mention vaccines. Vaccines connect most often to children (91 messages), studies (62), and health authorities (59).
- **The mix changes over time.** Links to scientific sources fall from 18-21% of links in 2021 and 2022 to 0% in 2024 and 2025. Gettr appears only in 2022 and 2023. In 2025, YouTube is 29% of links and mentions of children reach 23% of messages.
- **Few named people appear.** Peter McCullough appears in 7 messages, Pierre Kory in 7, Bill Gates in 6, and Robert F. Kennedy Jr. in 9 (`results/entity_mentions.csv`). Only 8 links go to Telegram.

## Limits

- This is one channel. The results show what this channel posted, not what its audience read or believed.
- Keyword coding finds words, not meaning. A message that mentions "vacina" is not always about vaccines. Read examples before you cite a count.
- The pattern for "Lives, congress, events" matches announcements and link words, so it is broad (39% of messages).
- The topic model has one mixed theme (theme 4: lives, congress, editorials). Read its examples in `results/themes.csv`.
- Photos, videos, and posts without text are not counted. Views are in the data but not analyzed here.
- Domain types are my own groups. Check `results/link_domains.csv` before you cite a number.

## Results files

- `results/themes.csv`, `themes_by_year_pct.csv`, `theme_names.csv`, `theme_model_selection.csv` (made by the `auto` run)
- `results/keyword_theme_counts.csv`, `keyword_theme_by_year_pct.csv`, `cooccurrence_edges.csv`, `entity_mentions.csv`
- `results/link_domains.csv`, `link_types.csv`, `link_types_by_year.csv`, `mentions.csv`, `tme_links.csv`
- `results/cooccurrence.gexf`, `sources.gexf`: networks for Gephi
- `results/fig_*.png`: figures

## Author

Marina Petric, Ph.D., Media and Communication, Texas Tech University
[marinapetric.com](https://marinapetric.com)
