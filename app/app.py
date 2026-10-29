import os,html, json
import numpy as np, pandas as pd, gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from engines import LANGS, RES, get_engine, search_all

NAMES = {"ar": "Arabic", "bn": "Bengali", "en": "English", "es": "Spanish", "fa": "Persian",
         "fi": "Finnish", "fr": "French", "hi": "Hindi", "id": "Indonesian", "ja": "Japanese",
         "ko": "Korean", "ru": "Russian", "sw": "Swahili", "te": "Telugu", "th": "Thai", "zh": "Chinese"}
LANG_CHOICES = [f"{l} - {NAMES[l]}" for l in LANGS]
MODELS = ["TF-IDF", "BM25", "Sentence Embedding", "Hybrid (RRF)"]
ALL = "Compare all four"
OWN = "(type your own)"

COMP = pd.read_csv(f"{RES}/comparison_table.csv").round(3)
MACRO = pd.read_csv(f"{RES}/macro_average.csv").round(3)

def code(choice):
    return choice.split(" - ")[0].strip()

def card(rank, docid, title, text, score, rel):
    ok = rel is not None and docid in rel
    border = "#2e9e5b" if ok else "rgba(128,128,128,.35)"
    bg = "rgba(46,158,91,.13)" if ok else "rgba(128,128,128,.07)"
    badge = (' <span style="background:#2e9e5b;color:#fff;border-radius:6px;padding:0 6px;'
             'font-size:.75em">✔ relevant (qrels)</span>') if ok else ""
    snip = html.escape(text[:320]) + ("…" if len(text) > 320 else "")
    return (f'<div style="border:1px solid {border};background:{bg};border-radius:10px;'
            f'padding:10px 12px;margin-bottom:8px">'
            f'<div style="display:flex;justify-content:space-between;font-size:.8em;opacity:.75">'
            f'<span>#{rank} · {html.escape(docid)}</span><span>score {score:.3f}</span></div>'
            f'<div style="font-weight:600;margin:3px 0">{html.escape(title)}{badge}</div>'
            f'<div style="font-size:.9em;line-height:1.45">{snip}</div></div>')

def run_search(lang_choice, query, model, k):
    query = (query or "").strip()
    if not query:
        return "<p>Please type a query or pick a sample query.</p>"
    eng = get_engine(code(lang_choice))
    qid = eng.q2qid.get(query)
    rel = eng.rel.get(qid) if qid else None
    note = ""
    if rel is not None:
        note = (f'<p style="font-size:.9em">This is a MIRACL dev query with {len(rel)} known relevant '
                f'passage(s). Green = judged relevant; unmarked = not judged relevant.</p>')
    cols = []
    for m in (MODELS if model == ALL else [model]):
        hits = eng.search(query, m, int(k))
        cards, n_rel = "", 0
        for r, (i, s) in enumerate(hits, 1):
            docid, title, text = eng.passage(i)
            n_rel += rel is not None and docid in rel
            cards += card(r, docid, title, text, s, rel)
        summary = (f' <span style="font-weight:400;font-size:.85em">({n_rel}/{len(hits)} relevant)</span>'
                   if rel is not None else "")
        cards = cards or "<i>No results (no searchable terms after preprocessing).</i>"
        cols.append(f'<div><div style="font-weight:700;margin-bottom:6px">{m}{summary}</div>{cards}</div>')
    grid = ('<div style="display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))">'
            + "".join(cols) + "</div>")
    return note + grid

def run_cross(query, k):
    query = (query or "").strip()
    if not query:
        return "<p>Type a query in any language.</p>"
    hits = search_all(query, int(k))
    return "".join(card(r, f"[{NAMES[l]}] {d}", t, x, sc, None)
                   for r, (l, d, t, x, sc) in enumerate(hits, 1))

def on_language(choice):
    eng = get_engine(code(choice))
    samples = [OWN] + eng.topics["query"].head(40).tolist()
    return gr.Dropdown(choices=samples, value=OWN), ""

def on_sample(s):
    return "" if s == OWN else s

def table_for(l):
    return COMP if l == "All languages" else COMP[COMP["lang"] == l]

def pr_plot(choice):
    langs = LANGS if choice.startswith("Average") else [code(choice)]
    curves = {}
    for l in langs:
        for part in ("sparse", "dense", "hybrid"):
            for name, v in json.load(open(f"{RES}/{l}_{part}.json")).items():
                curves.setdefault("Sentence Embedding" if name == "Dense" else name, []).append(v["pr"])
    lv = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m in MODELS:
        ax.plot(lv, np.mean(curves[m], axis=0), marker="o", label=m)
    ax.set_xlabel("Recall"); ax.set_ylabel("Interpolated precision")
    ax.set_title("Precision-Recall: " + ("average of 16 languages" if len(langs) > 1 else NAMES[langs[0]]))
    ax.grid(alpha=.3); ax.legend()
    return fig

with gr.Blocks(title="MIRACL Multilingual Search") as demo:
    gr.Markdown("# Multilingual Search on MIRACL\nSearch Wikipedia passages in 16 languages with four ranking models.")
    with gr.Tab("Search"):
        with gr.Row():
            lang = gr.Dropdown(LANG_CHOICES, value=[c for c in LANG_CHOICES if c.startswith("hi")][0], label="Language")
            sample = gr.Dropdown([OWN], value=OWN, label="Sample dev queries")
        query = gr.Textbox(label="Query", placeholder="Type a question in the selected language")
        with gr.Row():
            model = gr.Radio(MODELS + [ALL], value=ALL, label="Ranking model")
            topk = gr.Slider(5, 20, value=10, step=1, label="Results per model")
        btn = gr.Button("Search", variant="primary")
        out = gr.HTML()
        lang.change(on_language, lang, [sample, query])
        sample.change(on_sample, sample, query)
        btn.click(run_search, [lang, query, model, topk], out)
        query.submit(run_search, [lang, query, model, topk], out)
        demo.load(on_language, lang, [sample, query])
    with gr.Tab("Cross-lingual"):
        gr.Markdown("One embedding index over **all 16 languages** (~800,000 passages). "
                    "Write the query in any language; results can come from any language. "
                    "The first search loads the index and takes a little longer.")
        cq = gr.Textbox(label="Query (any language)")
        ck = gr.Slider(5, 20, value=10, step=1, label="Results")
        cb = gr.Button("Search all languages", variant="primary")
        co = gr.HTML()
        cb.click(run_cross, [cq, ck], co)
        cq.submit(run_cross, [cq, ck], co)
    with gr.Tab("Results"):
        gr.Markdown("**Macro average over 16 languages**")
        gr.Dataframe(value=MACRO, interactive=False)
        gr.Markdown("**Per language** (note: P@10 is capped by the number of relevant passages per query, "
                    "which is only about 1 to 4.6 depending on the language)")
        flt = gr.Dropdown(["All languages"] + LANGS, value="All languages", label="Filter by language")
        tbl = gr.Dataframe(value=COMP, interactive=False)
        flt.change(table_for, flt, tbl)
    with gr.Tab("Precision-Recall"):
        prd = gr.Dropdown(["Average of all 16"] + LANG_CHOICES, value="Average of all 16", label="Language")
        plot = gr.Plot()
        prd.change(pr_plot, prd, plot)
        demo.load(pr_plot, prd, plot)

if __name__ == "__main__":
    for l in ("hi", "en"):
        get_engine(l)                      # warm up: builds indexes before the first user query
        demo.queue().launch(share=os.environ.get("SHARE", "1") == "1")
