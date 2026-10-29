import regex, unicodedata, functools
from collections import Counter
from nltk.stem.snowball import SnowballStemmer

TOKEN_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+")
BIGRAM_LANGS = {"zh", "ja", "th"}
SNOWBALL = {"ar": "arabic", "en": "english", "es": "spanish",
            "fi": "finnish", "fr": "french", "ru": "russian"}

HI_SUFFIXES = sorted(["ाओं","ाएं","ाएँ","ियाँ","ियां","ियों","ेंगे","ोगे","ेगा","ेगी",
    "ाना","ाने","ाता","ाती","ाते","ता","ती","ते","ना","ने","ों","ें","ाँ","ां",
    "ी","ा","े","ो"], key=len, reverse=True)

def hindi_stem(w):
    for s in HI_SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 3:
            return w[:-len(s)]
    return w

def clean(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return regex.sub(r"https?://\S+", " ", text)

def tokenize(text, lang):
    toks = TOKEN_RE.findall(text)
    if lang in BIGRAM_LANGS:
        out = []
        for t in toks:
            out.extend([t] if len(t) == 1 else [t[i:i+2] for i in range(len(t) - 1)])
        return out
    return toks

def make_stemmer(lang):
    if lang in SNOWBALL:
        return functools.lru_cache(maxsize=None)(SnowballStemmer(SNOWBALL[lang]).stem)
    if lang == "hi":
        return functools.lru_cache(maxsize=None)(hindi_stem)
    return lambda w: w

def build_stopwords(token_lists, n=50):
    df = Counter()
    for toks in token_lists:
        df.update(set(toks))
    return {w for w, _ in df.most_common(n)}

def _norm(toks, stop, stem):
    out = []
    for w in toks:
        if w in stop:
            continue
        s = stem(w)
        if s not in stop:
            out.append(s)
    return out

def preprocess_corpus(df, lang):
    raw = [tokenize(clean(t), lang) for t in (df.title + " " + df.text)]
    stop = build_stopwords(raw)
    stem = make_stemmer(lang)
    return [_norm(t, stop, stem) for t in raw], stop, stem

def preprocess_query(q, lang, stop, stem):
    return _norm(tokenize(clean(q), lang), stop, stem)
