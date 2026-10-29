import os, pickle
from collections import OrderedDict
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from huggingface_hub import snapshot_download
from prep import preprocess_query, make_stemmer

LANGS = ['ar','bn','en','es','fa','fi','fr','hi','id','ja','ko','ru','sw','te','th','zh']
BASE = os.environ.get("MIRACL_DIR", "/content/drive/MyDrive/miracl")
CORPUS = f"{BASE}/corpus"
RES = f"{CORPUS}/results"
EMB = f"{CORPUS}/emb"
MODEL_NAME = "intfloat/multilingual-e5-small"

class TfidfRetriever:
    def __init__(self, texts):
        self.vec = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None,
                                   sublinear_tf=True, dtype=np.float32)
        self.D = self.vec.fit_transform(texts).T.tocsr()
    def scores(self, qtexts):
        return (self.vec.transform(qtexts) @ self.D).toarray()

class BM25:
    def __init__(self, texts, k1=1.2, b=0.75):
        self.cv = CountVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None)
        C = self.cv.fit_transform(texts).tocsr().astype(np.float32)
        N = C.shape[0]
        dl = np.asarray(C.sum(1)).ravel()
        df = np.bincount(C.indices, minlength=C.shape[1])
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        rows = np.repeat(np.arange(N), np.diff(C.indptr))
        tf = C.data
        C.data = (tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[rows] / dl.mean())) * idf[C.indices]).astype(np.float32)
        self.W = C.T.tocsr()
    def scores(self, qtexts):
        Q = self.cv.transform(qtexts).astype(np.float32)
        Q.data[:] = 1
        return (Q @ self.W).toarray()

def top_k(scores, k):
    k = min(k, len(scores))
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]

def rrf_fuse(rankings, k=60):
    sc = {}
    for r in rankings:
        for rank, d in enumerate(r, 1):
            sc[d] = sc.get(d, 0.0) + 1.0 / (k + rank)
    return sorted(sc.items(), key=lambda x: -x[1])

_model = None
def encode_query(q):
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
        _model.max_seq_length = 256
    return _model.encode("query: " + q, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

_dev_path = None
def load_dev(lang):
    global _dev_path
    if _dev_path is None:
        _dev_path = snapshot_download(repo_id="miracl/miracl", repo_type="dataset",
                                      allow_patterns=["miracl-v1.0-*/topics/*", "miracl-v1.0-*/qrels/*"])
    b = f"{_dev_path}/miracl-v1.0-{lang}"
    topics = pd.read_csv(f"{b}/topics/topics.miracl-v1.0-{lang}-dev.tsv", sep="\t", header=None,
                         names=["qid", "query"], dtype=str, quoting=3)
    qrels = pd.read_csv(f"{b}/qrels/qrels.miracl-v1.0-{lang}-dev.tsv", sep="\t", header=None,
                        names=["qid", "q0", "docid", "rel"], dtype={"qid": str, "docid": str, "rel": int})
    rel = qrels[qrels.rel == 1].groupby("qid").docid.apply(set).to_dict()
    return topics, rel

class LangEngine:
    def __init__(self, lang):
        self.lang = lang
        self.df = pd.read_parquet(f"{CORPUS}/{lang}.parquet")
        docids, docs, self.stop = pickle.load(open(f"{CORPUS}/{lang}_prep.pkl", "rb"))
        assert docids == self.df.docid.tolist(), "prep file does not match corpus"
        texts = [" ".join(d) for d in docs]
        self.tfidf, self.bm25 = TfidfRetriever(texts), BM25(texts)
        self.emb = np.load(f"{EMB}/{lang}_emb.npy").astype(np.float32)
        self.stem = make_stemmer(lang)
        self.topics, self.rel = load_dev(lang)
        self.q2qid = dict(zip(self.topics["query"].str.strip(), self.topics.qid))

    def _sparse(self, model, pq, k):
        s = model.scores([pq])[0]
        return [(int(i), float(s[i])) for i in top_k(s, k) if s[i] > 0]

    def _dense(self, q, k):
        s = self.emb @ encode_query(q)
        return [(int(i), float(s[i])) for i in top_k(s, k)]

    def search(self, query, model, k=10):
        pq = " ".join(preprocess_query(query, self.lang, self.stop, self.stem))
        if model == "TF-IDF":
            return self._sparse(self.tfidf, pq, k)
        if model == "BM25":
            return self._sparse(self.bm25, pq, k)
        if model == "Sentence Embedding":
            return self._dense(query, k)
        b = [i for i, _ in self._sparse(self.bm25, pq, 100)]      # Hybrid (RRF)
        d = [i for i, _ in self._dense(query, 100)]
        return rrf_fuse([b, d])[:k]

    def passage(self, i):
        r = self.df.iloc[i]
        return r.docid, r.title, r.text

_cache = OrderedDict()
MAX_LOADED = 4                                   # keep memory bounded
def get_engine(lang):
    if lang in _cache:
        _cache.move_to_end(lang)
        return _cache[lang]
    _cache[lang] = LangEngine(lang)
    while len(_cache) > MAX_LOADED:
        _cache.popitem(last=False)
    return _cache[lang]

# ---------- unified cross-lingual index (all 16 languages, embeddings only) ----------
_U = None; _U_meta = None; _dfs = {}
def get_unified():
    global _U, _U_meta
    if _U is None:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float16 if dev == "cuda" else torch.float32
        mats, langs, rows = [], [], []
        for l in LANGS:
            E = np.load(f"{EMB}/{l}_emb.npy"); mats.append(E)
            langs += [l] * len(E); rows += range(len(E))
        _U = torch.from_numpy(np.vstack(mats)).to(device=dev, dtype=dt)
        _U_meta = (np.array(langs), np.array(rows))
    return _U, _U_meta

def search_all(query, k=10):
    import torch
    U, (L, R) = get_unified()
    q = torch.from_numpy(encode_query(query)).to(device=U.device, dtype=U.dtype)
    vals, idx = (U @ q).topk(k)
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        l = str(L[i])
        if l not in _dfs:
            _dfs[l] = pd.read_parquet(f"{CORPUS}/{l}.parquet")
        r = _dfs[l].iloc[int(R[i])]
        out.append((l, r.docid, r.title, r.text, v))
    return out
