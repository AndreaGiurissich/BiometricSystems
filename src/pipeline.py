"""Model-agnostic identification + verification runner.

Any model exposing the unified interface plugs in here:
    .extract(image_gray_u8) -> 1D float32 descriptor
    .descriptor_length      -> int
Scoring is cosine (descriptors are compared by dot product after L2 norm).

The expensive gallery descriptors are cached per (model, condition). Preprocessing
(baseline vs contrast-mask) is applied to the grayscale image UPSTREAM, before the
model sees it, so every model shares the same condition definition.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


def _pair_seed(stem: str, identity) -> int:
    """Deterministic per-(probe, gallery-identity) RNG seed for RANSAC, so a
    pairwise score does not depend on execution order (serial/parallel/resume)."""
    return mf._stable_seed(0, f"{stem}|{mf._id_key(identity)}") & 0x7FFFFFFF


def _cache_sig(cfg: Dict, name: str, condition: str) -> str:
    """Signature of the params a cached gallery descriptor depends on.

    Covers the model's own config block and -- for the preprocessed condition --
    the preprocessing block, so changing any of them invalidates the cache instead
    of silently serving stale descriptors (project policy: no silent param changes)."""
    blocks = {"model": cfg["models"].get(name, {})}
    if condition == "preprocessed":
        blocks["preprocessing"] = cfg.get("preprocessing", {})
    payload = json.dumps(blocks, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:12]

from src import dataset as ds  # noqa: F401  (re-exported convenience)
from src import evaluation as ev
from src import manifest as mf
from src.preprocessing import contrast_mask


def build_model(name: str, cfg: Dict):
    if name == "gabor":
        from src.models.gabor import GaborModel
        return GaborModel(cfg)
    if name == "dinov2":
        from src.models.dinov2 import Dinov2Model
        return Dinov2Model(cfg)
    if name == "sift":
        from src.models.sift import SiftModel
        return SiftModel(cfg)
    raise SystemExit(f"unknown model: {name!r}")


# --- parallel scoring (pairwise models like SIFT are O(N x gallery)) ----------
# Workers are spawned (cross-platform; avoids cv2+fork issues) and each reloads
# the cached gallery features once via the initializer, then scores whole probes.
_W: Dict = {}


def _worker_init(name, cfg, gallery_cache, condition, ids):
    import pickle
    _W["model"] = build_model(name, cfg)
    _W["cfg"] = cfg
    _W["condition"] = condition
    _W["ids"] = ids
    _W["row_of"] = {idt: i for i, idt in enumerate(ids)}
    with open(gallery_cache, "rb") as fh:
        _W["gal"] = pickle.load(fh)["feats"]


def _score_probe(task):
    path, stem, identity, identity_str, alt = task
    m, gal, cfg = _W["model"], _W["gal"], _W["cfg"]
    ids, row_of, cond = _W["ids"], _W["row_of"], _W["condition"]
    pf = m.extract(read_image(path, cond, cfg))
    scores = np.array([m.score(pf, gf, pair_seed=_pair_seed(stem, gid))
                       for gid, gf in zip(ids, gal)], dtype=float)
    tr = row_of[identity]
    is_true = np.zeros(len(ids), dtype=bool)
    is_true[tr] = True
    rank = ev.rank_of_match(scores, is_true, higher_is_better=True)
    imp = mf.sample_impostor_ids(identity, ids, cfg)
    return {"stem": stem, "identity": identity_str, "alt": alt, "rank": int(rank),
            "genuine": float(scores[tr]),
            "impostors": [float(scores[row_of[i]]) for i in imp]}


def read_image(path: str, condition: str, cfg) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"cv2 could not read {path}")
    if condition == "preprocessed":
        img = contrast_mask(img, cfg)
    return img


def gallery_features(name, gallery, ids, model, cfg, condition, cache_dir):
    """Gallery features cached per (model, condition).

    Returns (mode, data):
      embedding -> ("embedding", (N, D) float32 matrix)
      pairwise  -> ("pairwise", list of per-image feature objects)
    """
    score_type = getattr(model, "score_type", "embedding")
    key = [mf._id_key(i) for i in ids]
    sig = _cache_sig(cfg, name, condition)
    cdir = Path(cache_dir) / condition

    if score_type == "embedding":
        cache = cdir / f"{name}_gallery.npz"
        if cache.exists():
            data = np.load(cache, allow_pickle=True)
            cached_sig = str(data["sig"]) if "sig" in data else ""
            if list(data["ids"]) == key and cached_sig == sig:
                print(f"  gallery descriptors: cache hit ({cache})")
                return "embedding", data["G"].astype(np.float32)
            print("  gallery cache stale (params changed) -> re-extracting")
        print(f"  extracting {len(ids)} gallery descriptors ({name}/{condition})...")
        G = np.zeros((len(ids), model.descriptor_length), dtype=np.float32)
        t0 = time.time()
        for i, idt in enumerate(ids):
            G[i] = model.extract(read_image(gallery[idt].path, condition, cfg))
            if (i + 1) % 1000 == 0:
                print(f"    {i + 1}/{len(ids)}  ({time.time() - t0:.0f}s)")
        cdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ids=np.array(key), G=G, sig=sig)
        return "embedding", G

    # pairwise (e.g. SIFT): variable-size features -> pickle cache.
    cache = cdir / f"{name}_gallery.pkl"
    if cache.exists():
        with cache.open("rb") as fh:
            data = pickle.load(fh)
        if data.get("ids") == key and data.get("sig") == sig:
            print(f"  gallery features: cache hit ({cache})")
            return "pairwise", data["feats"]
        print("  gallery cache stale (params changed) -> re-extracting")
    print(f"  extracting {len(ids)} gallery features ({name}/{condition})...")
    feats, t0 = [], time.time()
    for i, idt in enumerate(ids):
        feats.append(model.extract(read_image(gallery[idt].path, condition, cfg)))
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{len(ids)}  ({time.time() - t0:.0f}s)")
    cdir.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as fh:
        pickle.dump({"ids": key, "sig": sig, "feats": feats}, fh)
    return "pairwise", feats


def _load_done(jsonl: Path):
    records, done = [], set()
    if jsonl.exists():
        for line in jsonl.open(encoding="utf-8"):
            line = line.strip()
            if line:
                rec = json.loads(line)
                records.append(rec)
                done.add(rec["stem"])
    return records, done


def run_condition(name, level, condition, gallery, ids, probes, model, cfg, paths,
                  workers: int = 1, full: bool = False):
    tier = "B" if full else "A"
    print(f"\n== {name} | level={level} | condition={condition} | tier={tier} ==")
    row_of = {idt: i for i, idt in enumerate(ids)}
    # Gallery is the same 6000 reals regardless of tier, so the cache is shared.
    mode, gal = gallery_features(name, gallery, ids, model, cfg, condition, paths["cache_dir"])

    # Tier B (full probe set) uses a separate dir so it never collides with Tier A.
    suffix = "_full" if full else ""
    exp_dir = Path(paths["results_dir"]) / "raw" / f"{name}_{level}_{condition}{suffix}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    jsonl = exp_dir / "scores.jsonl"
    records, done = _load_done(jsonl)
    if done:
        print(f"  resuming: {len(done)} probes already scored")
    todo = [p for p in probes if p.stem not in done]

    flush_every = int(cfg["checkpointing"].get("flush_every_probes", 50))
    t0 = time.time()
    # Parallelize only the expensive pairwise path; embedding scoring is a fast
    # matrix multiply not worth the process overhead.
    use_parallel = mode == "pairwise" and workers > 1 and todo

    if use_parallel:
        import multiprocessing as mp
        gallery_cache = Path(paths["cache_dir"]) / condition / f"{name}_gallery.pkl"
        tasks = [(p.path, p.stem, p.identity, p.identity_str, p.alt) for p in todo]
        print(f"  scoring {len(todo)} probes on {workers} workers (spawn)...")
        ctx = mp.get_context("spawn")
        with jsonl.open("a", encoding="utf-8") as fh, \
                ctx.Pool(workers, initializer=_worker_init,
                         initargs=(name, cfg, str(gallery_cache), condition, ids)) as pool:
            for i, rec in enumerate(pool.imap_unordered(_score_probe, tasks, chunksize=2)):
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                if (i + 1) % flush_every == 0:
                    fh.flush()
                    print(f"    scored {len(done) + i + 1}/{len(probes)}  "
                          f"({time.time() - t0:.0f}s)")
    else:
        with jsonl.open("a", encoding="utf-8") as fh:
            for probe in todo:
                pf = model.extract(read_image(probe.path, condition, cfg))
                if mode == "embedding":
                    scores = gal @ pf
                else:
                    scores = np.array(
                        [model.score(pf, gf, pair_seed=_pair_seed(probe.stem, gid))
                         for gid, gf in zip(ids, gal)], dtype=float)
                true_row = row_of[probe.identity]
                is_true = np.zeros(len(ids), dtype=bool)
                is_true[true_row] = True
                rank = ev.rank_of_match(scores, is_true, higher_is_better=True)
                imp_ids = mf.sample_impostor_ids(probe.identity, ids, cfg)
                rec = {
                    "stem": probe.stem, "identity": probe.identity_str, "alt": probe.alt,
                    "rank": rank, "genuine": float(scores[true_row]),
                    "impostors": [float(scores[row_of[i]]) for i in imp_ids],
                }
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                if len(records) % flush_every == 0:
                    fh.flush()
                    print(f"    scored {len(records)}/{len(probes)}  ({time.time() - t0:.0f}s)")

    if todo:
        dt = time.time() - t0
        print(f"  scored {len(todo)} probes in {dt:.0f}s "
              f"({1000 * dt / len(todo):.0f} ms/probe vs {len(ids)} gallery"
              f"{', %d workers' % workers if use_parallel else ''})")
    return summarize(name, records, len(ids), cfg, level, condition, exp_dir, tier)


def summarize(name, records, n_gallery, cfg, level, condition, exp_dir, tier="A") -> dict:
    ranks = [r["rank"] for r in records]
    genuine = np.array([r["genuine"] for r in records], dtype=float)
    impostor = np.array([s for r in records for s in r["impostors"]], dtype=float)

    ec = cfg["evaluation"]
    ranks_to_report = ec.get("ranks_to_report", [1, 5, 10])
    idm = ev.identification_metrics(ranks, n_gallery, ranks_to_report)
    far_targets = ec.get("far_at_frr", [0.01])
    vm = ev.verification_metrics(genuine, impostor, higher_is_better=True,
                                 far_at_frr_targets=far_targets,
                                 frr_at_far_targets=far_targets)
    cmc = idm.pop("cmc")

    per_alt = {}
    for a in sorted({r["alt"] for r in records}):
        sub = [r for r in records if r["alt"] == a]
        sr = np.array([r["rank"] for r in sub])
        sg = np.array([r["genuine"] for r in sub], dtype=float)
        si = np.array([s for r in sub for s in r["impostors"]], dtype=float)
        per_alt[a] = {"n": len(sub),
                      **{f"rank_{k}": float(np.mean(sr <= k)) for k in ranks_to_report},
                      "eer": ev.eer(sg, si, higher_is_better=True),
                      "auc": ev.roc_auc(sg, si, higher_is_better=True)}

    summary = {
        "model": name, "level": level, "condition": condition, "tier": tier,
        "n_probes": len(records), "n_gallery": n_gallery,
        "n_impostor_scores": int(impostor.size),
        "identification": idm, "verification": vm, "per_alteration": per_alt,
    }
    with (exp_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    np.save(exp_dir / "cmc.npy", cmc)

    print(f"  -- results ({name}/{condition}) --")
    print(f"    Rank-1/5/10 : {idm.get('rank_1'):.4f} / {idm.get('rank_5'):.4f} "
          f"/ {idm.get('rank_10'):.4f}   MRR {idm['mrr']:.4f}")
    print(f"    EER {vm['eer']:.4f}   AUC {vm['auc']:.4f}   "
          f"FAR@FRR={far_targets[0]} {vm[f'far_at_frr_{far_targets[0]}']:.4f}")
    for a, pa in per_alt.items():
        print(f"      by-alt {a:<5} n={pa['n']:<5} "
              f"Rank-1 {pa['rank_1']:.4f}  EER {pa['eer']:.4f}")
    print(f"    summary -> {exp_dir / 'summary.json'}")
    return summary


def run(name, level, conditions, gallery, ids, probes, cfg, paths,
        workers: int = 1, full: bool = False) -> List[dict]:
    model = build_model(name, cfg)
    return [run_condition(name, level, c, gallery, ids, probes, model, cfg, paths,
                          workers, full)
            for c in conditions]
