"""Difficulty-layer alignment sensitivity + macro/micro overchallenge check."""
from pathlib import Path

import numpy as np, pandas as pd
import conference_simulation as cs
import extended_experiments as ex
from scipy.stats import spearmanr
orig = cs.concept_metadata
def make(sigma, shuffle):
    def f(n, seed=cs.GRAPH_SEED):
        feats, diff, pop = orig(n, seed)
        rng = np.random.default_rng(seed + 99)
        if shuffle:
            diff = rng.permutation(diff)
        elif sigma is not None:
            diff = np.clip(np.linspace(0.05, 0.95, n) + rng.normal(0, sigma, n), 0, 1)
        return feats, diff, pop
    return f
settings = [("original (sigma=0.035)", None, False), ("sigma=0.15", 0.15, False),
            ("sigma=0.30", 0.30, False), ("shuffled (no alignment)", None, True)]
out = []
for name, sg, sh in settings:
    cs.concept_metadata = make(sg, sh)
    _, d, _ = cs.concept_metadata(cs.N_CONCEPTS)
    rho = spearmanr(np.arange(cs.N_CONCEPTS), d).correlation
    df = ex.run(cs.GRAPH_SEED, cs.N_EDGES, (0.3, 1.0), [(0.08, 0.75)], cs.LEARNER_SEEDS, cs.N_LEARNERS, name)
    def per_seed_mean(v, b):
        s = df[(df.variant == v) & ((df.beta == b) if b == b else df.beta.isna())]
        return s.groupby("seed")["over"].mean().mean(), s.groupby("seed")["ndcg"].mean().mean()
    r, rn = per_seed_mean("relevance_only", np.nan)
    h1, h1n = per_seed_mean("hard_c1_c2", 1.0)
    h3, h3n = per_seed_mean("hard_c1_c2", 0.3)
    out.append((name, rho, r, h1, h3, (r - h1) / (r - h3), h1 - h3, h1n - h3n))
res = pd.DataFrame(out, columns=["setting","spearman(idx,diff)","over_rel","over_C1only","over_C1C2","C1_share","C2_extra","ndcg_cost_C2"])
pd.set_option("display.width", 220)
print(res.round(4).to_string(index=False))
outdir = Path("extended_results")
outdir.mkdir(parents=True, exist_ok=True)
res.round(5).to_csv(outdir / "E7_difficulty_alignment.csv", index=False)

# macro vs micro overchallenge on default setting
cs.concept_metadata = orig
df = ex.run(cs.GRAPH_SEED, cs.N_EDGES, (0.3, 1.0), [(0.08, 0.75)], cs.LEARNER_SEEDS, cs.N_LEARNERS, "main")
print("\nmacro (mean of per-list means) vs micro (pooled over returned items)")
for v, b in [("relevance_only", np.nan), ("hard_c1_c2", 1.0), ("hard_c1_c2", 0.3)]:
    s = df[(df.variant == v) & ((df.beta == b) if b == b else df.beta.isna())]
    macro = s.groupby("seed")["over"].mean().mean()
    micro = (s.assign(w=s.over * s.len).groupby("seed")["w"].sum() / s.groupby("seed")["len"].sum()).mean()
    pm = s.groupby("seed")["pvr"].mean().mean()
    print(v, b, f"macro={macro:.4f} micro={micro:.4f}  empty lists={int((s.len==0).sum())}")
