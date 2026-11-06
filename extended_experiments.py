#!/usr/bin/env python3
"""Extended experiments addressing reviewer comments (ISRITI 2027 revision).

Reuses build_dag / concept_metadata / sample_mastery from conference_simulation.py
and keeps the same RNG consumption order, so default-setting numbers reproduce
the original summary_results.csv exactly.

Added analyses
  E1  hard C1-only ablation (= hard variant at beta=1) + C1/C2 decomposition
  E2  soft comparator at low beta (fractional vs binary C1 penalty)
  E3  feasibility-aware NDCG (infeasible items earn zero gain; ideal over valid set)
  E4  multiple curriculum graphs (graph seeds) and edge densities
  E5  sensitivity to delta and the ability quantile
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import conference_simulation as cs

K = cs.K
N = cs.N_CONCEPTS
STAGES = ("early", "intermediate", "advanced")
SIZES = (8, 22, 38)
LEARNER_SEEDS = cs.LEARNER_SEEDS
DISC = 1.0 / np.log2(np.arange(2, K + 2))


def anc_matrix(ancestors, n):
    m = np.zeros((n, n), dtype=bool)
    for i, a in ancestors.items():
        for j in a:
            m[i, j] = True
    return m


def topk(score, cand):
    order = cand[np.argsort(-score[cand], kind="stable")]
    return order[:K]


def dcg(g):
    if len(g) == 0:
        return 0.0
    return float(np.sum((2.0 ** g - 1.0) * DISC[: len(g)]))


def idcg(g):
    return dcg(np.sort(g)[::-1][:K])


def metrics(ranked, gain, valid, diff, ability, delta, idcg_all, idcg_valid):
    if len(ranked) == 0:
        return (0.0, 0.0, np.nan, np.nan, 0, 0.0)
    g = gain[ranked]
    ndcg_all = dcg(g) / idcg_all if idcg_all > 0 else 0.0
    gv = g * valid[ranked]
    ndcg_val = dcg(gv) / idcg_valid if idcg_valid > 0 else 0.0
    pvr = float(np.mean(~valid[ranked]))
    over = float(np.mean(np.maximum(0.0, diff[ranked] - ability - delta)))
    return (ndcg_all, ndcg_val, pvr, over, len(ranked), float(len(ranked) == K))


def run(graph_seed, n_edges, betas, delta_q_grid, learner_seeds, n_learners, tag):
    edges, parents, ancestors = cs.build_dag(N, n_edges, graph_seed)
    feats, diff, pop = cs.concept_metadata(N, graph_seed)
    anc = anc_matrix(ancestors, N).astype(np.int64)
    anc_cnt = anc.sum(1)
    rows = []
    for seed in learner_seeds:
        rng = np.random.default_rng(seed)
        for l in range(n_learners):
            stage = l % 3
            mastered = cs.sample_mastery(SIZES[stage], parents, rng)
            m = np.zeros(N, dtype=bool)
            m[list(mastered)] = True
            cand = np.flatnonzero(~m)
            unmet = anc @ (~m).astype(np.int64)
            valid_all = unmet == 0
            valid_idx = cand[valid_all[cand]]
            pref = rng.normal(size=feats.shape[1])
            pref /= np.linalg.norm(pref)
            lat = feats @ pref
            lat = (lat - lat.min()) / (lat.max() - lat.min() + 1e-12)
            gain = lat
            base = np.clip(0.78 * lat + 0.22 * pop + rng.normal(0, 0.18, N), 0, 1)
            c1f = unmet / np.maximum(1, anc_cnt)
            c1b = (~valid_all).astype(float)
            id_all = idcg(gain[cand])
            id_val = idcg(gain[valid_idx]) if len(valid_idx) else 0.0

            def add(variant, ranked, ability, delta, beta, extra=""):
                mt = metrics(ranked, gain, valid_all, diff, ability, delta, id_all, id_val)
                rows.append((tag, seed, l, STAGES[stage], variant, beta, delta, extra, *mt))

            for dq in delta_q_grid:
                delta, q = dq
                ability = float(np.quantile(diff[m], q))
                c2 = np.maximum(0.0, diff - ability - delta) ** 2
                default = (delta == cs.ABILITY_MARGIN and q == 0.75)
                sfx = "" if default else f"|d={delta}|q={q}"
                add("relevance_only", topk(base, cand), ability, delta, np.nan, sfx)
                for b in (betas if default else (0.3, 1.0)):
                    hard = b * base - (1 - b) * c2
                    add("hard_c1_c2", topk(hard, valid_idx), ability, delta, b, sfx)
                    if default:
                        s_frac = b * base - (1 - b) * (0.5 * c1f + 0.5 * c2)
                        s_bin = b * base - (1 - b) * (0.5 * c1b + 0.5 * c2)
                        s_fullc2 = b * base - (1 - b) * (0.5 * c1f + c2)
                        add("soft_frac_c1", topk(s_frac, cand), ability, delta, b, sfx)
                        add("soft_bin_c1", topk(s_bin, cand), ability, delta, b, sfx)
                        add("soft_frac_c1_fullC2", topk(s_fullc2, cand), ability, delta, b, sfx)
    cols = ["tag", "seed", "learner", "stage", "variant", "beta", "delta", "extra",
            "ndcg", "ndcg_feas", "pvr", "over", "len", "full"]
    return pd.DataFrame(rows, columns=cols)


def agg(df, by):
    per = df.groupby(["seed"] + by, dropna=False, as_index=False)[
        ["ndcg", "ndcg_feas", "pvr", "over", "len", "full"]].mean()
    g = per.groupby(by, dropna=False)[["ndcg", "ndcg_feas", "pvr", "over", "len", "full"]]
    mean, sd = g.mean(), g.std()
    out = mean.join(sd, rsuffix="_sd")
    return out.reset_index()


def main(outdir: Path, quick: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    nl = 200 if quick else cs.N_LEARNERS
    lseeds = LEARNER_SEEDS[:2] if quick else LEARNER_SEEDS
    default_dq = [(0.08, 0.75)]
    sens_dq = [(d, q) for d in (0.0, 0.04, 0.08, 0.15) for q in (0.5, 0.75, 1.0)]
    betas_main = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)

    # E1-E3, E5 on the paper's graph
    main_df = run(cs.GRAPH_SEED, cs.N_EDGES, betas_main, sens_dq, lseeds, nl, "main")
    # E4: other graphs and densities (reduced beta set)
    parts = []
    for gs in (cs.GRAPH_SEED + 100, cs.GRAPH_SEED + 200, cs.GRAPH_SEED + 300, cs.GRAPH_SEED + 400):
        parts.append(run(gs, cs.N_EDGES, (0.1, 0.3, 1.0), default_dq, lseeds, nl, f"graph{gs}"))
    for ne in (60, 100):
        parts.append(run(cs.GRAPH_SEED, ne, (0.1, 0.3, 1.0), default_dq, lseeds, nl, f"edges{ne}"))
    other = pd.concat(parts)
    main_df.to_pickle(outdir / "_main.pkl")
    other.to_pickle(outdir / "_other.pkl")

    # Summaries
    dflt = main_df[main_df["extra"] == ""]
    e1 = agg(dflt, ["variant", "beta"])
    e1.to_csv(outdir / "E1_E3_main_by_beta.csv", index=False)
    stage = agg(dflt[dflt["beta"].isin([0.3, 1.0]) | dflt["beta"].isna()], ["stage", "variant", "beta"])
    stage.to_csv(outdir / "E1_by_stage.csv", index=False)
    sens_src = main_df[main_df["variant"].isin(["relevance_only", "hard_c1_c2"]) &
                       (main_df["beta"].isin([0.3, 1.0]) | main_df["beta"].isna())].copy()
    sens_src["q"] = sens_src["extra"].str.extract(r"q=([0-9.]+)")[0].astype(float).fillna(0.75)
    agg(sens_src, ["delta", "q", "variant", "beta"]).to_csv(outdir / "E5_sensitivity_delta_q.csv", index=False)
    agg(other, ["tag", "variant", "beta"]).to_csv(outdir / "E4_other_graphs.csv", index=False)
    print("done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("extended_results"))
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    main(a.output_dir, a.quick)
