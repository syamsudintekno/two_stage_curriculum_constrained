"""Stratify fixed-length violations by feasible-set shortfall.

Run from the repository root with python feasible_set_analysis.py.
Uses the released simulation and its original learner RNG sequence.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import conference_simulation as cs
import extended_experiments as ex

def main():
    data = ex.run(cs.GRAPH_SEED, cs.N_EDGES, (0.3, 1.0),
                  [(cs.ABILITY_MARGIN, 0.75)], cs.LEARNER_SEEDS,
                  cs.N_LEARNERS, "main")
    hard = data.loc[data["variant"].eq("hard_c1_c2") & data["beta"].eq(1.0),
                    ["seed", "learner", "len", "full"]].copy()
    hard["group"] = np.where(hard["full"].eq(1), "enough", "shortfall")
    hard["bound"] = (cs.K - hard["len"]) / cs.K
    profiles = hard.groupby("group").agg(n_profiles=("learner", "size"))
    profiles["share"] = profiles["n_profiles"] / len(hard)
    selected = data.loc[
        (data["variant"].eq("relevance_only")) |
        (data["beta"].eq(0.3) & data["variant"].isin(
            ["soft_frac_c1", "soft_bin_c1"])) |
        (data["beta"].eq(1.0) & data["variant"].eq("hard_c1_c2"))
    ].merge(hard[["seed", "learner", "group", "bound"]],
            on=["seed", "learner"], validate="many_to_one")
    per_seed = selected.groupby(["seed", "group", "variant"])[
        ["pvr", "bound", "ndcg", "ndcg_feas", "len", "full"]].mean()
    per_seed["excess_pvr"] = per_seed["pvr"] - per_seed["bound"]
    per_seed.loc[per_seed.index.get_level_values("variant") == "hard_c1_c2",
                 "excess_pvr"] = np.nan
    summary = per_seed.groupby(["group", "variant"]).agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.join(profiles, on="group").reset_index()
    output = Path(__file__).resolve().parent / "extended_results"
    output.mkdir(exist_ok=True)
    summary.to_csv(output / "E8_feasible_set_strata.csv", index=False)
    print(summary[["group", "variant", "share", "bound_mean", "pvr_mean",
                   "excess_pvr_mean", "ndcg_mean", "ndcg_feas_mean"]].to_string(index=False))

if __name__ == "__main__":
    main()
