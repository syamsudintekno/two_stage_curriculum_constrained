#!/usr/bin/env python3
"""Reproducible structural validation for a curriculum-constrained EdRS.

The experiment is intentionally simulation-based. It evaluates decision-layer
behavior, not learning effectiveness or the predictive quality of LightGCN.
Locked constraints:
  C1: prerequisite compliance (all ancestors must be mastered)
  C2: one-sided difficulty alignment (penalize overchallenge only)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


N_CONCEPTS = 52
N_EDGES = 74
K = 10
GRAPH_SEED = 20260904
LEARNER_SEEDS = (20260904, 20260905, 20260906, 20260907, 20260908)
BETAS = (0.30, 0.50, 0.70, 0.90, 1.00)
N_LEARNERS = 1000
ABILITY_MARGIN = 0.08


def build_dag(n_nodes: int = N_CONCEPTS, n_edges: int = N_EDGES, seed: int = GRAPH_SEED):
    """Build a deterministic DAG with exactly n_edges; edges always point forward."""
    rng = np.random.default_rng(seed)
    # Four broad curriculum layers keep the graph progressive without reducing
    # the feasible frontier to a single next concept (as a simple chain would).
    layer_size = n_nodes // 4
    layers = [list(range(i * layer_size, (i + 1) * layer_size)) for i in range(4)]
    edges: set[tuple[int, int]] = set()
    for layer in range(1, 4):
        for target in layers[layer]:
            source = int(rng.choice(layers[layer - 1]))
            edges.add((source, target))
    possible = [
        (source, target)
        for target_layer in range(1, 4)
        for target in layers[target_layer]
        for source_layer in range(target_layer)
        for source in layers[source_layer]
        if (source, target) not in edges
    ]
    rng.shuffle(possible)
    for edge in possible:
        if len(edges) >= n_edges:
            break
        edges.add(edge)

    parents = {i: set() for i in range(n_nodes)}
    for source, target in edges:
        parents[target].add(source)

    ancestors = {i: set() for i in range(n_nodes)}
    for node in range(n_nodes):
        for parent in parents[node]:
            ancestors[node].add(parent)
            ancestors[node].update(ancestors[parent])
    return sorted(edges), parents, ancestors


def concept_metadata(n_nodes: int, seed: int = GRAPH_SEED):
    rng = np.random.default_rng(seed + 11)
    features = rng.normal(size=(n_nodes, 6))
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    difficulty = np.clip(np.linspace(0.05, 0.95, n_nodes) + rng.normal(0, 0.035, n_nodes), 0, 1)
    popularity = rng.beta(2.0, 5.0, n_nodes)
    return features, difficulty, popularity


def sample_mastery(target_size: int, parents: dict[int, set[int]], rng: np.random.Generator):
    """Sample a prerequisite-closed mastered set."""
    mastered: set[int] = set()
    while len(mastered) < target_size:
        eligible = [i for i in parents if i not in mastered and parents[i].issubset(mastered)]
        if not eligible:
            break
        # Mostly follow progression but preserve learner heterogeneity.
        weights = np.exp(-0.055 * np.asarray(eligible, dtype=float))
        weights /= weights.sum()
        mastered.add(int(rng.choice(eligible, p=weights)))
    return mastered


def ndcg_at_k(ranked: list[int], candidates: list[int], gain: np.ndarray, k: int = K) -> float:
    if not ranked:
        return 0.0
    ranked_gain = gain[np.asarray(ranked[:k], dtype=int)]
    discounts = 1.0 / np.log2(np.arange(2, len(ranked_gain) + 2))
    dcg = float(np.sum((2.0**ranked_gain - 1.0) * discounts))
    ideal = np.sort(gain[np.asarray(candidates, dtype=int)])[::-1][:k]
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
    idcg = float(np.sum((2.0**ideal - 1.0) * ideal_discounts))
    return dcg / idcg if idcg > 0 else 0.0


def rank_top(scores: np.ndarray, candidates: list[int], k: int = K) -> list[int]:
    return sorted(candidates, key=lambda item: (-scores[item], item))[:k]


def evaluate_list(
    ranked: list[int],
    candidates: list[int],
    mastered: set[int],
    ancestors: dict[int, set[int]],
    difficulty: np.ndarray,
    ability: float,
    gain: np.ndarray,
):
    violations = [not ancestors[item].issubset(mastered) for item in ranked]
    overchallenge = [max(0.0, difficulty[item] - ability - ABILITY_MARGIN) for item in ranked]
    return {
        "ndcg": ndcg_at_k(ranked, candidates, gain, K),
        "pvr": float(np.mean(violations)) if ranked else np.nan,
        "overchallenge": float(np.mean(overchallenge)) if ranked else np.nan,
        "list_length": len(ranked),
        "full_list": float(len(ranked) == K),
        "candidate_count": len(candidates),
    }


def simulate_one_seed(seed: int, beta: float, parents, ancestors, features, difficulty, popularity):
    rng = np.random.default_rng(seed)
    rows = []
    mastery_sizes = (8, 22, 38)

    for learner in range(N_LEARNERS):
        stage = learner % len(mastery_sizes)
        mastered = sample_mastery(mastery_sizes[stage], parents, rng)
        candidates = [i for i in range(N_CONCEPTS) if i not in mastered]
        valid = [i for i in candidates if ancestors[i].issubset(mastered)]

        pref = rng.normal(size=features.shape[1])
        pref /= np.linalg.norm(pref)
        latent = features @ pref
        latent = (latent - latent.min()) / (latent.max() - latent.min() + 1e-12)
        # Ground-truth gains are latent learner-item affinities. Base predictions
        # are noisy observations with a popularity component, so evaluation is
        # not circular with respect to the predicted score.
        gain = latent.copy()
        base = np.clip(0.78 * latent + 0.22 * popularity + rng.normal(0, 0.18, N_CONCEPTS), 0, 1)

        # A robust learner reference point: the upper quartile of mastered-skill
        # difficulty, rather than the single hardest observed concept.
        ability = float(np.quantile(difficulty[list(mastered)], 0.75)) if mastered else 0.0
        c1 = np.asarray([
            0.0 if ancestors[i].issubset(mastered)
            else len(ancestors[i] - mastered) / max(1, len(ancestors[i]))
            for i in range(N_CONCEPTS)
        ])
        c2 = np.maximum(0.0, difficulty - ability - ABILITY_MARGIN) ** 2

        # Fixed equal weighting avoids expert-weight claims in this initial study.
        soft_score = beta * base - (1.0 - beta) * (0.5 * c1 + 0.5 * c2)
        hard_score = beta * base - (1.0 - beta) * c2

        variants = {
            "relevance_only": rank_top(base, candidates),
            "soft_c1_c2": rank_top(soft_score, candidates),
            "hard_c1_then_c2": rank_top(hard_score, valid),
        }
        for variant, ranked in variants.items():
            metrics = evaluate_list(
                ranked, candidates, mastered, ancestors, difficulty, ability, gain
            )
            rows.append({
                "seed": seed,
                "beta": beta,
                "learner": learner,
                "stage": ("early", "intermediate", "advanced")[stage],
                "variant": variant,
                **metrics,
            })
    return rows


def main(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    edges, parents, ancestors = build_dag()
    features, difficulty, popularity = concept_metadata(N_CONCEPTS)

    all_rows = []
    for beta in BETAS:
        for seed in LEARNER_SEEDS:
            all_rows.extend(simulate_one_seed(
                seed, beta, parents, ancestors, features, difficulty, popularity
            ))
    raw = pd.DataFrame(all_rows)
    raw.to_csv(output_dir / "learner_level_results.csv", index=False)

    per_seed = (
        raw.groupby(["beta", "seed", "variant"], as_index=False)
        .agg(
            ndcg=("ndcg", "mean"),
            pvr=("pvr", "mean"),
            overchallenge=("overchallenge", "mean"),
            mean_list_length=("list_length", "mean"),
            full_list_rate=("full_list", "mean"),
        )
    )
    per_seed.to_csv(output_dir / "seed_level_results.csv", index=False)

    summary = (
        per_seed.groupby(["beta", "variant"], as_index=False)
        .agg(
            ndcg_mean=("ndcg", "mean"), ndcg_sd=("ndcg", "std"),
            pvr_mean=("pvr", "mean"), pvr_sd=("pvr", "std"),
            overchallenge_mean=("overchallenge", "mean"),
            overchallenge_sd=("overchallenge", "std"),
            list_length_mean=("mean_list_length", "mean"),
            full_list_rate_mean=("full_list_rate", "mean"),
        )
    )
    summary.to_csv(output_dir / "summary_results.csv", index=False)

    assert len(edges) == N_EDGES
    assert raw.query("variant == 'hard_c1_then_c2'")["pvr"].max() == 0.0
    assert raw["ndcg"].between(0, 1).all()

    pd.DataFrame(edges, columns=["prerequisite", "concept"]).to_csv(
        output_dir / "curriculum_edges.csv", index=False
    )
    pd.DataFrame({
        "concept": np.arange(N_CONCEPTS),
        "difficulty": difficulty,
        "popularity": popularity,
        "ancestor_count": [len(ancestors[i]) for i in range(N_CONCEPTS)],
    }).to_csv(output_dir / "concept_metadata.csv", index=False)

    config = {
        "scope": "structural simulation; no learning-effectiveness claim",
        "constraints": {"C1": "prerequisite compliance", "C2": "difficulty alignment"},
        "n_concepts": N_CONCEPTS,
        "n_edges": N_EDGES,
        "top_k": K,
        "n_learners_per_seed": N_LEARNERS,
        "graph_seed": GRAPH_SEED,
        "learner_seeds": list(LEARNER_SEEDS),
        "betas": list(BETAS),
        "ability_margin": ABILITY_MARGIN,
        "mastery_sizes": [8, 22, 38],
    }
    (output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Compact two-panel sensitivity figure for an IEEE-width manuscript.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), constrained_layout=True)
    styles = {
        "relevance_only": ("Relevance only", "#4C78A8", "o"),
        "soft_c1_c2": ("Soft C1+C2", "#F58518", "s"),
        "hard_c1_then_c2": ("Hard C1 + C2 rerank", "#54A24B", "^"),
    }
    for variant, (label, color, marker) in styles.items():
        part = summary[summary["variant"] == variant].sort_values("beta")
        axes[0].plot(part["beta"], part["ndcg_mean"], label=label, color=color, marker=marker, lw=1.5)
        axes[1].plot(part["beta"], part["pvr_mean"], label=label, color=color, marker=marker, lw=1.5)
    axes[0].set_ylabel("NDCG@10")
    axes[1].set_ylabel("PVR@10")
    for ax in axes:
        ax.set_xlabel(r"Relevance weight $\beta$")
        ax.set_xlim(0.28, 1.02)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    fig.savefig(output_dir / "sensitivity_tradeoff.png", dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_results"))
    args = parser.parse_args()
    main(args.output_dir)
