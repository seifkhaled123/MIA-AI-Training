"""Find FIFA 19 players with profiles similar to M. Salah.

All four similarity metrics and the PCA projection are implemented directly
with vectorized NumPy operations.  No sklearn/scipy metric or PCA functions are
used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURES = [
    "Finishing",
    "ShortPassing",
    "Dribbling",
    "SprintSpeed",
    "Strength",
    "Stamina",
    "Interceptions",
    "StandingTackle",
    "Value",
]

METRICS = {
    "Cosine": {
        "higher_is_better": True,
        "explanation": (
            "Compares the angle between two vectors. It rewards the same "
            "overall direction/profile and is less sensitive to magnitude."
        ),
    },
    "Euclidean": {
        "higher_is_better": False,
        "explanation": (
            "Measures straight-line distance. It rewards players whose "
            "standardized attribute levels are jointly close to Salah's."
        ),
    },
    "Manhattan": {
        "higher_is_better": False,
        "explanation": (
            "Adds absolute feature-by-feature gaps. It measures absolute "
            "levels but is less dominated by one large mismatch than squared "
            "Euclidean distance."
        ),
    },
    "Pearson": {
        "higher_is_better": True,
        "explanation": (
            "Correlates the shapes of two player profiles after centering each "
            "player's vector. Similar rises and falls matter more than the "
            "players' absolute levels."
        ),
    },
}


def parse_value(value: object) -> float:
    """Convert FIFA values such as ``€69.5M`` and ``€500K`` to euros."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace("€", "").replace(",", "")
    if not text:
        return np.nan

    multiplier = 1.0
    if text[-1].upper() in {"K", "M", "B"}:
        multiplier = {
            "K": 1_000.0,
            "M": 1_000_000.0,
            "B": 1_000_000_000.0,
        }[text[-1].upper()]
        text = text[:-1]

    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def load_player_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the same complete-case player pool and features used in Task 2.2."""
    metadata = ["ID", "Name", "Age", "Nationality", "Club", "Position"]
    frame = pd.read_csv(
        csv_path,
        encoding="cp1252",
        usecols=[*metadata, *FEATURES],
    )
    frame["Value"] = frame["Value"].map(parse_value)

    for feature in FEATURES[:-1]:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    # Club and Age are display/business fields, not similarity features, so
    # they are not used to decide whether a feature vector is complete.
    return frame.dropna(subset=["Name", "Position", *FEATURES]).reset_index(
        drop=True
    )


def standardize_features(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return population z-scores together with feature means and deviations."""
    means = matrix.mean(axis=0)
    deviations = matrix.std(axis=0, ddof=0)
    if np.any(deviations == 0):
        raise ValueError("Cannot standardize a feature with zero variance.")
    return (matrix - means) / deviations, means, deviations


def cosine_similarity(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Cosine of the angle between every row and the target vector."""
    numerators = matrix @ target
    denominators = np.linalg.norm(matrix, axis=1) * np.linalg.norm(target)
    return np.divide(
        numerators,
        denominators,
        out=np.full(matrix.shape[0], -np.inf),
        where=denominators != 0,
    )


def euclidean_distance(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Straight-line (L2) distance from every row to the target vector."""
    differences = matrix - target
    return np.sqrt(np.sum(differences * differences, axis=1))


def manhattan_distance(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Sum of absolute (L1) differences from every row to the target."""
    return np.sum(np.abs(matrix - target), axis=1)


def pearson_correlation(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pearson correlation between each row's profile and the target profile."""
    centered_rows = matrix - matrix.mean(axis=1, keepdims=True)
    centered_target = target - target.mean()
    numerators = centered_rows @ centered_target
    denominators = (
        np.linalg.norm(centered_rows, axis=1)
        * np.linalg.norm(centered_target)
    )
    return np.divide(
        numerators,
        denominators,
        out=np.full(matrix.shape[0], -np.inf),
        where=denominators != 0,
    )


def calculate_metrics(
    matrix: np.ndarray, target_index: int
) -> dict[str, np.ndarray]:
    """Compute every metric against the complete player pool."""
    target = matrix[target_index]
    return {
        "Cosine": cosine_similarity(matrix, target),
        "Euclidean": euclidean_distance(matrix, target),
        "Manhattan": manhattan_distance(matrix, target),
        "Pearson": pearson_correlation(matrix, target),
    }


def rank_metric(
    scores: np.ndarray,
    target_index: int,
    higher_is_better: bool,
    top_n: int = 5,
) -> np.ndarray:
    """Return indices of the best matches, excluding the target player."""
    sortable = scores.copy()
    sortable[target_index] = -np.inf if higher_is_better else np.inf
    order = np.argsort(sortable, kind="stable")
    if higher_is_better:
        order = order[::-1]
    return order[:top_n]


def complete_ranks(
    scores: np.ndarray, target_index: int, higher_is_better: bool
) -> np.ndarray:
    """Assign a 1-based rank to every player without a ranking library."""
    sortable = scores.copy()
    sortable[target_index] = -np.inf if higher_is_better else np.inf
    order = np.argsort(sortable, kind="stable")
    if higher_is_better:
        order = order[::-1]

    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    ranks[target_index] = len(scores)
    return ranks


def result_rows(
    players: pd.DataFrame,
    scores_by_metric: dict[str, np.ndarray],
    target_index: int,
    experiment: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Build a tidy top-N result table for every metric."""
    tables: list[pd.DataFrame] = []
    display_columns = [
        "ID",
        "Name",
        "Age",
        "Nationality",
        "Club",
        "Position",
        "Value",
    ]

    for metric, scores in scores_by_metric.items():
        indices = rank_metric(
            scores,
            target_index,
            METRICS[metric]["higher_is_better"],
            top_n,
        )
        table = players.loc[indices, display_columns].copy()
        table.insert(0, "Rank", np.arange(1, len(table) + 1))
        table.insert(0, "Metric", metric)
        table.insert(0, "Experiment", experiment)
        table["Score"] = scores[indices]
        tables.append(table)

    return pd.concat(tables, ignore_index=True)


def pca_from_scratch(
    standardized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project standardized data using covariance and NumPy eigen-decomposition."""
    covariance = (standardized.T @ standardized) / (len(standardized) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Eigenvector signs are arbitrary; orient them reproducibly.
    for column in range(eigenvectors.shape[1]):
        largest_loading = np.argmax(np.abs(eigenvectors[:, column]))
        if eigenvectors[largest_loading, column] < 0:
            eigenvectors[:, column] *= -1

    return standardized @ eigenvectors[:, :2], eigenvalues / eigenvalues.sum()


def choose_final_shortlist(
    players: pd.DataFrame,
    standardized_scores: dict[str, np.ndarray],
    target_index: int,
    count: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose young consensus matches using mean rank across all four metrics.

    Age is only a scouting/business filter: the similarity vectors remain the
    exact nine features from Task 2.2.  A successor should not already be older
    than Salah at the time represented by FIFA 19.
    """
    metric_ranks = np.vstack(
        [
            complete_ranks(
                standardized_scores[name],
                target_index,
                METRICS[name]["higher_is_better"],
            )
            for name in METRICS
        ]
    )
    mean_rank = metric_ranks.mean(axis=0)

    target_age = players.loc[target_index, "Age"]
    eligible = players["Age"].notna() & (players["Age"] <= target_age)
    eligible.iloc[target_index] = False

    selection_score = mean_rank.copy()
    selection_score[~eligible.to_numpy()] = np.inf
    selected = np.argsort(selection_score, kind="stable")[:count]
    return selected, metric_ranks


def plot_pca(
    scores: np.ndarray,
    players: pd.DataFrame,
    target_index: int,
    shortlist_indices: np.ndarray,
    explained_ratio: np.ndarray,
    output_path: Path,
) -> None:
    """Plot the player pool, Salah, and the final shortlist in PC space."""
    fig, axis = plt.subplots(figsize=(11, 7))
    axis.scatter(
        scores[:, 0],
        scores[:, 1],
        color="lightgray",
        s=9,
        alpha=0.35,
        linewidths=0,
        label="All other players",
    )

    colors = plt.cm.tab10(np.linspace(0, 0.8, len(shortlist_indices)))
    for index, color in zip(shortlist_indices, colors):
        axis.scatter(
            scores[index, 0],
            scores[index, 1],
            color=color,
            edgecolor="black",
            s=75,
            zorder=3,
        )
        axis.annotate(
            players.loc[index, "Name"],
            (scores[index, 0], scores[index, 1]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )

    axis.scatter(
        scores[target_index, 0],
        scores[target_index, 1],
        marker="*",
        color="crimson",
        edgecolor="black",
        s=220,
        zorder=4,
        label="M. Salah",
    )
    axis.annotate(
        "M. Salah",
        (scores[target_index, 0], scores[target_index, 1]),
        xytext=(7, -14),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )

    axis.set_title("FIFA 19 player profiles: Salah and recommended successors")
    axis.set_xlabel(f"PC1 ({explained_ratio[0]:.1%} variance)")
    axis.set_ylabel(f"PC2 ({explained_ratio[1]:.1%} variance)")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    float_columns: set[str] | None = None,
) -> str:
    """Create a small Markdown table without requiring ``tabulate``."""
    float_columns = float_columns or set()
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [header, separator]

    for _, row in frame[columns].iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if column == "Value":
                text = f"€{float(value) / 1_000_000:.1f}M"
            elif column in float_columns:
                text = f"{float(value):.6g}"
            elif pd.isna(value):
                text = "N/A"
            elif isinstance(value, (float, np.floating)) and value.is_integer():
                text = str(int(value))
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def top_five_overlap(
    scores_by_metric: dict[str, np.ndarray], target_index: int
) -> pd.DataFrame:
    """Count shared names between every pair of metric top-five sets."""
    names = list(METRICS)
    top_sets = {
        name: set(
            rank_metric(
                scores_by_metric[name],
                target_index,
                METRICS[name]["higher_is_better"],
            )
        )
        for name in names
    }
    values = [
        [len(top_sets[left] & top_sets[right]) for right in names]
        for left in names
    ]
    return pd.DataFrame(values, index=names, columns=names)


def write_report(
    output_path: Path,
    players: pd.DataFrame,
    target_index: int,
    standardized_results: pd.DataFrame,
    raw_results: pd.DataFrame,
    overlap: pd.DataFrame,
    deviations: np.ndarray,
    shortlist: pd.DataFrame,
    explained_ratio: np.ndarray,
) -> None:
    """Write the requested explanation and results as a reproducible report."""
    target = players.loc[target_index]
    lines = [
        "# The Scouting Engine — Finding Salah's successor",
        "",
        "## Data and method",
        "",
        (
            f"The cleaned pool contains **{len(players):,} players**. The target "
            f"is **{target['Name']}** ({target['Club']}, {target['Position']}, "
            f"age {int(target['Age'])}). Every player is represented by the "
            "same nine features as Task 2.2:"
        ),
        "",
        "`" + "`, `".join(FEATURES) + "`.",
        "",
        (
            "For the main experiment, each feature is converted to a population "
            "z-score, `(value - mean) / standard deviation`. Thus a one-unit "
            "difference has the same meaning—one dataset standard deviation—"
            "for every feature. All metric calculations compare the target "
            "against the full matrix in vectorized NumPy code."
        ),
        "",
        "## Standardized top-five results",
        "",
    ]

    for metric in METRICS:
        table = standardized_results[
            standardized_results["Metric"] == metric
        ]
        direction = (
            "higher is more similar"
            if METRICS[metric]["higher_is_better"]
            else "lower is more similar"
        )
        lines.extend(
            [
                f"### {metric}",
                "",
                f"{METRICS[metric]['explanation']} **{direction}.**",
                "",
                markdown_table(
                    table,
                    ["Rank", "Name", "Age", "Club", "Position", "Score"],
                    {"Score"},
                ),
                "",
            ]
        )

    overlap_display = overlap.copy()
    overlap_display.insert(0, "Metric", overlap_display.index)
    lines.extend(
        [
            "## Agreement and differences between metrics",
            "",
            "Each cell is the number of shared players between two top-five lists:",
            "",
            markdown_table(overlap_display, ["Metric", *METRICS.keys()]),
            "",
            (
                "Euclidean and Manhattan agree most because both compare "
                "absolute standardized attribute levels. Manhattan adds each "
                "gap directly; Euclidean squares gaps before combining them, "
                "so one unusually large mismatch has more influence on "
                "Euclidean distance."
            ),
            "",
            (
                "Cosine and Pearson emphasize profile shape. Cosine keeps the "
                "origin and compares vector direction; Pearson first subtracts "
                "each player's own across-feature mean, so it ignores a general "
                "level shift even more strongly. They can therefore select an "
                "elite player with Salah-like strengths and weaknesses despite "
                "larger absolute gaps. Limited overlap is expected: 'same "
                "levels' and 'same shape' are different definitions of similar."
            ),
            "",
            "## Standardized versus raw experiment",
            "",
            "The raw top-five lists are:",
            "",
            markdown_table(
                raw_results,
                ["Metric", "Rank", "Name", "Position", "Value", "Score"],
                {"Score"},
            ),
            "",
        ]
    )

    scale_frame = pd.DataFrame(
        {"Feature": FEATURES, "Raw standard deviation": deviations}
    ).sort_values("Raw standard deviation", ascending=False)
    lines.extend(
        [
            "Raw feature scales make the cause explicit:",
            "",
            markdown_table(
                scale_frame,
                ["Feature", "Raw standard deviation"],
                {"Raw standard deviation"},
            ),
            "",
            (
                "**Value dominates the raw search.** It is measured in euros "
                "(millions), while skill ratings are roughly on a 0–100 scale. "
                "For raw Euclidean and Manhattan distance, being close in market "
                "value overwhelms hundreds of rating points—hence even "
                "positionally implausible matches can appear. Raw cosine and "
                "Pearson also become almost 1 for elite players because the "
                "huge Value coordinate controls their vector direction and "
                "covariation."
            ),
            "",
            (
                "Without standardization, 'similar' secretly means **similar in "
                "the numerically largest unit**, not equally similar across the "
                "chosen football attributes. This is a units-of-measure result, "
                "not a football conclusion."
            ),
            "",
            "## Recommended shortlist",
            "",
            (
                "The recommendation averages each player's full-pool rank "
                "across all four standardized metrics, then applies one "
                "practical successor rule: the player must be no older than "
                f"Salah's FIFA 19 age ({int(target['Age'])}). Age is not added "
                "to the similarity vector; it is only a final scouting filter."
            ),
            "",
            markdown_table(
                shortlist,
                [
                    "Recommendation",
                    "Name",
                    "Age",
                    "Club",
                    "Position",
                    "Mean metric rank",
                ],
                {"Mean metric rank"},
            ),
            "",
            (
                f"**Primary recommendation: {shortlist.iloc[0]['Name']}.** "
                "This player has the strongest combined agreement across "
                "absolute-level and profile-shape measures while satisfying "
                "the age requirement. The remaining names provide alternatives "
                "with different trade-offs: distance-led matches reproduce "
                "Salah's current levels, while angle/correlation-led matches "
                "reproduce the shape of his strengths."
            ),
            "",
            (
                "This is a first-pass shortlist, not a transfer decision. The "
                "data are FIFA 19 ratings, Value is time-sensitive and partly "
                "subjective, and position, footedness, tactical role, injuries, "
                "cost, and current form still require scout review."
            ),
            "",
            "## Bonus: PCA visual check",
            "",
            "![PCA view of the shortlist](outputs/pca_shortlist.png)",
            "",
            (
                f"PC1 and PC2 retain **{explained_ratio[:2].sum():.2%}** of the "
                "nine-feature standardized variance "
                f"({explained_ratio[0]:.2%} + {explained_ratio[1]:.2%}). The "
                "highlighted candidates occupy Salah's broader high-level "
                "attacking region, which supports the shortlist as a sensible "
                "first pass. They need not be the five closest points in this "
                "plot: the search used all nine dimensions and four definitions "
                "of similarity, while the chart discards the variance outside "
                "the first two components."
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path("../Task 2.2/data/kl.csv"),
        help="Path to the FIFA 19 complete player CSV.",
    )
    parser.add_argument(
        "--target",
        default="M. Salah",
        help="Exact player name to use as the target (default: M. Salah).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated CSV, JSON, and PNG files.",
    )
    args = parser.parse_args()

    players = load_player_data(args.csv_path)
    target_matches = players.index[players["Name"] == args.target].to_numpy()
    if len(target_matches) != 1:
        raise ValueError(
            f"Expected one exact match for {args.target!r}; found "
            f"{len(target_matches)}."
        )
    target_index = int(target_matches[0])

    raw_matrix = players[FEATURES].to_numpy(dtype=float)
    standardized, _, deviations = standardize_features(raw_matrix)
    standardized_scores = calculate_metrics(standardized, target_index)
    raw_scores = calculate_metrics(raw_matrix, target_index)

    standardized_results = result_rows(
        players, standardized_scores, target_index, "Standardized"
    )
    raw_results = result_rows(players, raw_scores, target_index, "Raw")

    shortlist_indices, metric_ranks = choose_final_shortlist(
        players, standardized_scores, target_index
    )
    shortlist = players.loc[
        shortlist_indices,
        ["ID", "Name", "Age", "Nationality", "Club", "Position", "Value"],
    ].copy()
    shortlist.insert(0, "Recommendation", np.arange(1, len(shortlist) + 1))
    shortlist["Mean metric rank"] = metric_ranks[:, shortlist_indices].mean(
        axis=0
    )
    for row, metric in enumerate(METRICS):
        shortlist[f"{metric} rank"] = metric_ranks[row, shortlist_indices]

    pca_scores, explained_ratio = pca_from_scratch(standardized)
    pca_target_distance = np.linalg.norm(
        pca_scores - pca_scores[target_index], axis=1
    )
    shortlist["PC1"] = pca_scores[shortlist_indices, 0]
    shortlist["PC2"] = pca_scores[shortlist_indices, 1]
    shortlist["2D PCA distance"] = pca_target_distance[shortlist_indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    standardized_results.to_csv(
        args.output_dir / "top5_standardized.csv", index=False
    )
    raw_results.to_csv(args.output_dir / "top5_raw.csv", index=False)
    shortlist.to_csv(args.output_dir / "final_shortlist.csv", index=False)

    plot_pca(
        pca_scores,
        players,
        target_index,
        shortlist_indices,
        explained_ratio,
        args.output_dir / "pca_shortlist.png",
    )

    overlap = top_five_overlap(standardized_scores, target_index)
    summary = {
        "target": args.target,
        "player_count": len(players),
        "features": FEATURES,
        "explained_variance_ratio": explained_ratio.tolist(),
        "top5_overlap": overlap.to_dict(),
        "final_shortlist": shortlist["Name"].tolist(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_report(
        Path("REPORT.md"),
        players,
        target_index,
        standardized_results,
        raw_results,
        overlap,
        deviations,
        shortlist,
        explained_ratio,
    )

    print(f"Analysed {len(players):,} players around {args.target}.")
    print("Recommended shortlist: " + ", ".join(shortlist["Name"]))
    print(f"Results written to {args.output_dir}/ and REPORT.md.")


if __name__ == "__main__":
    main()
