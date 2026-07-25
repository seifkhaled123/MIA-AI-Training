"""PCA from scratch for the FIFA 19 player dataset.

No pre-built PCA implementation is used.  The file contains both a NumPy
pipeline and a list-of-lists implementation for the bonus comparison.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
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


@dataclass
class PCAResult:
    standardized: np.ndarray | list[list[float]]
    covariance: np.ndarray | list[list[float]]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    scores: np.ndarray | list[list[float]]
    explained_variance_ratio: np.ndarray


def parse_value(value: object) -> float:
    """Convert FIFA values such as '€110.5M' and '€500K' to euros."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace("€", "").replace(",", "")
    if not text:
        return np.nan

    multiplier = 1.0
    suffix = text[-1].upper()
    if suffix in {"K", "M", "B"}:
        multiplier = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}[suffix]
        text = text[:-1]

    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def load_player_data(csv_path: str | Path) -> pd.DataFrame:
    """Load cp1252 FIFA data, parse Value, and remove incomplete selected rows."""
    columns = ["Name", "Position", *FEATURES]
    frame = pd.read_csv(csv_path, encoding="cp1252", usecols=columns)
    frame["Value"] = frame["Value"].map(parse_value)

    for feature in FEATURES[:-1]:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    return frame.dropna(subset=columns).reset_index(drop=True)


def _orient_eigenvectors(eigenvectors: np.ndarray) -> np.ndarray:
    """Resolve arbitrary eigenvector signs to make output reproducible."""
    oriented = eigenvectors.copy()
    for column in range(oriented.shape[1]):
        largest = int(np.argmax(np.abs(oriented[:, column])))
        if oriented[largest, column] < 0:
            oriented[:, column] *= -1
    return oriented


def numpy_pca(data: np.ndarray, n_components: int = 2) -> PCAResult:
    """Run PCA with NumPy matrix operations, following the requested steps."""
    means = data.mean(axis=0)
    standard_deviations = data.std(axis=0, ddof=0)
    if np.any(standard_deviations == 0):
        raise ValueError("PCA cannot standardize a feature with zero variance.")

    standardized = (data - means) / standard_deviations
    covariance = (standardized.T @ standardized) / (len(standardized) - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = _orient_eigenvectors(eigenvectors[:, order])

    projection_matrix = eigenvectors[:, :n_components]
    scores = standardized @ projection_matrix
    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    return PCAResult(
        standardized,
        covariance,
        eigenvalues,
        eigenvectors,
        scores,
        explained_variance_ratio,
    )


# ---------------------------------------------------------------------------
# Pure Python matrix operations (lists of lists)


def transpose_matrix(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        return []
    return [list(column) for column in zip(*matrix)]


def multiply_matrices(
    left: list[list[float]], right: list[list[float]]
) -> list[list[float]]:
    if not left or not right:
        return []
    if len(left[0]) != len(right):
        raise ValueError("Incompatible matrix dimensions.")

    right_transposed = transpose_matrix(right)
    return [
        [sum(a * b for a, b in zip(row, column)) for column in right_transposed]
        for row in left
    ]


def standardize_data(data: list[list[float]]) -> list[list[float]]:
    if not data:
        return []

    row_count = len(data)
    column_count = len(data[0])
    means = [
        sum(data[row][column] for row in range(row_count)) / row_count
        for column in range(column_count)
    ]
    standard_deviations = [
        (
            sum(
                (data[row][column] - means[column]) ** 2
                for row in range(row_count)
            )
            / row_count
        )
        ** 0.5
        for column in range(column_count)
    ]

    if any(deviation == 0 for deviation in standard_deviations):
        raise ValueError("PCA cannot standardize a feature with zero variance.")

    return [
        [
            (row[column] - means[column]) / standard_deviations[column]
            for column in range(column_count)
        ]
        for row in data
    ]


def calculate_covariance_matrix(
    standardized: list[list[float]],
) -> list[list[float]]:
    if len(standardized) < 2:
        raise ValueError("At least two rows are required.")

    products = multiply_matrices(transpose_matrix(standardized), standardized)
    denominator = len(standardized) - 1
    return [[value / denominator for value in row] for row in products]


def project_data(
    standardized: list[list[float]], eigenvectors: list[list[float]]
) -> list[list[float]]:
    return multiply_matrices(standardized, eigenvectors)


def pure_python_pca(
    data: list[list[float]], n_components: int = 2
) -> PCAResult:
    """Run all requested matrix work with lists; NumPy is used only for eigh."""
    standardized = standardize_data(data)
    covariance = calculate_covariance_matrix(standardized)

    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(covariance))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = _orient_eigenvectors(eigenvectors[:, order])

    selected = eigenvectors[:, :n_components].tolist()
    scores = project_data(standardized, selected)
    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    return PCAResult(
        standardized,
        covariance,
        eigenvalues,
        eigenvectors,
        scores,
        explained_variance_ratio,
    )


def benchmark_pipelines(
    array_data: np.ndarray,
    list_data: list[list[float]],
    repeats: int = 3,
) -> tuple[dict[str, float], PCAResult, PCAResult]:
    """Time complete pipelines and compare their numerical output."""
    numpy_times: list[float] = []
    python_times: list[float] = []
    numpy_result: PCAResult | None = None
    python_result: PCAResult | None = None

    for _ in range(repeats):
        start = time.perf_counter()
        numpy_result = numpy_pca(array_data)
        numpy_times.append(time.perf_counter() - start)

    for _ in range(repeats):
        start = time.perf_counter()
        python_result = pure_python_pca(list_data)
        python_times.append(time.perf_counter() - start)

    assert numpy_result is not None and python_result is not None
    numpy_seconds = float(np.median(numpy_times))
    python_seconds = float(np.median(python_times))
    metrics = {
        "numpy_seconds": numpy_seconds,
        "pure_python_seconds": python_seconds,
        "speedup": python_seconds / numpy_seconds,
        "max_score_absolute_error": float(
            np.max(
                np.abs(
                    np.asarray(numpy_result.scores)
                    - np.asarray(python_result.scores)
                )
            )
        ),
        "max_covariance_absolute_error": float(
            np.max(
                np.abs(
                    np.asarray(numpy_result.covariance)
                    - np.asarray(python_result.covariance)
                )
            )
        ),
    }
    return metrics, numpy_result, python_result


def make_scatter_plot(
    scores: np.ndarray,
    positions: pd.Series,
    explained_variance_ratio: np.ndarray,
    output_path: str | Path,
) -> None:
    """Create the requested PC1/PC2 plot, colored by original position."""
    categories = sorted(positions.unique())
    colors = plt.colormaps["turbo"](np.linspace(0, 1, len(categories)))

    fig, axis = plt.subplots(figsize=(13, 8))
    for position, color in zip(categories, colors):
        mask = positions.to_numpy() == position
        axis.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=12,
            alpha=0.55,
            color=color,
            label=position,
            edgecolors="none",
        )

    axis.axhline(0, color="0.75", linewidth=0.8)
    axis.axvline(0, color="0.75", linewidth=0.8)
    axis.set_xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.1f}% variance)")
    axis.set_ylabel(f"PC2 ({explained_variance_ratio[1] * 100:.1f}% variance)")
    axis.set_title("FIFA 19 players projected onto the first two principal components")
    axis.grid(alpha=0.15)
    axis.legend(
        title="Position",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        ncol=2,
        fontsize=8,
        markerscale=1.7,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    csv_path: str | Path = "data/kl.csv",
    output_directory: str | Path = "outputs",
) -> dict[str, object]:
    """Load data, benchmark both implementations, and write result artifacts."""
    frame = load_player_data(csv_path)
    array_data = frame[FEATURES].to_numpy(dtype=float)
    list_data = array_data.tolist()
    benchmark, result, manual_result = benchmark_pipelines(array_data, list_data)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    plot_path = output_directory / "pca_positions.png"
    make_scatter_plot(
        result.scores,
        frame["Position"],
        result.explained_variance_ratio,
        plot_path,
    )

    loadings = pd.DataFrame(
        result.eigenvectors[:, :2],
        index=FEATURES,
        columns=["PC1", "PC2"],
    )
    explained = pd.DataFrame(
        {
            "Component": [f"PC{i + 1}" for i in range(len(FEATURES))],
            "Eigenvalue": result.eigenvalues,
            "Explained variance": result.explained_variance_ratio,
            "Cumulative variance": np.cumsum(result.explained_variance_ratio),
        }
    )
    scores = pd.DataFrame(result.scores, columns=["PC1", "PC2"])
    scores.insert(0, "Position", frame["Position"])
    scores.insert(0, "Name", frame["Name"])
    scores.to_csv(output_directory / "player_pca_scores.csv", index=False)

    serializable = {
        "players_loaded": int(len(frame)),
        "features": FEATURES,
        "benchmark": benchmark,
        "explained_variance_ratio": result.explained_variance_ratio.tolist(),
        "eigenvalues": result.eigenvalues.tolist(),
        "loadings": {
            feature: {
                "PC1": float(loadings.loc[feature, "PC1"]),
                "PC2": float(loadings.loc[feature, "PC2"]),
            }
            for feature in FEATURES
        },
    }
    (output_directory / "results.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )

    return {
        "players": frame,
        "numpy_result": result,
        "manual_result": manual_result,
        "benchmark": benchmark,
        "loadings": loadings,
        "explained_variance": explained,
        "scores": scores,
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    analysis = run_analysis()
    print(f"Players analyzed: {len(analysis['players']):,}")
    print("\nExplained variance:")
    print(analysis["explained_variance"].head(4).to_string(index=False))
    print("\nPC loadings:")
    print(analysis["loadings"].round(4).to_string())
    print("\nBenchmark:")
    for name, value in analysis["benchmark"].items():
        print(f"{name}: {value:.8g}")
