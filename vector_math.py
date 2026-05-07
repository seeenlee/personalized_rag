"""Vector combination and user-vector update helpers."""

from __future__ import annotations

import numpy as np

COMBINE_STRATEGIES = ("query-only", "linear-comb", "spherical-comb")
UPDATE_STRATEGIES = ("none", "moving-average")
DEFAULT_MOVING_AVERAGE_QUERY_WEIGHT = 0.1


def validate_alpha(alpha: float) -> float:
    """Validate an interpolation alpha value."""
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return value


def as_vector(vector: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Return a 1-D float vector."""
    result = np.asarray(vector, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("expected a non-empty 1-D vector")
    return result


def normalize_vector(vector: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Return a unit vector, preserving zero vectors."""
    result = as_vector(vector)
    norm = float(np.linalg.norm(result))
    if norm == 0.0:
        return result.copy()
    return result / norm


def _assert_same_shape(user_vector: np.ndarray, query_vector: np.ndarray) -> None:
    if user_vector.shape != query_vector.shape:
        raise ValueError(
            f"user and query vectors must have the same shape: "
            f"{user_vector.shape} != {query_vector.shape}"
        )


def linear_combination(
    user_vector: np.ndarray | list[float],
    query_vector: np.ndarray | list[float],
    alpha: float,
) -> np.ndarray:
    """Blend user and query vectors, then normalize the result."""
    alpha = validate_alpha(alpha)
    user = as_vector(user_vector)
    query = as_vector(query_vector)
    _assert_same_shape(user, query)
    fused = ((1.0 - alpha) * user) + (alpha * query)
    norm = float(np.linalg.norm(fused))
    if norm > 0.0:
        return fused / norm
    return normalize_vector(query)


def spherical_combination(
    user_vector: np.ndarray | list[float],
    query_vector: np.ndarray | list[float],
    alpha: float,
) -> np.ndarray:
    """Spherical interpolation from user vector to query vector."""
    alpha = validate_alpha(alpha)
    user = as_vector(user_vector)
    query = as_vector(query_vector)
    _assert_same_shape(user, query)

    user_norm = float(np.linalg.norm(user))
    query_norm = float(np.linalg.norm(query))
    if user_norm == 0.0:
        return normalize_vector(query)
    if query_norm == 0.0:
        return normalize_vector(user)

    v0 = user / user_norm
    v1 = query / query_norm
    dot = float(np.clip(np.dot(v0, v1), -1.0, 1.0))
    if abs(dot) > 0.9995:
        return linear_combination(v0, v1, alpha)

    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    scale_0 = np.sin((1.0 - alpha) * theta_0) / sin_theta_0
    scale_1 = np.sin(alpha * theta_0) / sin_theta_0
    return normalize_vector((scale_0 * v0) + (scale_1 * v1))


def combine_vectors(
    user_vector: np.ndarray | list[float],
    query_vector: np.ndarray | list[float],
    strategy: str,
    *,
    alpha: float | None = None,
) -> np.ndarray:
    """Combine user and query vectors with a named strategy."""
    query = as_vector(query_vector)
    if strategy == "query-only":
        return query
    if alpha is None:
        raise ValueError(f"{strategy} requires an alpha value")
    if strategy == "linear-comb":
        return linear_combination(user_vector, query, alpha)
    if strategy == "spherical-comb":
        return spherical_combination(user_vector, query, alpha)
    raise ValueError(f"unknown combination strategy: {strategy}")


def moving_average_user_vector(
    user_vector: np.ndarray | list[float],
    query_vector: np.ndarray | list[float],
    *,
    query_weight: float = DEFAULT_MOVING_AVERAGE_QUERY_WEIGHT,
) -> np.ndarray:
    """Update the user vector with a normalized moving average."""
    return linear_combination(user_vector, query_vector, alpha=query_weight)

