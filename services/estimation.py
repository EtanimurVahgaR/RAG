from __future__ import annotations
from typing import Optional, Dict, Any
import re


def _clamp(value: int, min_v: int, max_v: int) -> int:
    return max(min_v, min(max_v, value))


def _parse_fraction(text: str) -> Optional[float]:
    """Very small helper to detect fractions/percentages in plain text.
    Returns a float in (0,1] or None.
    """
    t = (text or "").lower()
    # Percent like 25% or 30 percent
    m = re.search(r"(\d{1,3})\s*%", t)
    if m:
        v = int(m.group(1))
        if 0 < v <= 100:
            return v / 100.0
    m2 = re.search(r"(\d{1,2})\s*percent", t)
    if m2:
        v = int(m2.group(1))
        if 0 < v <= 100:
            return v / 100.0
    # Common words
    if "half" in t or "first half" in t or "second half" in t:
        return 0.5
    if "quarter" in t:
        return 0.25
    if "third" in t or "one third" in t:
        return 1 / 3
    return None


def estimate_k(
    user_query: str,
    min_k: int = 3,
    max_k: int = 40,
    default_k: int = 7,
    chunk_count: Optional[int] = None,
) -> int:
    """
    Simple, deterministic k estimator with no AI calls and no external lookups.
    Rules:
    - If the query asks for "top N", return N (clamped).
    - If the query contains a fraction/percentage (e.g., half, 25%), scale k by that fraction.
      If chunk_count is unknown, scale against max_k.
    - If the query looks broad/overview-like, use a larger fixed k.
    - If the query looks highly specific, use a small k.
    - Otherwise, use default_k.
    Always clamp to [min_k, min(max_k, chunk_count if provided)].
    """
    q = f" {user_query.lower()} " if user_query else ""

    # 1) Explicit "top N"
    m = re.search(r"\btop\s*(\d+)\b", q)
    if m:
        k = int(m.group(1))
        bound_max = min(max_k, chunk_count if isinstance(chunk_count, int) and chunk_count > 0 else max_k)
        return _clamp(k, min_k, bound_max)

    # 2) Fractions/percentages
    frac = _parse_fraction(q)
    if frac:
        if isinstance(chunk_count, int) and chunk_count > 0:
            target = int(round(frac * chunk_count))
        else:
            target = int(round(frac * max_k))
        bound_max = min(max_k, chunk_count if isinstance(chunk_count, int) and chunk_count > 0 else max_k)
        return _clamp(max(target, default_k), min_k, bound_max)

    # 3) Scope heuristics
    specific_words = [" just ", " only ", " exactly ", " specific ", " specifically ", " precise "]
    broad_words = [
        " overview ", " entire ", " whole ", " across ", " summary ", " summarize ",
        " introduction ", " intro ", " all sections ", " for each ", " ending ", " conclusion ", " epilogue ",
    ]

    if any(w in q for w in specific_words):
        base = min(5, max(default_k, 3))
    elif any(w in q for w in broad_words):
        base = max(default_k, 15)
    else:
        base = default_k

    bound_max = min(max_k, chunk_count if isinstance(chunk_count, int) and chunk_count > 0 else max_k)
    return _clamp(base, min_k, bound_max)


def estimate_k_ai(
    user_query: str,
    min_k: int = 3,
    max_k: int = 40,
    default_k: int = 7,
    vector_store=None,
) -> int:
    """Deprecated AI-based estimator retained for compatibility. Now a thin wrapper over estimate_k()."""
    return estimate_k(user_query=user_query, min_k=min_k, max_k=max_k, default_k=default_k)


def estimate_k_informed(
    user_query: str,
    vector_store,
    min_k: int = 3,
    max_k: int = 40,
    default_k: int = 7,
    probe_k: int = 10,
) -> Dict[str, Any]:
    """Deprecated heuristic estimator retained for compatibility. Now returns k from estimate_k()."""
    k = estimate_k(user_query=user_query, min_k=min_k, max_k=max_k, default_k=default_k)
    return {"k": k, "file_id": None, "chunk_count": None}
