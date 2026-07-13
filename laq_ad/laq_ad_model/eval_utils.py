"""Shared evaluation helpers for LAQ-AD.

Metrics:
    NMI  — Normalized Mutual Information between codebook assignment and
            ground-truth maneuver label. Range [0, 1]; higher = better alignment.
    Perplexity — exp(H(code_distribution)). Max = codebook_size; higher = more uniform.
    Per-code maneuver table — shows which maneuver(s) each code specializes in.
"""

from collections import Counter, defaultdict
from math import log, exp

from .data import MANEUVERS


def normalized_mutual_information(codes: list[int], labels: list[str]) -> float:
    """NMI between discrete code assignments and string maneuver labels.

    NMI(X; Y) = 2 * I(X; Y) / (H(X) + H(Y))
    """
    assert len(codes) == len(labels)
    n = len(codes)
    if n == 0:
        return 0.0

    code_counts = Counter(codes)
    label_counts = Counter(labels)
    joint_counts: dict[tuple, int] = Counter(zip(codes, labels))

    def entropy(counts: Counter) -> float:
        total = sum(counts.values())
        return -sum((c / total) * log(c / total) for c in counts.values() if c > 0)

    h_codes = entropy(code_counts)
    h_labels = entropy(label_counts)

    if h_codes == 0 or h_labels == 0:
        return 0.0

    mutual_info = sum(
        (cnt / n) * log((cnt / n) / ((code_counts[c] / n) * (label_counts[l] / n)))
        for (c, l), cnt in joint_counts.items()
        if cnt > 0
    )
    return 2.0 * mutual_info / (h_codes + h_labels)


def print_summary(name: str, codes: list[int], labels: list[str], codebook_size: int):
    """Print perplexity, NMI, and per-code maneuver distribution table."""
    n = len(codes)
    usage = Counter(codes)
    probs = [c / n for c in usage.values()]
    perplexity = exp(-sum(p * log(p) for p in probs if p > 0))
    nmi = normalized_mutual_information(codes, labels)

    print(f"\n── {name} ──────────────────────────────────────────────")
    print(f"  samples:    {n}")
    print(f"  perplexity: {perplexity:.3f}  (max = {codebook_size})")
    print(f"  maneuver NMI: {nmi:.4f}")
    print(f"  code usage: {dict(sorted(usage.items()))}")

    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c, l in zip(codes, labels):
        counts[c][l] += 1

    header = f"  {'Code':>4} | " + " | ".join(f"{m:>11}" for m in MANEUVERS) + " | total"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for code in range(codebook_size):
        row = counts[code]
        total = sum(row.values())
        cells = " | ".join(f"{row[m]:>11}" for m in MANEUVERS)
        print(f"  {code:>4} | {cells} | {total}")

    return {"perplexity": perplexity, "nmi": nmi}
