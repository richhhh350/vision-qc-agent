from __future__ import annotations

import numpy as np
from scipy.ndimage import label as components
from scipy.integrate import trapezoid
from scipy.stats import beta, rankdata


def auroc(labels, scores) -> float | None:
    y, s = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    positives, negatives = int(y.sum()), int((~y).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(s)
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def binomial_upper(misses: int, total: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound; no trials means no evidence."""
    if not 0 <= misses <= total or not 0 < confidence < 1:
        raise ValueError("Invalid binomial counts or confidence")
    if total == 0 or misses == total:
        return 1.0
    return float(beta.ppf(confidence, misses + 1, total - misses))


def ece(labels, probabilities, bins: int = 10) -> float:
    y, p = np.asarray(labels), np.asarray(probabilities)
    value = 0.0
    for i in range(bins):
        selected = (p >= i / bins) & (p <= (i + 1) / bins if i == bins - 1 else p < (i + 1) / bins)
        if selected.any():
            value += selected.mean() * abs(p[selected].mean() - y[selected].mean())
    return float(value)


def routing_metrics(labels, decisions: list[dict]) -> dict:
    y = np.asarray(labels, dtype=int)
    routes = np.array([d["route"] for d in decisions])
    bad, good = y == 1, y == 0
    misses = int(np.sum(bad & (routes == "pass")))
    covered = np.array([int(label) in d.get("prediction_set", []) for label, d in zip(y, decisions)])
    return dict(n=len(y), n_defect=int(bad.sum()), missed_defects=misses,
                auto_release_fnr=float(misses / bad.sum()) if bad.any() else None,
                fnr_upper_95=binomial_upper(misses, int(bad.sum())),
                review_rate=float(np.mean(routes == "review")),
                pass_rate=float(np.mean(routes == "pass")), alarm_rate=float(np.mean(routes == "alarm")),
                false_alarm_rate=float(np.mean(routes[good] == "alarm")) if good.any() else None,
                coverage=float(covered.mean()),
                defect_coverage=float(covered[bad].mean()) if bad.any() else None)


def pixel_metrics(masks: list[np.ndarray], maps: list[np.ndarray], max_fpr: float = 0.3, steps: int = 100) -> dict:
    if not masks:
        return {"pixel_auroc": None, "aupro_0_30_approx": None}
    y, s = np.stack(masks).astype(bool), np.stack(maps)
    roc = auroc(y.ravel(), s.ravel())
    region_maps, n_regions = [], 0
    for mask in y:
        regions, count = components(mask)
        region_maps.append((regions, count))
        n_regions += count
    if not n_regions or not (~y).any():
        return {"pixel_auroc": roc, "aupro_0_30_approx": None}
    thresholds = np.r_[np.inf, np.quantile(s, np.linspace(1, 0, steps)), -np.inf]
    points = []
    for threshold in thresholds:
        pred = s >= threshold
        overlaps = []
        for prediction, (regions, count) in zip(pred, region_maps):
            if count:
                size = np.bincount(regions.ravel(), minlength=count + 1)[1:]
                hit = np.bincount(regions[prediction], minlength=count + 1)[1:]
                overlaps.extend((hit / size).tolist())
        points.append((float(pred[~y].mean()), float(np.mean(overlaps))))
    # Vertical segments at equal FPR have zero area; retain their highest endpoint.
    envelope = {}
    for fpr, pro in points:
        envelope[fpr] = max(pro, envelope.get(fpr, 0.0))
    fprs = np.array(sorted(envelope))
    pros = np.array([envelope[x] for x in fprs])
    x = np.r_[fprs[fprs < max_fpr], max_fpr]
    z = np.interp(x, fprs, pros)
    return {"pixel_auroc": roc, "aupro_0_30_approx": float(trapezoid(z, x) / max_fpr)}
