from __future__ import annotations

import numpy as np
from scipy.ndimage import label
from scipy.integrate import trapezoid


class NativePixelMetrics:
    """Streaming native-resolution metrics; histogram approximation explicitly reported."""
    def __init__(self, bins=8192):
        self.bins = bins
        self.good = np.zeros(bins, np.int64)
        self.bad = np.zeros(bins, np.int64)
        self.regions = np.zeros(bins, np.float64)
        self.n_regions = self.images = 0
        self.areas = {"1-16": [0, 0], "17-64": [0, 0], "65-256": [0, 0], ">256": [0, 0]}

    def update(self, mask, heatmap, threshold):
        mask, heatmap = np.asarray(mask, bool), np.asarray(heatmap, float)
        if mask.shape != heatmap.shape or not np.isfinite(heatmap).all():
            raise ValueError("Native mask/map shape or finiteness mismatch")
        indexes = np.clip(((np.arctan(heatmap)/np.pi+0.5)*self.bins).astype(int), 0, self.bins-1)
        self.good += np.bincount(indexes[~mask], minlength=self.bins)
        self.bad += np.bincount(indexes[mask], minlength=self.bins)
        components, count = label(mask)
        for i in range(1, count+1):
            selected = components == i
            area = int(selected.sum())
            self.regions += np.bincount(indexes[selected], minlength=self.bins) / area
            bucket = "1-16" if area <= 16 else "17-64" if area <= 64 else "65-256" if area <= 256 else ">256"
            self.areas[bucket][0] += int(np.any(heatmap[selected] > threshold))
            self.areas[bucket][1] += 1
        self.n_regions += count
        self.images += 1

    def result(self):
        n0, n1 = int(self.good.sum()), int(self.bad.sum())
        auc = error = pro = None
        if n0 and n1:
            below = np.cumsum(self.good) - self.good
            auc = float(np.dot(self.bad.astype(float), below+0.5*self.good) / (n0*n1))
            error = float(0.5*np.dot(self.bad.astype(float), self.good) / (n0*n1))
        if n0 and self.n_regions:
            fpr = np.r_[0, np.cumsum(self.good[::-1])/n0]
            recall = np.r_[0, np.cumsum(self.regions[::-1])/self.n_regions]
            envelope = {}
            for x, y in zip(fpr, recall):
                envelope[float(x)] = max(float(y), envelope.get(float(x), 0))
            x = np.array(sorted(envelope))
            y = np.array([envelope[v] for v in x])
            grid = np.r_[x[x < 0.3], 0.3]
            pro = float(np.clip(trapezoid(np.interp(grid, x, y), grid)/0.3,0,1))
        return dict(pixel_auroc_binned=auc, pixel_auroc_max_binning_error=error,
                    aupro_0_30_binned=pro, pixel_evaluation_images=self.images,
                    normal_pixels=n0, defect_pixels=n1, region_count=self.n_regions,
                    region_hit_at_train_threshold={k: dict(hits=v[0], total=v[1], recall=v[0]/v[1] if v[1] else None) for k,v in self.areas.items()},
                    pixel_protocol=f"native masks, restored maps; {self.bins} atan bins; 4-connected regions; approximate AUPRO")


def prevalence_cost(labels, predictions):
    labels = np.asarray(labels)
    review = np.array([p["route"] == "review" for p in predictions])
    if not np.any(labels == 0) or not np.any(labels == 1):
        return []
    r0, r1 = float(review[labels == 0].mean()), float(review[labels == 1].mean())
    return [dict(defect_prevalence=p, review_rate=(1-p)*r0+p*r1) for p in (0.001, 0.01, 0.05)]
