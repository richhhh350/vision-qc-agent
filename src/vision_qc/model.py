from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def coreset_indices(features: np.ndarray, budget: int, seed: int = 42) -> np.ndarray:
    if len(features) == 0 or budget <= 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(features)))]
    distance = np.full(len(features), np.inf)
    for _ in range(min(budget, len(features)) - 1):
        delta = features - features[chosen[-1]]
        distance = np.minimum(distance, np.einsum("ij,ij->i", delta, delta))
        distance[chosen] = -np.inf
        chosen.append(int(np.argmax(distance)))
    return np.array(chosen)


def patch_features(image: Image.Image, size: int = 96, patch: int = 8) -> np.ndarray:
    x = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32) / 255
    grid = size // patch
    blocks = x.reshape(grid, patch, grid, patch, 3).transpose(0, 2, 1, 3, 4)
    mean = blocks.mean(axis=(2, 3))
    std = blocks.std(axis=(2, 3))
    dx = np.abs(np.diff(blocks, axis=3)).mean(axis=(2, 3))
    dy = np.abs(np.diff(blocks, axis=2)).mean(axis=(2, 3))
    return np.concatenate([mean, std, dx, dy], axis=-1).reshape(-1, 12)


@dataclass
class Detection:
    score: float
    heatmap: np.ndarray
    embedding: np.ndarray


class PatchMemory:
    """Small CPU baseline inspired by patch memory methods, NOT the published PatchCore model.

    Uses local RGB/texture statistics and a greedy k-center memory bank. No pretrained weights.
    """
    backend = "patch_memory_cpu"

    def __init__(self, size: int = 96, patch: int = 8, bank_size: int = 256, seed: int = 42):
        if size <= 0 or patch < 2 or size % patch or bank_size <= 0:
            raise ValueError("Invalid image size, patch size or memory budget")
        self.size, self.patch, self.bank_size, self.seed = size, patch, bank_size, seed

    def fit(self, images: list[Path | Image.Image]) -> "PatchMemory":
        if not images:
            raise ValueError("Normal training images are required")
        features = []
        for image in images:
            if isinstance(image, Path):
                with Image.open(image) as opened:
                    features.append(patch_features(opened, self.size, self.patch))
            else:
                features.append(patch_features(image, self.size, self.patch))
        x = np.concatenate(features)
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 0.02)
        x = (x - self.mean) / self.scale
        if len(x) > 6000:
            x = x[np.random.default_rng(self.seed).choice(len(x), 6000, replace=False)]
        self.bank = x[coreset_indices(x, self.bank_size, self.seed)].astype(np.float32)
        self.tree = cKDTree(self.bank)
        return self

    def predict(self, image: Image.Image) -> Detection:
        features = (patch_features(image, self.size, self.patch) - self.mean) / self.scale
        distances = self.tree.query(features, k=1)[0].astype(np.float32)
        grid = self.size // self.patch
        heatmap = np.asarray(Image.fromarray(distances.reshape(grid, grid)).resize((self.size, self.size), Image.Resampling.BILINEAR))
        return Detection(float(distances.max()), heatmap, np.concatenate([features.mean(0), features.std(0)]))

    def save(self, path: Path) -> None:
        np.savez_compressed(path, mean=self.mean, scale=self.scale, bank=self.bank,
                            size=self.size, patch=self.patch, seed=self.seed)

    @classmethod
    def load(cls, path: Path) -> "PatchMemory":
        with np.load(path, allow_pickle=False) as data:
            obj = cls(int(data["size"]), int(data["patch"]), len(data["bank"]), int(data["seed"]))
            obj.mean, obj.scale, obj.bank = data["mean"], data["scale"], data["bank"]
        obj.tree = cKDTree(obj.bank)
        return obj
