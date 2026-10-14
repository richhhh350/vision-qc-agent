from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Geometry:
    size: int = 256
    mode: str = "letterbox"
    overlap: float = 0.25
    max_tiles: int = 64

    def __post_init__(self):
        if self.size < 32 or self.size % 16 or self.mode not in {"letterbox", "tiles"}:
            raise ValueError("Geometry needs a multiple-of-16 size >=32 and letterbox/tiles mode")
        if not 0 <= self.overlap < 1 or self.max_tiles < 1:
            raise ValueError("Invalid tile overlap or limit")

    def to_dict(self):
        return asdict(self)

    def prepare(self, image: Image.Image):
        image = image.convert("RGB")
        w, h = image.size
        def starts(length):
            stride = max(1, round(self.size * (1 - self.overlap)))
            return sorted(set(range(0, max(1, length - self.size + 1), stride)) | {max(0, length - self.size)})
        bounds = [(0, 0, w, h)] if self.mode == "letterbox" else [
            (x, y, min(x + self.size, w), min(y + self.size, h)) for y in starts(h) for x in starts(w)]
        if len(bounds) > self.max_tiles:
            raise ValueError(f"Image needs {len(bounds)} tiles, limit is {self.max_tiles}")
        tensors, records = [], []
        for box in bounds:
            crop = image.crop(box)
            ratio = self.size / max(crop.size)
            rw, rh = max(1, round(crop.width * ratio)), max(1, round(crop.height * ratio))
            pixels = np.asarray(crop.resize((rw, rh), Image.Resampling.BILINEAR))
            left, top = (self.size - rw) // 2, (self.size - rh) // 2
            padded = np.pad(pixels, ((top, self.size-rh-top), (left, self.size-rw-left), (0, 0)), mode="edge")
            tensors.append(padded.transpose(2, 0, 1).astype(np.float32) / 255)
            records.append((box, (left, top, left+rw, top+rh)))
        return np.stack(tensors), records

    def restore(self, maps: np.ndarray, records, image_size):
        w, h = image_size
        total, count = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
        if len(maps) != len(records):
            raise ValueError("Tile prediction count mismatch")
        for values, (box, valid) in zip(maps, records):
            values = np.asarray(values).squeeze()
            if values.ndim != 2 or not np.isfinite(values).all():
                raise ValueError("Expected finite 2D heatmap")
            restored = Image.fromarray(values.astype(np.float32)).resize((self.size, self.size), Image.Resampling.BILINEAR)
            x0, y0, x1, y1 = box
            region = np.asarray(restored.crop(valid).resize((x1-x0, y1-y0), Image.Resampling.BILINEAR))
            total[y0:y1, x0:x1] += region
            count[y0:y1, x0:x1] += 1
        if np.any(count == 0):
            raise ValueError("Uncovered image pixels")
        return total / count
