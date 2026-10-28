from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from ..model import Detection, PatchMemory
from .geometry import Geometry


class Detector(Protocol):
    backend: str
    def fit(self, images: list[Path]): ...
    def predict(self, image: Image.Image) -> Detection: ...
    def save(self, directory: Path): ...
    def metadata(self) -> dict: ...


class LegacyDetector:
    backend = "patch_memory_cpu"
    def __init__(self, **config):
        self.model = PatchMemory(**config)
    def fit(self, images):
        self.model.fit(images)
        return self
    def predict(self, image):
        result = self.model.predict(image)
        heatmap = np.asarray(Image.fromarray(result.heatmap).resize(image.size, Image.Resampling.BILINEAR))
        return Detection(result.score, heatmap, result.embedding)
    def metadata(self):
        return dict(backend=self.backend, size=self.model.size, patch=self.model.patch,
                    bank_size=self.model.bank_size, seed=self.model.seed)
    def save(self, directory):
        self.model.save(directory / "detector.npz")
        return ["detector.npz"]


class DeepDetector:
    """Official Anomalib torch models with explicit CPU preprocessing and geometry.

    Fitting is managed here to avoid allowing framework defaults to split the official test.
    """
    def __init__(self, backend="patchcore", geometry=None, backbone="resnet18", bank_size=512,
                 max_features=12000, seed=42, threads=4, pretrained=True, **extra):
        if threads < 1 or bank_size < 9 or max_features < bank_size:
            raise ValueError("Require threads >=1, max_features >= bank_size >=9")
        cache = Path("data/pretrained/huggingface").resolve()
        if cache.is_dir():
            os.environ.setdefault("HF_HOME", str(cache))
        import torch
        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        self.backend, self.geometry = backend, Geometry(**(geometry or {}))
        self.config = dict(backend=backend, geometry=self.geometry.to_dict(), backbone=backbone,
                           bank_size=bank_size, max_features=max_features, seed=seed, threads=threads)
        self._embedding = None
        if backend == "patchcore":
            from anomalib.models.image.patchcore.torch_model import PatchcoreModel
            self.network = PatchcoreModel(layers=["layer2", "layer3"], backbone=backbone,
                                         pre_trained=pretrained, num_neighbors=9)
            self.hook = self.network.feature_extractor.register_forward_hook(self._capture)
        elif backend == "efficientad":
            if self.geometry.size != 256:
                raise ValueError("EfficientAD CPU contract requires size=256")
            from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel
            self.network = EfficientAdModel(teacher_out_channels=384)
            self.hook = self.network.teacher.register_forward_hook(self._capture)
        else:
            raise ValueError("Unknown deep detector")
        self.network.cpu()

    def _capture(self, module, inputs, outputs):
        import torch
        values = list(outputs.values()) if isinstance(outputs, dict) else [outputs]
        self._embedding = torch.cat([v.detach().mean(dim=(-2, -1)) for v in values], dim=1)

    def tensor(self, pixels):
        import torch
        x = torch.from_numpy(pixels)
        if self.backend == "patchcore":
            x = (x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return x

    def fit(self, images):
        if self.backend != "patchcore":
            raise ValueError("EfficientAD requires fit_efficientad with a pretrained teacher")
        import torch
        if not images:
            raise ValueError("Normal training images required")
        rng = np.random.default_rng(self.config["seed"])
        self.network.train()
        self.network.feature_extractor.eval()
        chunks = []
        per_image = max(1, self.config["max_features"] // len(images))
        with torch.inference_mode():
            for i, path in enumerate(images):
                with Image.open(path) as image:
                    pixels, _ = self.geometry.prepare(image)
                embedding = self.network(self.tensor(pixels)).detach().cpu()
                self.network.embedding_store.clear()
                indices = rng.choice(len(embedding), min(per_image, len(embedding)), replace=False)
                chunks.append(embedding[indices].clone())
                if (i+1) % 100 == 0:
                    print(f"PatchCore features {i+1}/{len(images)}", flush=True)
            features = torch.cat(chunks)
        # Clone out of inference mode; Anomalib projection and k-center use ordinary tensors.
        features = features.clone()
        if len(features) > self.config["max_features"]:
            features = features[rng.choice(len(features), self.config["max_features"], replace=False)]
        if len(features) < 9:
            raise ValueError("PatchCore requires at least 9 candidate patches")
        ratio = min(1.0, self.config["bank_size"] / len(features))
        self.network.embedding_store = [features]
        self.network.subsample_embedding(sampling_ratio=ratio)
        self.network.eval()
        return self

    def predict(self, image):
        import torch
        start = time.perf_counter()
        pixels, records = self.geometry.prepare(image)
        prepared = time.perf_counter()
        self.network.eval()
        with torch.inference_mode():
            output = self.network(self.tensor(pixels))
            maps = output.anomaly_map.detach().cpu().numpy()
            score = float(output.pred_score.max())
            embedding = self._embedding.mean(0).cpu().numpy().astype(np.float32)
        inferred = time.perf_counter()
        heatmap = self.geometry.restore(maps, records, image.size)
        self.last_timing = dict(preprocess_ms=(prepared-start)*1000, detector_ms=(inferred-prepared)*1000,
                                restore_ms=(time.perf_counter()-inferred)*1000, tiles=len(records))
        if not np.isfinite(score) or not np.isfinite(embedding).all():
            raise ValueError("Non-finite detector output")
        return Detection(score, heatmap, embedding)

    def metadata(self):
        return dict(self.config, implementation="anomalib 2.2.0 torch model",
                    score_aggregation="max official per-tile image score", map_aggregation="mean overlapping pixels",
                    embedding="mean pooled frozen feature maps", input_color="RGB", native_coordinates=True,
                    training=getattr(self, "training_record", None))

    def save(self, directory):
        import torch
        torch.save(self.network.state_dict(), directory / "detector.pt")
        return ["detector.pt"]


def load_detector(run: Path, config):
    backend = config["backend"]
    if backend == "patch_memory_cpu":
        detector = LegacyDetector()
        detector.model = PatchMemory.load(run / "detector.npz")
    else:
        import torch
        detector = DeepDetector(**config, pretrained=False)
        state = torch.load(run / "detector.pt", map_location="cpu", weights_only=True)
        # Anomalib's dynamic memory-bank loader handles the uninitialized buffer shape.
        detector.network.load_state_dict(state)
        detector.network.eval()
    return detector
