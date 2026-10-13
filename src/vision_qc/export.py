from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from .model import PatchMemory
from .pipeline import dump_json


def torch_module(model: PatchMemory):
    import torch
    from torch import nn

    class TorchPatchMemory(nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("mean", "scale", "bank"):
                self.register_buffer(name, torch.from_numpy(getattr(model, name)))

        def forward(self, image):
            grid, patch = model.size // model.patch, model.patch
            blocks = image.permute(0, 2, 3, 1).reshape(-1, grid, patch, grid, patch, 3).permute(0, 1, 3, 2, 4, 5)
            mean = blocks.mean(dim=(3, 4))
            variance = ((blocks - mean[:, :, :, None, None, :]) ** 2).mean(dim=(3, 4))
            dx = (blocks[:, :, :, :, 1:, :] - blocks[:, :, :, :, :-1, :]).abs().mean(dim=(3, 4))
            dy = (blocks[:, :, :, 1:, :, :] - blocks[:, :, :, :-1, :, :]).abs().mean(dim=(3, 4))
            features = torch.cat([mean, variance.sqrt(), dx, dy], dim=-1).reshape(-1, grid * grid, 12)
            features = (features - self.mean) / self.scale
            distance = ((features[:, :, None, :] - self.bank[None, None, :, :]) ** 2).sum(-1).min(-1).values.sqrt()
            return distance.max(-1).values, distance.reshape(-1, grid, grid)

    return TorchPatchMemory().eval()


def export_onnx(run: Path, image_path: Path) -> dict:
    import onnx
    import onnxruntime as ort
    import torch

    model = PatchMemory.load(run / "model.npz")
    module = torch_module(model)
    torch.set_num_threads(1)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    data = np.asarray(image.resize((model.size, model.size)), dtype=np.float32).transpose(2, 0, 1)[None] / 255
    tensor = torch.from_numpy(data.copy())
    path = run / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(module, tensor, str(path), input_names=["image"], output_names=["score", "patch_map"],
                          dynamic_axes={"image": {0: "batch"}, "score": {0: "batch"}, "patch_map": {0: "batch"}},
                          opset_version=17, dynamo=False)
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = module(tensor)
    actual = session.run(None, {"image": data})
    for reference, value in zip(expected, actual):
        np.testing.assert_allclose(reference.numpy(), value, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(model.predict(image).score, actual[0][0], rtol=2e-4, atol=2e-4)
    measured = {}
    for name in ("pytorch", "onnxruntime"):
        times = []
        with torch.inference_mode():
            for index in range(60):
                start = time.perf_counter()
                module(tensor) if name == "pytorch" else session.run(None, {"image": data})
                if index >= 10:
                    times.append((time.perf_counter() - start) * 1000)
        measured[name] = dict(p50_ms=float(np.percentile(times, 50)), p95_ms=float(np.percentile(times, 95)))
    result = dict(path=str(path), backend="CPU patch-memory equivalent graph", parity="passed", batch_size=1,
                  timing_scope="preprocessed NCHW float32 input, feature extraction and nearest neighbor; excludes resize/routing",
                  onnx_bytes=path.stat().st_size, max_map_abs_error=float(np.abs(expected[1].numpy() - actual[1]).max()),
                  measured=measured, device="cpu", threads=1, gpu_memory_bytes=None)
    dump_json(run / "onnx-benchmark.json", result)
    return result
