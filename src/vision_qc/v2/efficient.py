"""CPU-budget EfficientAD training using Anomalib's official networks.

The external ImageNet penalty is deliberately omitted; this is recorded in every
model and must not be described as a reproduction of the paper's training recipe.
"""
from pathlib import Path
import hashlib
import time

import numpy as np
from PIL import Image


def fit_efficientad(detector, images, config):
    import torch
    teacher = Path(config["teacher"])
    if not teacher.is_file():
        raise FileNotFoundError(f"Download the official teacher first: {teacher}")
    steps = int(config.get("steps", 300))
    if steps < 1 or len(images) < 2:
        raise ValueError("Positive training steps and at least two normal images required")
    net = detector.network
    net.teacher.load_state_dict(torch.load(teacher, map_location="cpu", weights_only=True))
    net.teacher.requires_grad_(False).eval()
    net.mean_std.requires_grad_(False)
    net.quantiles.requires_grad_(False)
    rng = np.random.default_rng(detector.config["seed"])
    order = rng.permutation(len(images))
    # Train-only holdout for map normalization; never probability/calibration/test.
    holdout_n = min(64, max(1, len(images)//10))
    holdout = [images[i] for i in order[:holdout_n]]
    fitting = [images[i] for i in order[holdout_n:]]

    def tensor(path):
        with Image.open(path) as image:
            pixels, _ = detector.geometry.prepare(image)
        return detector.tensor(pixels)

    total, square, n = 0, 0, 0
    with torch.no_grad():
        for path in fitting[:min(128, len(fitting))]:
            features = net.teacher(tensor(path)).double()
            total = total + features.sum((0, 2, 3), keepdim=True)
            square = square + features.square().sum((0, 2, 3), keepdim=True)
            n += features.shape[0]*features.shape[2]*features.shape[3]
        mean = total/n
        net.mean_std["mean"].copy_(mean.float())
        net.mean_std["std"].copy_((square/n-mean.square()).clamp_min(1e-8).sqrt().float())
    parameters = list(net.student.parameters())+list(net.ae.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.get("lr", 1e-4), weight_decay=1e-5)
    start, losses = time.perf_counter(), []
    net.train()
    net.teacher.eval()
    for step in range(steps):
        x = tensor(fitting[int(rng.integers(len(fitting)))])
        _, distance = net.compute_student_teacher_distance(x)
        hard = torch.quantile(distance.detach().flatten(), 0.999)
        loss_st = distance[distance >= hard].mean()
        augmented = net.choose_random_aug_image(x)
        ae = net.ae(augmented, x.shape[-2:])
        with torch.no_grad():
            target = (net.teacher(augmented)-net.mean_std["mean"])/net.mean_std["std"]
        student = net.student(augmented)[:, net.teacher_out_channels:]
        loss = loss_st+(ae-target).square().mean()+(ae-student).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite EfficientAD training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1) % 25 == 0:
            print(f"EfficientAD {step+1}/{steps}: loss={losses[-1]:.4f}", flush=True)
    net.eval()
    st, ae = [], []
    with torch.no_grad():
        for path in holdout:
            a, b = net.get_maps(tensor(path), normalize=False)
            st.append(a.flatten()[::16]); ae.append(b.flatten()[::16])
        for kind, values in (("st", st), ("ae", ae)):
            low, high = torch.quantile(torch.cat(values), torch.tensor([.9, .995]))
            net.quantiles[f"qa_{kind}"].copy_(low)
            net.quantiles[f"qb_{kind}"].copy_(torch.maximum(high, low+1e-6))
    detector.training_record = dict(steps=steps, train_images=len(fitting), normalization_images=holdout_n,
        teacher_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(),
        teacher_source="https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip",
        imagenet_penalty=False, recipe="CPU-budget ablation, not paper reproduction",
        final_loss=losses[-1], train_seconds=time.perf_counter()-start)
    return detector
