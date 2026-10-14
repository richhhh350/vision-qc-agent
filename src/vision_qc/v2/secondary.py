"""Small supervised U-Net; its predictions are advice, never release evidence."""
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .geometry import Geometry


class TinyUNet(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1), nn.ReLU(), nn.Conv2d(b,b,3,padding=1), nn.ReLU())
        self.down1 = block(3,width)
        self.down2 = block(width,width*2)
        self.center = block(width*2,width*4)
        self.up2 = block(width*6,width*2)
        self.up1 = block(width*3,width)
        self.out = nn.Conv2d(width,1,1)

    def forward(self, x):
        a = self.down1(x)
        b = self.down2(F.max_pool2d(a,2))
        c = self.center(F.max_pool2d(b,2))
        d = self.up2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode="bilinear",align_corners=False),b],1))
        e = self.up1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode="bilinear",align_corners=False),a],1))
        return self.out(e)


class Segmenter:
    def __init__(self, seed=42, size=128, width=8, threads=4):
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        self.config = dict(seed=seed,size=size,width=width,threads=threads)
        self.geometry = Geometry(size=size,mode="tiles",max_tiles=256)
        self.network = TinyUNet(width)
        self.training_record = {}

    def fit(self, samples, steps=60, lr=1e-3, cancel=None):
        if not samples or steps < 1:
            raise ValueError("Labeled samples and positive steps required")
        if any(s.split not in {"train", "pool"} or s.label not in (0,1) for s in samples):
            raise ValueError("Supervised training may only use revealed train/pool labels")
        if any(s.label == 1 and s.mask is None for s in samples):
            raise ValueError("Defect pixel masks required; image labels cannot invent segmentation truth")
        rng = np.random.default_rng(self.config["seed"])
        cached = []
        for s in samples:
            with Image.open(s.path) as image:
                x, records = self.geometry.prepare(image)
                shape = (image.height,image.width)
            if s.mask:
                with Image.open(s.mask) as opened:
                    if opened.size != (shape[1],shape[0]):
                        raise ValueError("Mask and image dimensions differ")
                    mask = opened.convert("L")
            else:
                mask = Image.new("L",(shape[1],shape[0]))
            masks = []
            for box, valid in records:
                l,t,r,b = valid
                crop = np.asarray(mask.crop(box).resize((r-l,b-t),Image.Resampling.NEAREST)) > 0
                padded = np.pad(crop,((t,self.geometry.size-b),(l,self.geometry.size-r)))
                masks.append(padded[None])
            # Byte cache bounds memory; expand only selected training tiles.
            cached.append((np.uint8(np.rint(x*255)),np.asarray(masks,dtype=np.float32)))
            if s.label==1 and not np.any(masks):
                raise ValueError("A defect label requires a nonempty pixel mask")
        positive = [i for i,(_,y) in enumerate(cached) if np.any(y)]
        normal = [i for i,s in enumerate(samples) if s.label==0]
        optimizer = torch.optim.Adam(self.network.parameters(),lr=lr)
        self.network.train()
        start, loss_value, positive_tiles = time.perf_counter(), None, 0
        for step in range(steps):
            if cancel and cancel():
                raise InterruptedError("Training cancelled")
            # These masks have ALREADY been annotated. Balance supervised training
            # so tiny defects are not lost through uniform whole-image tile sampling.
            use_positive = bool(positive) and step%2==0
            candidates = positive if use_positive else normal or list(range(len(cached)))
            x,y = cached[int(rng.choice(candidates))]
            tiles = np.flatnonzero(y.reshape(len(y),-1).any(1)) if use_positive else np.arange(len(x))
            tile = int(rng.choice(tiles))
            x = torch.from_numpy(x[tile:tile+1].astype(np.float32)/255)
            y = torch.from_numpy(y[tile:tile+1])
            positive_tiles += int(y.any())
            logits = self.network(x)
            bce = F.binary_cross_entropy_with_logits(logits,y,pos_weight=logits.new_tensor(20.0))
            p = logits.sigmoid()
            dice = 1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)
            loss = bce+dice
            if not torch.isfinite(loss):
                raise ValueError("Non-finite segmentation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())
        self.network.eval()
        self.training_record = dict(steps=steps,images=len(samples),defects=sum(s.label==1 for s in samples),
            defect_masks=sum(s.label==1 and s.mask is not None for s in samples),
            seconds=time.perf_counter()-start,final_loss=loss_value,positive_tiles_seen=positive_tiles,
            sampling="alternate known defect tiles and normal tiles; no unrevealed pool masks")
        return self

    def heatmap(self, image):
        x, records = self.geometry.prepare(image)
        self.network.eval()
        maps = []
        with torch.inference_mode():
            for offset in range(0,len(x),8):
                maps.append(self.network(torch.from_numpy(x[offset:offset+8])).sigmoid().numpy())
        return self.geometry.restore(np.concatenate(maps),records,image.size)

    def predict(self, image):
        heatmap = self.heatmap(image)
        # Ranking score only; no calibrated probability or automatic decision.
        k = max(1,int(heatmap.size*.001))
        score = float(np.partition(heatmap.ravel(),-k)[-k:].mean())
        return dict(score=score,role="human_advice_only",score_kind="uncalibrated_top_0.1_percent",
                    area_fraction_above_half=float(np.mean(heatmap>.5)))

    def save(self, path):
        torch.save(dict(config=self.config,state=self.network.state_dict(),training=self.training_record),path)

    @classmethod
    def load(cls, path):
        data = torch.load(Path(path),map_location="cpu",weights_only=True)
        result = cls(**data["config"])
        result.network.load_state_dict(data["state"])
        result.training_record = data["training"]
        result.network.eval()
        return result
