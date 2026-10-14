from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import threading
import time
import warnings
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from .active import select_batch
from .pipeline import Inspector, overlay
from .store import ReviewStore


class Feedback(BaseModel):
    label: Literal[0, 1]
    reviewer: str = Field(min_length=1, max_length=100)
    note: str = Field(default="", max_length=2000)
    revision: int = Field(ge=0)


def create_app(run: Path | None = None, state: Path | None = None) -> FastAPI:
    if run is None and os.getenv("QC_REGISTRY_DIR"):
        from .v2.registry import Registry
        run = Registry(os.environ["QC_REGISTRY_DIR"]).active_run()
    run = run or Path(os.getenv("QC_RUN_DIR", "runs/demo"))
    state = (state or Path(os.getenv("QC_STATE_DIR", "runs/state"))).resolve()
    state.mkdir(parents=True, exist_ok=True)
    store = ReviewStore(state / "reviews.sqlite")
    inspector = Inspector(run) if (run / "model.json").exists() else None
    lock = threading.Lock()
    key = os.getenv("QC_API_KEY", "")
    jobs = None
    if inspector and inspector.meta.get("schema_version") == 2:
        from .v2.jobs import TrainingJobs
        jobs = TrainingJobs(run,state)

    def auth(x_api_key: str | None = Header(default=None)):
        if key and not secrets.compare_digest(x_api_key or "", key):
            raise HTTPException(401, "Invalid API key")

    from vision_qc import __version__
    app = FastAPI(title="视觉质检 Agent", version=__version__, dependencies=[Depends(auth)])

    def ready() -> Inspector:
        if inspector is None:
            raise HTTPException(503, "No model. Run: vision-qc demo")
        return inspector

    @app.get("/health")
    def health():
        return {"status": "ready" if inspector else "untrained", "version": inspector.version if inspector else None}

    @app.get("/model")
    def model_info():
        model = ready()
        if model.meta.get("schema_version") == 2:
            from .v2.evidence import check_certificate
            approved,reason = check_certificate(model.meta)
            model.meta["release_gate"].update(approved=approved,reason=reason)
        return dict(run_dir=str(run.resolve()), state_dir=str(state),
                    schema_version=model.meta.get("schema_version",1),
                    secondary=model.meta.get("secondary"),certificate=model.meta.get("certificate"),
                    validation_reference_only=bool(model.meta.get("parent_version") and not model.meta.get("certificate")),
                    **{k: model.meta[k] for k in ("version", "category", "backend", "synthetic", "release_gate", "counts")})

    @app.post("/inspect")
    def inspect(file: UploadFile = File(...), category: str = Form(...)):
        decode_started=time.perf_counter()
        model = ready()
        if not 1 <= len(category) <= 100:
            raise HTTPException(422, "Invalid category")
        raw = file.file.read(10 * 1024 * 1024 + 1)
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(413, "Image exceeds 10 MB")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as opened:
                    if opened.width * opened.height > 16_000_000:
                        raise HTTPException(413, "Image exceeds 16 megapixels")
                    image = opened.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning, ValueError):
            raise HTTPException(422, "Unsupported or damaged image") from None
        decode_ms=(time.perf_counter()-decode_started)*1000
        image_hash = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
        inspection_id = hashlib.sha256(f"{image_hash}:{model.version}:{category}".encode()).hexdigest()[:24]
        with lock:
            previous = None
            try:
                previous = store.get(inspection_id)
                if model.meta.get("schema_version") != 2:
                    return public(previous)
            except KeyError:
                pass
            try:
                result, heatmap = model.inspect(image, category)
            except ValueError as exc:
                raise HTTPException(422,str(exc)) from None
            # A deterministic 5% audit stream limits feedback bias on automatic decisions.
            if model.meta.get("schema_version") != 2 and result["route"] != "review" and int(image_hash[:8], 16) % 20 == 0:
                result.update(route="review", reason="random_audit")
            storage_started=time.perf_counter()
            upload = state / "images"
            upload.mkdir(exist_ok=True)
            image_path = upload / f"{inspection_id}.png"
            image.save(image_path)
            overlay(image, heatmap, model.meta["pixel_threshold"], result["boxes"]).save(upload / f"{inspection_id}-overlay.png")
            row = store.refresh_result(inspection_id,result) if previous else store.add(inspection_id, str(image_path), result)
            row["result"].setdefault("stage_timing",{}).update(decode_ms=decode_ms,storage_ms=(time.perf_counter()-storage_started)*1000)
        return public(row)

    def public(row):
        return {k: v for k, v in row.items() if k != "image_path"}

    @app.get("/reviews")
    def reviews(status: Literal["pending", "auto", "labeled"] | None = None, limit: int = Query(100, ge=1, le=1000)):
        return [public(row) for row in store.list(status, limit)]

    @app.get("/reviews/{inspection_id}/image")
    def review_image(inspection_id: str, heatmap: bool = False):
        try:
            row = store.get(inspection_id)
        except KeyError:
            raise HTTPException(404, "Inspection not found") from None
        path = Path(row["image_path"])
        if heatmap:
            path = path.with_name(path.stem + "-overlay.png")
        if not path.is_file():
            raise HTTPException(404, "Stored image not found")
        return FileResponse(path, media_type="image/png")

    @app.post("/reviews/{inspection_id}/feedback")
    def feedback(inspection_id: str, body: Feedback):
        try:
            return public(store.feedback(inspection_id, body.label, body.reviewer, body.note, body.revision))
        except KeyError:
            raise HTTPException(404, "Inspection not found") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/reviews/{inspection_id}/history")
    def history(inspection_id: str):
        try:
            return store.history(inspection_id)
        except KeyError:
            raise HTTPException(404, "Inspection not found") from None

    @app.post("/reviews/{inspection_id}/mask")
    def save_mask(inspection_id: str, file: UploadFile = File(...), revision: int = Form(...),
                  annotation_seconds: float | None = Form(default=None)):
        if annotation_seconds is not None and (not 0 <= annotation_seconds <= 86400):
            raise HTTPException(422,"Invalid annotation duration")
        try:
            row = store.get(inspection_id)
            raw = file.file.read(10*1024*1024+1)
            if len(raw)>10*1024*1024:
                raise ValueError("Mask exceeds 10 MB")
            with warnings.catch_warnings():
                warnings.simplefilter("error",Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as opened, Image.open(row["image_path"]) as original:
                    if opened.size != original.size:
                        raise ValueError("Mask must have original image dimensions")
                    mask = opened.convert("L").point(lambda v:255 if v>0 else 0)
                    if not mask.getbbox():
                        raise ValueError("Defect mask must contain foreground pixels")
            directory = state/"masks"
            directory.mkdir(exist_ok=True)
            digest = hashlib.sha256(mask.tobytes()).hexdigest()[:16]
            path = directory/f"{inspection_id}-{revision}-{digest}.png"
            mask.save(path)
            store.save_mask(inspection_id,path,revision,annotation_seconds)
            return dict(id=inspection_id,revision=revision,mask_saved=True)
        except KeyError:
            raise HTTPException(404,"Inspection not found") from None
        except (ValueError,UnidentifiedImageError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:
            raise HTTPException(422,str(exc)) from None

    @app.post("/training-jobs")
    def start_training(steps: int = Query(120,ge=1,le=10000)):
        if jobs is None:
            raise HTTPException(409,"Background candidate training requires a v2 model")
        try:
            return jobs.start(steps)
        except ValueError as exc:
            raise HTTPException(409,str(exc)) from None

    @app.get("/training-jobs/{job_id}")
    def training_status(job_id: str):
        try:
            if jobs is None:
                raise KeyError(job_id)
            return jobs.get(job_id)
        except KeyError:
            raise HTTPException(404,"Training job not found") from None

    @app.post("/training-jobs/{job_id}/cancel")
    def cancel_training(job_id: str):
        try:
            if jobs is None:
                raise KeyError(job_id)
            return jobs.cancel(job_id)
        except KeyError:
            raise HTTPException(404,"Training job not found") from None

    @app.get("/active-learning")
    def active_learning(strategy: Literal["random", "uncertainty", "coreset", "hybrid"] = "uncertainty", budget: int = Query(10, ge=1, le=100)):
        version = ready().version
        rows = [r for r in store.list("pending", 10000) if r["result"]["model_version"] == version
                and r["result"]["category"] == ready().meta["category"]]
        items = [dict(id=r["id"], **r["result"]) for r in rows]
        if strategy == "hybrid":
            # Pure acquisition has no torch dependency; live results contain no oracle labels.
            ranked = sorted(items,key=lambda x:-min(x["p_normal"],x["p_defect"]))
            selected = select_batch(ranked[:max(budget,len(ranked)//2)],budget,"coreset")
            return {"strategy":strategy,"eligible":len(items),"ids":selected}
        return {"strategy": strategy, "eligible": len(items), "ids": select_batch(items, budget, strategy)}

    @app.get("/stats")
    def stats():
        return store.summary()

    @app.get("/reports")
    def reports():
        values = []
        # Only aggregate metric files in sibling local runs, never arbitrary client paths.
        report_root=run.resolve().parent
        paths=set(report_root.glob("*/metrics*.json")) | set(report_root.glob("*/*/metrics*.json"))
        for path in sorted(paths)[:100]:
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                values.append({"name": path.relative_to(report_root).as_posix(), "report": report})
            except (ValueError, OSError):
                continue
        return sorted(values, key=lambda item: (item["report"].get("synthetic", False), item["name"]))

    @app.get("/metrics")
    def prometheus():
        values = store.summary()
        body = "# TYPE vision_qc_inspections gauge\n" + "".join(
            f'vision_qc_inspections{{status="{status}"}} {values.get(status, 0)}\n' for status in ("auto", "pending", "labeled"))
        return Response(body, media_type="text/plain")

    return app
