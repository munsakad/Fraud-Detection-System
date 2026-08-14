import json

from fastapi import APIRouter, HTTPException

from .. import config

router = APIRouter(prefix="/api/models", tags=["models"])


def _load_report(filename: str):
    path = config.REPORTS_DIR / filename
    if not path.exists():
        raise HTTPException(
            400,
            f"{filename} not found in {config.REPORTS_DIR}. "
            "Run `python train_models.py` inside ml/ first.",
        )
    with open(path) as f:
        return json.load(f)


@router.get("/metrics")
async def metrics():
    return _load_report("metrics.json")


@router.get("/roc-curves")
async def roc_curves():
    return _load_report("roc_curves.json")


@router.get("/pr-curves")
async def pr_curves():
    return _load_report("pr_curves.json")


@router.get("/feature-importance")
async def feature_importance():
    return _load_report("feature_importance.json")
