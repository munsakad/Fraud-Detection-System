from fastapi import APIRouter, HTTPException

from .. import simulator
from ..inference import ArtifactsNotFound, get_bundle
from ..schemas import TransactionScoreRequest

router = APIRouter(prefix="/api/transactions", tags=["transactions"])


@router.get("/recent")
async def recent(limit: int = 50):
    return simulator.recent_events(limit)


@router.get("/stats")
async def stats():
    return simulator.get_stats()


@router.post("/score")
async def score_transaction(req: TransactionScoreRequest):
    try:
        bundle = get_bundle()
    except ArtifactsNotFound as e:
        raise HTTPException(400, str(e))
    scores = bundle.score(req.to_feature_row())
    return {"input": req.model_dump(), "scores": scores}
