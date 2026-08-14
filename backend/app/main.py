import asyncio
import contextlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config, simulator
from .inference import ArtifactsNotFound, get_bundle
from .routers import models, stream, transactions

_simulator_task: asyncio.Task | None = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global _simulator_task
    try:
        get_bundle()  # warm the model cache and fail fast with a clear error
        _simulator_task = asyncio.create_task(simulator.run_simulator())
    except (ArtifactsNotFound, FileNotFoundError) as e:
        print(f"[startup] Live simulator not started: {e}")
    yield
    if _simulator_task:
        _simulator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _simulator_task


app = FastAPI(title="Fraud Detection API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transactions.router)
app.include_router(models.router)
app.include_router(stream.router)


@app.get("/api/health")
async def health():
    try:
        get_bundle()
        models_ready = True
    except (ArtifactsNotFound, FileNotFoundError):
        models_ready = False
    return {"status": "ok", "models_ready": models_ready}
