import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.routes import router
from src.services.pipeline import warmup_models

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_models()
    yield


app = FastAPI(title="Doc-Nexus API", version="2.0.0", lifespan=lifespan)
app.include_router(router)