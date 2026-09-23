import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import (
    agent,
    description,
    image_search,
    kanji_word_set,
    model_comparison,
    temperature_description,
)
from app.core.config import AGENT_DIGEST_SCHEDULER_ENABLED, LOG_LEVEL
from app.services import digest_scheduler

logging.basicConfig(level=LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """The digest scheduler runs for as long as the app does.

    Started here rather than on a request, because that is the point of a
    periodic task: once created, it keeps running with nobody asking.
    """
    scheduler = digest_scheduler.start() if AGENT_DIGEST_SCHEDULER_ENABLED else None

    try:
        yield
    finally:
        await digest_scheduler.stop(scheduler)


app = FastAPI(lifespan=lifespan)

app.include_router(image_search.router)
app.include_router(description.router)
app.include_router(kanji_word_set.router)
app.include_router(temperature_description.router)
app.include_router(model_comparison.router)
app.include_router(agent.router)
