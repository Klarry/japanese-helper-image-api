import logging

from fastapi import FastAPI

from app.api.routes import description, image_search
from app.core.config import LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)

app = FastAPI()

app.include_router(image_search.router)
app.include_router(description.router)
