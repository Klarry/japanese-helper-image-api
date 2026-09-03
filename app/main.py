import logging

from fastapi import FastAPI

from app.api.routes import description, image_search, kanji_word_set, temperature_description
from app.core.config import LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)

app = FastAPI()

app.include_router(image_search.router)
app.include_router(description.router)
app.include_router(kanji_word_set.router)
app.include_router(temperature_description.router)
