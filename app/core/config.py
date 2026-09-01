import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
IMAGE_SEARCH_MODEL = "gemini-3.1-flash-image"
TEXT_MODEL = "gemini-3.1-flash"

MAX_IMAGE_WIDTH = 800
JPEG_QUALITY = 75

HTTP_TIMEOUT = 120.0
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
