import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
# The Interactions API above has no temperature knob (confirmed by Google's
# docs and by the API itself: "Unknown parameter 'temperature'."). The
# classic generateContent endpoint does, via generationConfig.temperature,
# so temperature-controlled calls use this URL instead - same model, same
# API key.
GEMINI_GENERATE_CONTENT_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
# Gemini's own tokenizer, exposed as a plain REST call - used to get a real
# (not estimated) token count for an arbitrary piece of text, such as the
# agent's conversation history alone.
GEMINI_COUNT_TOKENS_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens"
)
IMAGE_SEARCH_MODEL = "gemini-3.1-flash-image"
TEXT_MODEL = "gemini-3.5-flash"

MAX_IMAGE_WIDTH = 800
JPEG_QUALITY = 75

HTTP_TIMEOUT = 120.0
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Where the Japanese Learning Agent's conversation history is persisted -
# a single JSON file, relative to the working directory (same convention
# as .env above). Overridable so tests can point it at a throwaway path.
AGENT_HISTORY_FILE_PATH = os.getenv("AGENT_HISTORY_FILE_PATH", "data/agent_history.json")
