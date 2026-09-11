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

# --- Agent history compression (Day 7 experiment) --------------------------
# Off by default: with the flag unset the agent behaves exactly as it did
# before - the whole conversation is sent to Gemini on every turn. Turning it
# on is what the experiment compares against, so it is a plain env flag
# rather than something baked into the request contract.
AGENT_COMPRESSION_ENABLED = os.getenv("AGENT_COMPRESSION_ENABLED", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# How many of the newest messages are always kept verbatim, and how many
# messages have to age out past that window before the summary is rewritten.
# Summarising costs a Gemini call, so it happens in batches rather than on
# every turn.
AGENT_RECENT_MESSAGES_KEPT = int(os.getenv("AGENT_RECENT_MESSAGES_KEPT", "6"))
AGENT_SUMMARY_UPDATE_THRESHOLD = int(os.getenv("AGENT_SUMMARY_UPDATE_THRESHOLD", "10"))
# One record per /agent/chat call, so token spend with and without
# compression can be compared after the fact.
AGENT_USAGE_LOG_FILE_PATH = os.getenv("AGENT_USAGE_LOG_FILE_PATH", "data/agent_token_usage.json")

# --- Agent context strategies (Day 8 experiment) ---------------------------
# How many key-value facts the Sticky Facts strategy keeps. It is a memory of
# what stays useful later - goals, constraints, preferences, decisions - not a
# transcript, so it is deliberately small.
AGENT_FACTS_LIMIT = int(os.getenv("AGENT_FACTS_LIMIT", "20"))
