"""Day 12: who the learner is, as settings rather than as memory.

The memory layers answer "what has been said" - this answers "how this
learner wants to be answered": their level, the style and length they want,
which language to reply in, which language to translate into. That is a
setting, not something the conversation happens to mention, so it is kept
apart from all three memory layers, in its own file, and applied to every
request no matter which context strategy is running.

Keeping it out of the layers matters in both directions: clearing the
conversation, the task or long-term memory must not reset the learner's
level to nothing, and setting a profile must not put words into a
conversation the learner never said.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import AGENT_PROFILE_PREFERENCES_LIMIT, AGENT_USER_PROFILE_FILE_PATH

logger = logging.getLogger(__name__)

_MAX_VALUE_LENGTH = 120

# The prompt line each setting becomes, in the order the model reads them.
# Everything the profile can say is here, so a new setting is one entry plus
# one field - and nothing else in the agent has to know about it.
_SETTING_LABELS = (
    ("preferred_language", "Write the answer in"),
    ("japanese_level", "The learner's Japanese level (JLPT)"),
    ("explanation_style", "Explanation style"),
    ("answer_format", "Answer format"),
    ("translation_language", "Translate Japanese into"),
)

_PROFILE_CAPTION = (
    "USER PROFILE (how this learner wants every answer written - apply it without "
    "being asked; if their message asks for something different, the message wins):"
)


def _as_value(value: Any) -> str:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return ""

    return str(value).strip()[:_MAX_VALUE_LENGTH]


def _as_preferences(value: Any, limit: int = AGENT_PROFILE_PREFERENCES_LIMIT) -> list[str]:
    if not isinstance(value, list):
        return []

    return [entry for entry in (_as_value(item) for item in value) if entry][:limit]


@dataclass
class UserProfile:
    """The learner's answering preferences. Every field is optional: an unset
    one is simply not mentioned to the model, rather than sent as an empty
    instruction it would have to interpret."""

    preferred_language: str = ""
    japanese_level: str = ""
    explanation_style: str = ""
    answer_format: str = ""
    translation_language: str = ""
    preferences: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(
            (
                self.preferred_language,
                self.japanese_level,
                self.explanation_style,
                self.answer_format,
                self.translation_language,
                self.preferences,
            )
        )

    def as_section(self) -> str:
        """The profile as the model reads it. Empty when nothing is set, so a
        learner without a profile gets exactly the prompt they got before."""
        if self.is_empty():
            return ""

        lines = [
            f"- {label}: {getattr(self, name)}"
            for name, label in _SETTING_LABELS
            if getattr(self, name)
        ]
        lines += [f"- Also: {entry}" for entry in self.preferences]

        return f"{_PROFILE_CAPTION}\n" + "\n".join(lines)


def user_profile_from_json(value: Any) -> UserProfile:
    if not isinstance(value, dict):
        return UserProfile()

    return UserProfile(
        preferred_language=_as_value(value.get("preferred_language")),
        japanese_level=_as_value(value.get("japanese_level")),
        explanation_style=_as_value(value.get("explanation_style")),
        answer_format=_as_value(value.get("answer_format")),
        translation_language=_as_value(value.get("translation_language")),
        preferences=_as_preferences(value.get("preferences")),
    )


def user_profile_as_json(profile: UserProfile) -> dict[str, Any]:
    return {
        "preferred_language": profile.preferred_language,
        "japanese_level": profile.japanese_level,
        "explanation_style": profile.explanation_style,
        "answer_format": profile.answer_format,
        "translation_language": profile.translation_language,
        "preferences": profile.preferences,
    }


class AgentUserProfileStorage:
    """The profile on disk, in its own file.

    Same fail-safe rules as the conversation and long-term memory: a missing,
    empty or broken file reads as "no profile set" rather than raising, so a
    damaged file can never take the agent down with it.
    """

    def __init__(self, file_path: str = AGENT_USER_PROFILE_FILE_PATH) -> None:
        self._path = Path(file_path)

    def load(self) -> UserProfile:
        if not self._path.exists():
            return UserProfile()

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read user profile from %s: %s", self._path, exc)
            return UserProfile()

        return user_profile_from_json(data)

    def save(self, profile: UserProfile) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(user_profile_as_json(profile), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self.save(UserProfile())


user_profile_storage = AgentUserProfileStorage()
