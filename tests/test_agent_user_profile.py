import json

from app.services.agent_user_profile import AgentUserProfileStorage, UserProfile


def _profile(**kwargs) -> UserProfile:
    return UserProfile(**kwargs)


# --- what the model is told ------------------------------------------------


def test_a_profile_that_is_not_set_says_nothing_at_all():
    """A learner without a profile gets exactly the prompt they got before
    the profile existed - not an empty heading the model has to interpret."""
    assert UserProfile().as_section() == ""
    assert UserProfile().is_empty()


def test_every_setting_becomes_one_line_the_model_can_follow():
    section = _profile(
        preferred_language="Russian",
        japanese_level="N4",
        explanation_style="simple",
        answer_format="short",
        translation_language="Russian",
    ).as_section()

    assert section.startswith("USER PROFILE")
    assert "Write the answer in: Russian" in section
    assert "The learner's Japanese level (JLPT): N4" in section
    assert "Explanation style: simple" in section
    assert "Answer format: short" in section
    assert "Translate Japanese into: Russian" in section


def test_settings_that_are_not_set_are_left_out_rather_than_sent_empty():
    section = _profile(japanese_level="N2").as_section()

    assert "N2" in section
    assert "Explanation style" not in section
    assert "Translate Japanese into" not in section


def test_extra_preferences_are_sent_as_their_own_lines():
    section = _profile(japanese_level="N4", preferences=["без ромадзи", "примеры из жизни"]).as_section()

    assert "- Also: без ромадзи" in section
    assert "- Also: примеры из жизни" in section


def test_the_learners_own_message_is_allowed_to_override_the_profile():
    """The profile is a default, not a gag order - the caption says so."""
    assert "the message wins" in _profile(japanese_level="N4").as_section()


def test_two_profiles_produce_two_different_sections():
    a = _profile(japanese_level="N4", explanation_style="simple", answer_format="short",
                 translation_language="Russian").as_section()
    b = _profile(japanese_level="N2", explanation_style="detailed", answer_format="detailed",
                 translation_language="English").as_section()

    assert a != b
    assert "N4" in a and "N4" not in b
    assert "English" in b and "English" not in a


# --- on disk ---------------------------------------------------------------


def test_a_profile_survives_a_fresh_storage_instance(tmp_path):
    path = str(tmp_path / "profile.json")
    AgentUserProfileStorage(file_path=path).save(
        _profile(japanese_level="N4", answer_format="short", preferences=["без ромадзи"])
    )

    restarted = AgentUserProfileStorage(file_path=path).load()

    assert restarted.japanese_level == "N4"
    assert restarted.answer_format == "short"
    assert restarted.preferences == ["без ромадзи"]


def test_no_profile_file_reads_as_nothing_set(tmp_path):
    assert AgentUserProfileStorage(file_path=str(tmp_path / "missing.json")).load().is_empty()


def test_a_broken_profile_file_reads_as_nothing_set(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("{ not json at all", encoding="utf-8")

    assert AgentUserProfileStorage(file_path=str(path)).load().is_empty()


def test_a_profile_keeps_only_what_it_can_read(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps({"japanese_level": "N4", "answer_format": {"nested": 1}, "preferences": "not a list"}),
        encoding="utf-8",
    )

    profile = AgentUserProfileStorage(file_path=str(path)).load()

    assert profile.japanese_level == "N4"
    assert profile.answer_format == ""
    assert profile.preferences == []


def test_clearing_the_profile_empties_the_file_rather_than_deleting_it(tmp_path):
    path = tmp_path / "profile.json"
    storage = AgentUserProfileStorage(file_path=str(path))
    storage.save(_profile(japanese_level="N2"))

    storage.clear()

    assert storage.load().is_empty()
    assert json.loads(path.read_text(encoding="utf-8"))["japanese_level"] == ""
