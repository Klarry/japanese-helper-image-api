import json

from app.schemas.agent import InvariantCategory
from app.services.agent_invariants import (
    PROJECT_INVARIANTS,
    AgentInvariantsStorage,
    Invariant,
    as_section,
)


def _rules(invariants):
    return [item.rule for item in invariants]


# --- what the model is told ------------------------------------------------


def test_no_rules_means_nothing_is_sent():
    """An agent without invariants gets exactly the prompt it got before
    this layer existed."""
    assert as_section([]) == ""


def test_the_rules_are_grouped_by_category_in_a_fixed_order():
    section = as_section(list(PROJECT_INVARIANTS))

    assert section.index("Architecture:") < section.index("Technology stack:")
    assert section.index("Technology stack:") < section.index("Technical decisions:")


def test_a_category_with_no_rules_is_left_out():
    section = as_section([Invariant("a", InvariantCategory.ARCHITECTURE, "ViewModel -> Repository -> API")])

    assert "Architecture:" in section
    assert "Business rules:" not in section
    assert "Technology stack:" not in section


def test_every_project_rule_reaches_the_model():
    section = as_section(list(PROJECT_INVARIANTS))

    for rule in _rules(PROJECT_INVARIANTS):
        assert rule in section


def test_the_section_carries_the_protocol_for_a_conflict():
    """The rules alone would only make the agent refuse; the protocol is
    what makes it explain and offer something instead."""
    section = as_section(list(PROJECT_INVARIANTS))

    assert "must never be broken" in section
    assert "do not help work around it" in section
    assert "name the rule it conflicts with" in section
    assert "explain why the rule exists" in section
    assert "offer an alternative that respects the rules" in section


# --- on disk ---------------------------------------------------------------


def test_a_fresh_install_starts_out_constrained(tmp_path):
    """No file yet means the project's own rules, written down - not an
    unconstrained agent."""
    path = tmp_path / "invariants.json"

    loaded = AgentInvariantsStorage(file_path=str(path)).load()

    assert _rules(loaded) == _rules(PROJECT_INVARIANTS)
    assert path.exists()


def test_a_rule_deleted_stays_deleted_across_restarts(tmp_path):
    """Seeding happens once, on the missing file - it must not resurrect a
    rule someone removed on purpose."""
    path = str(tmp_path / "invariants.json")
    storage = AgentInvariantsStorage(file_path=path)
    kept = [item for item in storage.load() if item.id != "decision-no-sqlite"]
    storage.save(kept)

    restarted = AgentInvariantsStorage(file_path=path).load()

    assert "decision-no-sqlite" not in [item.id for item in restarted]
    assert len(restarted) == len(PROJECT_INVARIANTS) - 1


def test_rules_survive_a_fresh_storage_instance(tmp_path):
    path = str(tmp_path / "invariants.json")
    AgentInvariantsStorage(file_path=path).save(
        [Invariant("br-1", InvariantCategory.BUSINESS_RULES, "Учебные материалы только для JLPT")]
    )

    restarted = AgentInvariantsStorage(file_path=path).load()

    assert restarted == [Invariant("br-1", InvariantCategory.BUSINESS_RULES, "Учебные материалы только для JLPT")]


def test_an_empty_list_is_kept_rather_than_reseeded(tmp_path):
    path = str(tmp_path / "invariants.json")
    storage = AgentInvariantsStorage(file_path=path)
    storage.save([])

    assert storage.load() == []


def test_a_broken_file_reads_as_no_rules(tmp_path):
    path = tmp_path / "invariants.json"
    path.write_text("{ not json", encoding="utf-8")

    assert AgentInvariantsStorage(file_path=str(path)).load() == []


def test_entries_that_cannot_be_read_are_skipped_rather_than_stored(tmp_path):
    path = tmp_path / "invariants.json"
    path.write_text(
        json.dumps(
            {
                "invariants": [
                    {"id": "ok", "category": "architecture", "rule": "ViewModel -> Repository -> API"},
                    {"id": "bad-category", "category": "vibes", "rule": "что-нибудь"},
                    {"id": "", "category": "architecture", "rule": "без id"},
                    {"id": "no-rule", "category": "architecture", "rule": "   "},
                    "not an object",
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = AgentInvariantsStorage(file_path=str(path)).load()

    assert [item.id for item in loaded] == ["ok"]


def test_the_invariants_file_holds_no_conversation(tmp_path):
    """They are rules, not memory: nothing a conversation says is in here."""
    path = tmp_path / "invariants.json"
    AgentInvariantsStorage(file_path=str(path)).load()

    stored = json.loads(path.read_text(encoding="utf-8"))

    assert set(stored) == {"invariants"}
    assert all(set(entry) == {"id", "category", "rule"} for entry in stored["invariants"])
