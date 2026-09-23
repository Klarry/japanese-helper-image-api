"""Day 18: the periodic digest - the task, one run, and what piles up.

The JLPT API is the local stand-in (the same one the other MCP tests use);
everything else - the files, the counters, the summary - is real.
"""

import asyncio
import json
from contextlib import contextmanager
from datetime import timedelta

import pytest

from app.services.digest import (
    WORDS_PER_RUN,
    DigestStore,
    DigestTaskStorage,
    build_digest,
    collect_once,
    from_iso,
    level_in,
    next_run_at,
)

from tests.jlpt_api_standin import closed_port_url, jlpt_api


@pytest.fixture(autouse=True)
def _local_traffic_skips_any_proxy(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setenv("DIGEST_TASKS_FILE_PATH", str(tmp_path / "digest_tasks.json"))
    monkeypatch.setenv("DIGEST_STORE_FILE_PATH", str(tmp_path / "digest_store.json"))
    return tmp_path


@contextmanager
def api_at(monkeypatch, **behaviour):
    """The stand-in for the JLPT API, with the digest pointed at it - the
    same environment variable the MCP client passes to the server."""
    with jlpt_api(**behaviour) as (url, received):
        monkeypatch.setenv("JLPT_VOCAB_API_URL", url)
        yield received


def _run(task, store):
    return asyncio.run(collect_once(task, store))


# --- creating the task -----------------------------------------------------


def test_a_task_is_created_and_written_to_disk(files):
    task = DigestTaskStorage().create("Japanese words", 15)

    assert task.id == "digest-1"
    assert task.interval_seconds == 15
    assert task.created_at
    stored = json.loads((files / "digest_tasks.json").read_text(encoding="utf-8"))
    assert stored["tasks"][0]["query"] == "Japanese words"


def test_a_level_named_in_the_query_is_what_the_task_collects(files):
    assert DigestTaskStorage().create("collect N3 words", 20).level == 3
    assert level_in("Japanese words") is None


def test_an_interval_outside_the_allowed_range_is_brought_back_into_it(files):
    storage = DigestTaskStorage()

    assert storage.create("Japanese words", 1).interval_seconds == 10
    assert storage.create("Japanese words", 10_000_000).interval_seconds == 86_400


def test_tasks_pile_up_rather_than_replace_each_other(files):
    storage = DigestTaskStorage()
    storage.create("Japanese words", 15)
    second = storage.create("N5 words", 30)

    assert [task.id for task in storage.load()] == ["digest-1", "digest-2"]
    assert storage.latest() == second


# --- one run ---------------------------------------------------------------


def test_a_run_collects_words_from_the_api_and_stores_them_with_a_timestamp(files, monkeypatch):
    with api_at(monkeypatch) as received:
        task = DigestTaskStorage().create("N3 words", 15)
        store = DigestStore()

        run = _run(task, store)

    assert [request["path"] for request in received] == ["/api/words/random"] * WORDS_PER_RUN
    assert received[0]["query"] == {"level": "3"}
    assert run.ok and run.items == WORDS_PER_RUN

    state = store.state_of(task.id)
    assert state.total_runs == 1
    assert state.total_items == WORDS_PER_RUN
    assert state.items[0].collected_at == run.started_at
    assert state.items[0].jlpt_level == "N3"


def test_every_run_adds_to_what_is_already_there(files, monkeypatch):
    with api_at(monkeypatch):
        task = DigestTaskStorage().create("Japanese words", 15)
        store = DigestStore()
        _run(task, store)
        _run(task, store)

    state = store.state_of(task.id)
    assert state.total_runs == 2
    assert state.total_items == 2 * WORDS_PER_RUN
    assert state.first_run_at and state.last_run_at


def test_an_api_that_fails_makes_a_failed_run_not_a_dead_task(files, monkeypatch):
    task = DigestTaskStorage().create("Japanese words", 15)
    store = DigestStore()

    with api_at(monkeypatch, status=500):
        failed = _run(task, store)

    with api_at(monkeypatch):
        recovered = _run(task, store)

    assert not failed.ok and "status 500" in failed.error
    state = store.state_of(task.id)
    assert state.total_runs == 2
    assert state.failed_runs == 1
    assert state.last_error == ""  # the run after it succeeded
    assert recovered.items == WORDS_PER_RUN


def test_an_api_that_is_down_is_recorded_with_its_reason(files, monkeypatch):
    monkeypatch.setenv("JLPT_VOCAB_API_URL", closed_port_url())
    task = DigestTaskStorage().create("Japanese words", 15)
    store = DigestStore()

    run = _run(task, store)

    assert not run.ok
    assert "unreachable" in run.error
    assert store.state_of(task.id).last_error == run.error


# --- the aggregate ---------------------------------------------------------


def test_the_digest_reports_the_runs_the_words_and_a_summary(files, monkeypatch):
    with api_at(monkeypatch):
        task = DigestTaskStorage().create("N5 words", 20)
        store = DigestStore()
        _run(task, store)
        _run(task, store)

    digest = build_digest(task, store.state_of(task.id))

    assert digest["runs"] == 2
    assert digest["items_collected"] == 2 * WORDS_PER_RUN
    assert digest["last_run"] == store.state_of(task.id).last_run_at
    assert digest["levels"] == {"N5": 2 * WORDS_PER_RUN}
    assert len(digest["latest_items"]) == 5
    assert "2 run(s)" in digest["summary"] and "N5 x6" in digest["summary"]
    assert digest["query"] == "N5 words"


def test_a_task_that_has_not_run_yet_says_so(files):
    task = DigestTaskStorage().create("Japanese words", 15)

    digest = build_digest(task, DigestStore().state_of(task.id))

    assert digest["runs"] == 0
    assert digest["items_collected"] == 0
    assert "has not run yet" in digest["summary"]


def test_the_next_run_follows_the_last_one_by_the_interval(files, monkeypatch):
    with api_at(monkeypatch):
        task = DigestTaskStorage().create("Japanese words", 30)
        store = DigestStore()
        _run(task, store)

    state = store.state_of(task.id)
    assert next_run_at(task, state) == from_iso(state.last_run_at) + timedelta(seconds=30)


# --- across a restart ------------------------------------------------------


def test_everything_survives_a_restart_of_the_backend(files, monkeypatch):
    """New storage objects over the same files - which is all a restart is
    to this module."""
    with api_at(monkeypatch):
        task = DigestTaskStorage().create("Japanese words", 15)
        _run(task, DigestStore())
        _run(task, DigestStore())

    restarted_task = DigestTaskStorage().latest()
    restarted_state = DigestStore().state_of(restarted_task.id)

    assert restarted_task.id == task.id
    assert restarted_task.interval_seconds == 15
    assert restarted_state.total_runs == 2
    assert restarted_state.total_items == 2 * WORDS_PER_RUN
    assert build_digest(restarted_task, restarted_state)["runs"] == 2


def test_a_broken_store_reads_as_empty_rather_than_crashing(files):
    (files / "digest_store.json").write_text("{not json", encoding="utf-8")
    task = DigestTaskStorage().create("Japanese words", 15)

    assert DigestStore().state_of(task.id).total_runs == 0
