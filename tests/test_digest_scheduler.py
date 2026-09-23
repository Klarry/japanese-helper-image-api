"""Day 18: the scheduler - what makes the task periodic.

No HTTP here at all: the scheduler is a loop, and these tests drive it tick
by tick with the clock in hand. The collector is a stand-in so the runs are
instant and countable; what it does when it really runs is test_digest's job.
"""

import asyncio
from datetime import timedelta

import pytest

from app.services.digest import (
    DigestRun,
    DigestStore,
    DigestTaskStorage,
    as_iso,
    now_utc,
)
from app.services.digest_scheduler import DigestScheduler


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setenv("DIGEST_TASKS_FILE_PATH", str(tmp_path / "digest_tasks.json"))
    monkeypatch.setenv("DIGEST_STORE_FILE_PATH", str(tmp_path / "digest_store.json"))
    return tmp_path


def _scheduler(collector=None, calls=None):
    async def fake_collect(task, store):
        (calls if calls is not None else []).append(task.id)
        run = DigestRun(started_at=as_iso(now_utc()), ok=True, items=3)
        store.record(task.id, run, [])
        return run

    return DigestScheduler(collector=collector or fake_collect)


def test_a_new_task_runs_at_the_first_tick(files):
    DigestTaskStorage().create("Japanese words", 30)
    calls: list[str] = []

    ran = asyncio.run(_scheduler(calls=calls).tick())

    assert ran == ["digest-1"]
    assert calls == ["digest-1"]
    assert DigestStore().state_of("digest-1").total_runs == 1


def test_it_does_not_run_again_until_the_interval_has_passed(files):
    task = DigestTaskStorage().create("Japanese words", 30)
    scheduler = _scheduler()
    asyncio.run(scheduler.tick())
    after_first = DigestStore().state_of(task.id).last_run_at

    asyncio.run(scheduler.tick())  # immediately afterwards
    too_soon = DigestStore().state_of(task.id)

    assert too_soon.total_runs == 1
    assert too_soon.last_run_at == after_first


def test_it_runs_again_once_the_interval_has_passed(files):
    DigestTaskStorage().create("Japanese words", 30)
    scheduler = _scheduler()
    asyncio.run(scheduler.tick())

    asyncio.run(scheduler.tick(now=now_utc() + timedelta(seconds=31)))

    assert DigestStore().state_of("digest-1").total_runs == 2


def test_a_task_that_was_missed_while_the_backend_was_down_runs_once_not_sixty(files):
    """The whole point of working the next run out from the last one."""
    DigestTaskStorage().create("Japanese words", 60)
    scheduler = _scheduler()
    asyncio.run(scheduler.tick())

    asyncio.run(scheduler.tick(now=now_utc() + timedelta(hours=1)))

    assert DigestStore().state_of("digest-1").total_runs == 2


def test_every_due_task_runs_on_the_same_tick(files):
    DigestTaskStorage().create("Japanese words", 15)
    DigestTaskStorage().create("N5 words", 15)
    calls: list[str] = []

    assert asyncio.run(_scheduler(calls=calls).tick()) == ["digest-1", "digest-2"]


def test_nothing_happens_without_a_task(files):
    assert asyncio.run(_scheduler().tick()) == []


def test_one_task_that_blows_up_does_not_stop_the_others(files):
    DigestTaskStorage().create("explodes", 15)
    DigestTaskStorage().create("works", 15)
    survivors: list[str] = []

    async def collector(task, store):
        if task.query == "explodes":
            raise RuntimeError("the collector crashed")

        survivors.append(task.id)
        store.record(task.id, DigestRun(started_at=as_iso(now_utc()), ok=True, items=1), [])

    ran = asyncio.run(DigestScheduler(collector=collector).tick())

    assert survivors == ["digest-2"]
    assert ran == ["digest-2"]


def test_the_loop_keeps_ticking_on_its_own_with_nobody_asking(files):
    """No HTTP request, no call: once started, the loop runs the task by
    itself - which is the difference between a tool and a periodic task."""
    DigestTaskStorage().create("Japanese words", 10)
    calls: list[str] = []

    async def drive():
        loop = asyncio.create_task(_scheduler(calls=calls).run_forever())
        await asyncio.sleep(0.05)
        loop.cancel()

        try:
            await loop
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())

    assert calls == ["digest-1"]
    assert DigestStore().state_of("digest-1").total_runs == 1


def test_the_runs_already_done_are_read_back_after_a_restart(files):
    """A fresh scheduler over the same files does not start the count again,
    and does not re-run a task whose interval has not passed."""
    DigestTaskStorage().create("Japanese words", 30)
    asyncio.run(_scheduler().tick())

    restarted = _scheduler()

    assert asyncio.run(restarted.tick()) == []
    assert DigestStore().state_of("digest-1").total_runs == 1
