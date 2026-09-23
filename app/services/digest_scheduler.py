"""Day 18: the thing that makes a periodic task periodic.

One asyncio task, started with the backend and running for as long as it
does. Every tick it reads the task file, asks each task whether it is due,
and runs the ones that are. That is the whole scheduler - no Celery, no
Redis, no second process: the backend is already a long-lived asyncio
program, and this is a loop inside it.

What that buys, deliberately:

* it is **independent of HTTP**. The request that created the task is long
  gone; nothing has to be called again for the runs to keep happening.
* it **survives a restart**. The tasks are on disk, the runs already done
  are on disk, and the next run is worked out from the last one - so a
  backend that was down for an hour does one run on the way back up, not
  sixty.
* a run that fails **does not stop the task**. The failure is recorded and
  the next tick carries on.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime

from app.services.digest import (
    DigestRun,
    DigestStore,
    DigestTask,
    DigestTaskStorage,
    collect_once,
    next_run_at,
    now_utc,
)

logger = logging.getLogger(__name__)

# How often to look at the clock. Short enough that a 10-second task is not
# noticeably late, cheap enough to leave running: one small JSON read.
TICK_SECONDS = 1.0

Collector = Callable[[DigestTask, DigestStore], Awaitable[DigestRun]]


class DigestScheduler:
    """Runs the digest tasks that are due."""

    def __init__(
        self,
        tasks: DigestTaskStorage | None = None,
        store: DigestStore | None = None,
        collector: Collector = collect_once,
        tick_seconds: float = TICK_SECONDS,
    ) -> None:
        self._tasks = tasks or DigestTaskStorage()
        self._store = store or DigestStore()
        self._collect = collector
        self._tick_seconds = tick_seconds

    def due(self, task: DigestTask, now: datetime) -> bool:
        if not task.active:
            return False

        scheduled = next_run_at(task, self._store.state_of(task.id))

        return scheduled is None or now >= scheduled

    async def tick(self, now: datetime | None = None) -> list[str]:
        """Run every task that is due, and report which ones ran.

        Never raises: one task's failure is recorded by collect_once, and
        anything unexpected is logged rather than allowed to kill the loop
        that every other task depends on.
        """
        moment = now or now_utc()
        ran: list[str] = []

        for task in self._tasks.load():
            if not self.due(task, moment):
                continue

            try:
                await self._collect(task, self._store)
                ran.append(task.id)
            except Exception as error:  # noqa: BLE001 - see docstring
                logger.exception("Digest %s could not run: %s", task.id, error)

        return ran

    async def run_forever(self) -> None:
        logger.info("Digest scheduler started, ticking every %ss", self._tick_seconds)

        while True:
            await self.tick()
            await asyncio.sleep(self._tick_seconds)


def start(scheduler: DigestScheduler | None = None) -> asyncio.Task:
    """Start the loop as a background task of the running app."""
    return asyncio.create_task((scheduler or DigestScheduler()).run_forever(), name="digest-scheduler")


async def stop(task: asyncio.Task | None) -> None:
    if task is None:
        return

    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    logger.info("Digest scheduler stopped")
