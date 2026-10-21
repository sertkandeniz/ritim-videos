"""Shared stage context: config, course dir, manifest, glossary, backend factories and run timing."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..glossary import Glossary
from ..manifest import ManifestStore, RunRecord, Timer, now_iso
from ..runner import gpu_sampler
from ..workdir import CourseDir

log = logging.getLogger("ritim")


@dataclass
class RunStats:
    items: int = 0
    media_seconds: float = 0.0
    usage: dict[str, float] = field(default_factory=dict)
    backend: str = ""
    gpu: bool = False


@dataclass
class Ctx:
    cfg: Config
    cd: CourseDir
    store: ManifestStore
    langs: list[str]
    force: bool = False
    # Tests (and the blind test tool) inject backends here: key -> zero-arg factory.
    overrides: dict[str, Callable[[], Any]] = field(default_factory=dict)
    _glossary: Glossary | None = None

    @property
    def glossary(self) -> Glossary:
        if self._glossary is None:
            self._glossary = Glossary.load(self.cfg.glossary_path())
        return self._glossary

    def backend(self, key: str, factory: Callable[[], Any]) -> Any:
        return self.overrides[key]() if key in self.overrides else factory()

    def done(self, stage: str, item: str, fingerprint: str = "") -> bool:
        return not self.force and self.store.is_done(stage, item, fingerprint)

    @contextmanager
    def timed(self, stage: str) -> Iterator[RunStats]:
        stats = RunStats()
        gpu: dict[str, float] = {}
        started = now_iso()
        timer = Timer()
        with gpu_sampler(gpu):
            yield stats
        if stats.items:
            usage = {**stats.usage, **gpu}
            self.store.add_run(RunRecord(stage=stage, started_at=started, seconds=round(timer.elapsed, 2),
                                         items=stats.items, gpu=stats.gpu, backend=stats.backend,
                                         media_seconds=round(stats.media_seconds, 2), usage=usage))
            log.info("%s: %d items in %.1fs", stage, stats.items, timer.elapsed)
        else:
            log.info("%s: nothing to do", stage)
