"""Chain of Responsibility: every image travels through a chain of steps.

Each step does its own work on the shared ``ImageContext`` and decides whether the image goes on
to the next step. An exception stops the chain for that image only (the run continues), and the
output gathered so far is kept: e.g. if the LLM fails, the rules output stays.

Before the first image, every step validates once what it needs for the whole run (``validate``).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar

from menu_poc.data_classes import ImageContext
from menu_poc.typings import ValueT

log = logging.getLogger(__name__)


class InputValidationError(RuntimeError):
    """The run cannot start: no images, or a step that every image needs cannot run."""


class MissingInputError(RuntimeError):
    """A step ran without the output of an earlier step it depends on."""


def require(value: ValueT | None, what: str) -> ValueT:
    """The output of an earlier step. The builder's fixed order guarantees it, so a missing one
    is a bug; it fails this image only, like any other step error."""
    if value is None:
        raise MissingInputError(f"{what} is missing: the steps ran out of order")
    return value


class Step(ABC):
    name: ClassVar[str]

    def __init__(self) -> None:
        self.next: Step | None = None

    def validate(self) -> list[str]:
        """Checked once, before any image. Raise ``InputValidationError`` when the run cannot go
        on without this step; return warnings for what the step will not do. Default: nothing."""
        return []

    def handle(self, ctx: ImageContext) -> ImageContext:
        try:
            proceed = self.process(ctx)
        except Exception as exc:  # one bad image must not stop the run
            log.exception("%s failed for %s", self.name, ctx.ref.image_id)
            ctx.result.error = f"{self.name}: {exc}"
            return ctx
        if proceed and self.next is not None:
            return self.next.handle(ctx)
        return ctx

    @abstractmethod
    def process(self, ctx: ImageContext) -> bool:
        """Do this step's work; return True to pass the image to the next step."""
