from dataclasses import dataclass
from typing import cast, TYPE_CHECKING

from yaml import YAMLObject

if TYPE_CHECKING:
    from typing import Protocol
    from _typeshed import (
        SupportsAllComparisons,
        SupportsAdd
    )
    class Curriculable(SupportsAllComparisons, SupportsAdd, Protocol): ... # pyright: ignore[reportMissingTypeArgument]

@dataclass
class Curriculum[T: Curriculable](YAMLObject):

    yaml_tag = u"!Curriculum"

    start : T
    stop : T
    step_len : int
    step_size : T

    @staticmethod
    def wrap(item: "T | Curriculum[T]", typ: type) -> "Curriculum[T]":
        if isinstance(item, Curriculum):
            return item

        return Curriculum[T](
            start=item,
            stop=item,
            step_size=typ(False), # careful: this is only tested on `int | float`
            step_len=int(1e99)
        )

    def _prep_phases(self):
        if getattr(self, "phases", None) is None:
            self.phases: list[tuple[T, int]] = [ ]
            current_value: T = self.start
            current_step = 0
            while current_value < self.stop:
                self.phases.append((current_value, current_step))
                current_step += self.step_len
                current_value = cast(T, current_value + self.step_size)

    @property
    def _max_phases(self) -> int:
        self._prep_phases()
        return len(self.phases)

    @property
    def partitioning_steps(self) -> list[int]:
        self._prep_phases()
        return [ p[1] for p in self.phases ]

    @property
    def max(self) -> T:
        self._prep_phases()
        return max(self.phases[0][0], self.phases[-1][0])

    @property
    def min(self) -> T:
        self._prep_phases()
        return min(self.phases[0][0], self.phases[-1][0])

    def at_step(self, step_num: int) -> T:
        if step_num < 0:
            raise ValueError(f"The step cannot be less than 0! Got: {step_num}")
        
        self._prep_phases()
        if not hasattr(self, "step_len"):
            # shortcut for passthrough
            return self.phases[0][0]

        idx = step_num // self.step_len
        idx = min(idx, len(self.phases) - 1)
        return self.phases[idx][0]

    def __repr__(self):
        return f"Curriculum({self.start} to {self.stop}," + \
                f"step_size={self.step_size}, step_length={self.step_len})"

