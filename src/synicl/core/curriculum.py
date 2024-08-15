
from typing import (
    TypeAlias,
    Any,
    override,
    cast,
    TYPE_CHECKING,
)
if TYPE_CHECKING:
    from typing import Protocol
    from _typeshed import (
        SupportsAllComparisons,
        SupportsAdd
    )
    class Curriculable(SupportsAllComparisons, SupportsAdd, Protocol): ... # pyright: ignore[reportMissingTypeArgument]

YamlMap: TypeAlias = dict[str, Any]
YamlList: TypeAlias = list[Any]

class Curriculum[T: Curriculable]:

    @override
    def __init__(self, obj: "T | Curriculum[T]") -> None: ... # pyright: ignore[reportRedeclaration, reportMissingSuperCall]

    @override
    def __init__(self,
            start : "T | Curriculum[T]",
            stop : None | T = None,
            step_len : int = 0,
            step_size : None | T = None,
        ):

        super().__init__()

        if stop is None or step_len == 0 or step_size is None:
            if stop is not None or step_len != 0 or step_size is not None:
                raise TypeError(f"Curriculum accepts only 1 or 4 arguments")
            if not isinstance(start, Curriculum):
                self.phases: list[tuple[T, int]] = [ (start, 0) ]
                return
            obj = start
            self.start = obj.start
            self.stop = obj.stop
            self.step_len = obj.step_len
            self.step_size = obj.step_size
            self.phases = obj.phases
            return

        if isinstance(start, Curriculum):
            raise TypeError(f"Double-wrapping a Curriculum cannot have two sets of parameters!")

        self.start = start
        self.stop = stop
        self.step_len = step_len
        self.step_size = step_size
        self.phases: list[tuple[T, int]] = [ ]
        current_value: T = start
        current_step = 0
        while current_value < stop:
            self.phases.append((current_value, current_step))
            current_step += step_len
            current_value = cast(T, current_value + step_size)

    @property
    def _max_phases(self) -> int:
        return len(self.phases)

    @property
    def partitioning_steps(self) -> list[int]:
        return [ p[1] for p in self.phases ]

    @property
    def max(self) -> T:
        return max(self.phases[0][0], self.phases[-1][0])

    @property
    def min(self) -> T:
        return min(self.phases[0][0], self.phases[-1][0])

    def at_step(self, step_num: int) -> T:
        if step_num < 0:
            raise ValueError(f"The step cannot be less than 0! Got: {step_num}")

        if not hasattr(self, "step_len"):
            # shortcut for passthrough
            return self.phases[0][0]

        idx = step_num // self.step_len
        idx = min(idx, len(self.phases) - 1)
        return self.phases[idx][0]

    def __repr__(self):
        return f"Curriculum({self.start} to {self.stop}," + \
                f"step_size={self.step_size}, step_length={self.step_len})"
    
    @property
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError(f"Metadata not yet implemented")
