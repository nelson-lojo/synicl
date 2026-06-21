import os

from dataclasses import dataclass, asdict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tqdm import tqdm
from functools import partial
from enum import StrEnum

import wandb

from torch import optim, nn
from torch import save as torch_save
from yaml import YAMLObject

from ..core import ShapeError, UninitializedError
from ..model import TrainableModel
from ..function_class import FunctionClass

@dataclass
class Logger(YAMLObject):

    yaml_tag = u'!Logger'

    class Type(StrEnum):
        WANDB_OFFLINE = "WANDB_OFFLINE"
        WANDB_ONLINE = "WANDB_ONLINE"
        CONSOLE = "CONSOLE"

    type: Type
    log_freq: int
    checkpoint_freq: int
    write_dir: str # absolute path to write information/checkpoints to
    project_name: str = ""
    run_name: str = ""

    def start(self, trainer: "Trainer"):
        """Start a wandb run and put it in `self.wandb_run: wandb.Run`"""

        match self.type:
            case Logger.Type.WANDB_OFFLINE | Logger.Type.WANDB_ONLINE:
                init_args: dict[str, Any] = { 
                    "config" : asdict(trainer),
                    "mode" : "online" if self.type is Logger.Type.WANDB_ONLINE else "offline",
                }
                if self.project_name != "":
                    init_args |= { "project" : self.project_name }
                if self.run_name != "":
                    init_args |= { "name" : self.run_name }

                self.wandb_run = wandb.init(**init_args)
            case Logger.Type.CONSOLE:
                pass
            case _: ...

    def log(self, step: int, data: dict[str, Any]):
        if step % self.log_freq != 0:
            return

        match self.type:
            case Logger.Type.WANDB_OFFLINE | Logger.Type.WANDB_ONLINE:
                if getattr(self, "wandb_run", None) is None:
                    raise UninitializedError(f"Logger cannot save logs to wandb without starting a wandb run!")

                wandb.log(
                    data=data,
                    step=step,
                    commit=True
                )
            case _:
                raise NotImplementedError(f"Logging is not supported for logger type {self.type}")

    def checkpoint(self, step: int, model: TrainableModel, optim: optim.Optimizer):
        if self.checkpoint_freq > 0 and step % self.checkpoint_freq != 0:
            return

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'steps' : step,
        }

        # save locally
        os.makedirs(self.write_dir, exist_ok=True)
        checkpoint_name = f"checkpoint_{step}"
        checkpoint_filename = os.path.join(self.write_dir, checkpoint_name)
        torch_save(checkpoint, checkpoint_filename)

        match self.type:
            case Logger.Type.WANDB_OFFLINE | Logger.Type.WANDB_ONLINE:
                if getattr(self, "wandb_run", None) is None:
                    raise UninitializedError(f"Logger cannot save checkpoints to wandb without starting a wandb run!")

                artifact = wandb.Artifact(
                    name=f"{model.name}.{step}.{self.run_name}",
                    type=f"{model.name}_model",
                    metadata=model.metadata
                )

                artifact.add_file(
                    checkpoint_filename,
                    name=checkpoint_name,
                    policy="immutable"
                )

                wandb.log_artifact(
                    # self.wandb_run,
                    artifact,
                    artifact.name,
                    artifact.type
                )
            case Logger.Type.CONSOLE:
                print(f"Checkpoint saved at {step} steps")


@dataclass
class Trainer(YAMLObject):

    yaml_tag = u'!Trainer'

    function_class: FunctionClass
    optimizer: partial[optim.Optimizer]
    loss_fn: nn.Module
    steps: int
    model: TrainableModel
    logger: Logger
    # prediction_indices: list[int] = field(default_factory=lambda: [-1]) # implement this as a custom loss_fn

    def train(self, skip_steps: int = 0, validate_shapes: bool = True, progress_bar: bool = False):
        if getattr(self, "optim", None) is None:
            self.model.setup()
            self.optim = self.optimizer(self.model.parameters())

        self.logger.start(trainer=self)

        pbar: "tqdm[Any] | None" = None
        if progress_bar:
            from tqdm import trange
            pbar = trange(self.steps, unit="step")
            pbar.update(skip_steps)

        for i in range(skip_steps, self.steps):
            x_batch, y_batch = self.function_class.sample_at_step(step_num=i)

            if validate_shapes: # TODO: make sure this is actually sensical
                x_shape_err_msg = " \n\t".join([
                    "", "Expected: (batch_size, seq_len, x_dim)",
                    f"Got: {', '.join(map(str, x_batch.shape))}"
                ])
                y_shape_err_msg = " \n\t".join([
                    "", "Expected: (batch_size, seq_len, x_dim)",
                    f"Got: {', '.join(map(str, y_batch.shape))}"
                ])
            
            
                if len(x_batch.shape) < 3:
                    raise ShapeError("Not enough x dimensions" + x_shape_err_msg)
                elif len(x_batch.shape) > 3:
                    raise ShapeError("Too many x dimensions!" + x_shape_err_msg)
                
                if len(y_batch.shape) < 3:
                    raise ShapeError("Not enough y dimensions" + y_shape_err_msg)
                elif len(y_batch.shape) > 3:
                    raise ShapeError("Too many y dimensions!" + y_shape_err_msg)
                
                x_bs, x_seq, x_dim = x_batch.shape
                y_bs, y_seq, y_dim = y_batch.shape
                
                if x_bs != y_bs:
                    raise ShapeError(
                        f"Batch sizes for x and y don't match! x: {x_bs}, y: {y_bs}"
                    )
                if x_seq not in (y_seq, y_seq+1):
                    raise ShapeError(" \n\t".join([
                        f"Unexpected number of y values!",
                        f"Expected: {(x_seq, x_seq-1)}",
                        f"Got: {y_seq}"
                    ]))

                if x_dim < y_dim:
                    raise ShapeError( 
                        "y dimension cannot be larger than x dimension! "
                        + f"Got: x_dim = {x_dim}, y_dim = {y_dim}" 
                    )
                
                if x_seq not in (y_seq, y_seq + 1):
                    raise ValueError(
                        "Can only inference with x sequences either 1 longer or as long as y sequences!" + \
                        f"Got: X sequences of length {x_seq} and Y sequences of lengh {y_seq}"
                    )

            output = self.model.evaluate(x_batch, y_batch)

            if validate_shapes and output.shape != y_batch.shape:
                raise ValueError(
                    f"Model {self.model.name} produced ill-shaped predictions!"
                    + f"Expected: {y_batch.shape}    Got: {output.shape}"
                )

            loss = self.loss_fn(output, y_batch)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if progress_bar and pbar is not None:
                pbar.update(1)
                pbar.set_description(f"loss {loss}")

            self.logger.log(step=i, data={
                "overall_loss": loss,
            })

            self.logger.checkpoint(step=i, model=self.model, optim=self.optim)

        if progress_bar and pbar is not None:
            pbar.close()

        return self.model
