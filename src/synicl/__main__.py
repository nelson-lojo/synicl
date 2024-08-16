import argparse
import os

from pathlib import Path

import yaml, torch

from . import *

def compile_yaml(conffile: str, includedir: None | str) -> str:
    included = ""
    if includedir is not None:
        for file in Path(includedir).rglob("*.yml"):
            with open(file, "r") as f:
                included += f.read() + "\n"

    with open(conffile, 'r') as f:
        lines = f.readlines()

    lines = [ ' '*4 + line for line in lines ]
    content = f"train: !Trainer\n" + "\n".join(lines)

    full_yaml = included.strip() + '\n\n' + content
    return full_yaml

def parse_cli_args() -> argparse.Namespace:
    conf_include_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conf", "include")
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", '-c', type=str, default="", action='store', dest="conffile", required=True,
                        help="Path to the config file to use for training")
    parser.add_argument("--resume", '-r', type=str, default="", action='store', dest="checkpointfile",
                        help="Path to the checkpoint to resume training with")
    parser.add_argument("--wandb-project", type=str, default="", dest="projectname",
                        help="The project to log to in weights and biases")
    parser.add_argument("--run-name", type=str, default="", dest="runname",
                        help="What to name this run in wandb")
    parser.add_argument("--include-dir", type=str, default=conf_include_dir, action='store', dest="includedir",
                        help="From what directory to include configurations")
    parser.add_argument("--validate-shapes", type=bool, default=True, dest="validate",
                       help="Validate tensor shapes during execution")
    parser.add_argument("--progress", type=bool, default=False, dest="progbar",
                        help="Show a progress bar")
    args = parser.parse_args()
    return args

def load_checkpoint(checkpointfile: str, trainer: Trainer) -> tuple[Trainer, int]:
    checkpoint_dict = torch.load( checkpointfile, # see train.base.Logger for keys in this dict
                                  map_location=("cuda" if torch.cuda.is_available() else "cpu"))
    model_state = checkpoint_dict['model_state_dict']
    optim_state = checkpoint_dict['optimizer_state_dict']
    latest_step = checkpoint_dict['steps']

    trainer.model.load_state_dict(model_state)
    trainer.optim = trainer.optimizer(trainer.model.parameters())
    trainer.optim.load_state_dict(optim_state)

    return trainer, latest_step

def main():

    args = parse_cli_args()
    
    full_yaml = compile_yaml(args.conffile, args.includedir)

    trainer = yaml.load(full_yaml, yaml.Loader)['train']
    latest_step = 0

    if args.checkpointfile != "":
        trainer, latest_step = load_checkpoint(args.checkpointfile, trainer)

    import code
    code.interact(local=locals())
    exit(0)
    trainer.train(skip_steps=latest_step, validate_shapes=args.validate, progress_bar=args.progbar)


if __name__ == "__main__":
    main()
