"""EXPERIMENTAL: run synicl training jobs on Modal (cloud GPUs).

This is an entrypoint for `modal run`; it does NOT touch the core package.
It copies `src/` into the container and sets PYTHONPATH (so the package is
importable without a build step — avoids the missing README/setuptools issue),
installs the runtime deps with pip (cross-platform: torch==2.2.0 from PyPI is
the CUDA 12.1 build on Linux x86_64, so it Just Works on Modal GPUs and also
runs on CPU), mounts a Volume at /root/run_data so checkpoints persist across
runs, streams the config file in as text so any local config can be used, and
ships any `--include-dir` YAML fragments the same way (relpaths preserved so
the CLI's `rglob("*.yml")` behaves identically on the remote path).

Setup (one-time):
    pip install modal  # or: uv add --dev modal  (already done in this repo)
    modal setup        # authenticate / pick a workspace

    # only needed if your config uses WANDB_ONLINE logging:
    modal secret create wandb WANDB_API_KEY=<your-key>

Run a job (from the repo root):
    modal run modal_app.py --config run_conf.yml
    modal run modal_app.py --config run_conf.yml --gpu A100
    modal run modal_app.py --config run_conf.yml --gpu none        # CPU
    modal run modal_app.py --config run_conf.yml --wandb-secret wandb
    modal run modal_app.py --config run_conf.yml --resume checkpoint_50000
    modal run modal_app.py --config run_conf.yml --include-dir conf/include

Notes / limitations (experimental):
  * Checkpoints are written to the `synicl-run-data` Volume at /root/run_data.
    The config's `write_dir: "run_data"` resolves there because the function's
    cwd is /root. To list/download them: `modal volume ls synicl-run-data`.
  * `--resume <name>` is interpreted relative to /root/run_data.
  * `--include-dir` is supported: the local include dir is shipped to the
    container at /root/includes (relpaths preserved) and passed to the CLI.
    Omit it to fall back to the CLI's default (<package>/../conf/include),
    which is an empty no-op in the container.
  * Modal's max function timeout is 24h. For the default 500k-step run, you'll
    likely need to chunk it: stop, then `--resume` from the latest checkpoint.
  * wandb offline logs are written to the ephemeral /root/wandb and are lost
    when the container exits; use WANDB_ONLINE + --wandb-secret for persistence.
"""

from __future__ import annotations

import pathlib

import modal

APP_NAME = "synicl"
SRC_REMOTE = "/root/synicl/src"
CONFIG_REMOTE = "/root/config.yml"
RUNDATA_REMOTE = "/root/run_data"
INCLUDES_REMOTE = "/root/includes"

# ignore files we never want in the image
_IGNORE = lambda p: "__pycache__" in p.parts or ".venv" in p.parts

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.2.0",
        "pyyaml==6.0.1",
        "tqdm==4.66.5",
        "transformers[torch]==4.39.3",
        "wandb==0.17.6",
    )
    .add_local_dir("src", SRC_REMOTE, ignore=_IGNORE)
    .env({"PYTHONPATH": SRC_REMOTE})
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name("synicl-run-data", create_if_missing=True)


@app.function(
    volumes={RUNDATA_REMOTE: vol},
    timeout=24 * 60 * 60,  # Modal max; chunk long runs with --resume
)
def train(
    config_text: str,
    includes: dict[str, str] | None = None,
    resume: str = "",
    validate_shapes: bool = True,
    progress_bar: bool = False,
) -> None:
    """Run `python -m synicl -c <config>` inside the container.

    `includes` maps a relative path (as it lived under the local --include-dir)
    to file contents; it is materialized at INCLUDES_REMOTE so the CLI's
    `Path(--include-dir).rglob("*.yml")` behaves identically to a local run.
    """
    import os
    import pathlib
    import subprocess
    import sys

    with open(CONFIG_REMOTE, "w") as f:
        f.write(config_text)

    include_dir_arg: str | None = None
    if includes:
        for rel, content in includes.items():
            p = pathlib.Path(INCLUDES_REMOTE) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        include_dir_arg = INCLUDES_REMOTE

    cmd = [sys.executable, "-m", "synicl", "-c", CONFIG_REMOTE]
    if include_dir_arg is not None:
        cmd += ["--include-dir", include_dir_arg]
    if resume:
        cmd += ["--resume", os.path.join(RUNDATA_REMOTE, resume)]
    cmd += ["--progress-bar"] if progress_bar else ["--no-progress-bar"]
    cmd += ["--validate-shapes"] if validate_shapes else ["--no-validate-shapes"]

    try:
        subprocess.run(cmd, check=True)
    finally:
        # persist checkpoint files to the volume even if the run crashes
        vol.commit()


@app.local_entrypoint()
def main(
    config: str,
    resume: str = "",
    validate_shapes: bool = True,
    progress_bar: bool = False,
    gpu: str = "T4",
    wandb_secret: str = "",
    include_dir: str = "",
) -> None:
    """Local entrypoint: read the config + includes locally, dispatch to the cloud."""
    config_text = pathlib.Path(config).read_text()
    includes = _gather_includes(include_dir)
    secrets = [modal.Secret.from_name(wandb_secret)] if wandb_secret else []
    gpu_value: str | None = None if gpu.lower() == "none" else gpu

    train.with_options(gpu=gpu_value, secrets=secrets).remote(
        config_text,
        includes=includes or None,
        resume=resume,
        validate_shapes=validate_shapes,
        progress_bar=progress_bar,
    )


def _gather_includes(include_dir: str) -> dict[str, str]:
    """Read every `*.yml` under `include_dir` into {relpath: content}.

    Mirrors `__main__.compile_yaml`'s `rglob("*.yml")` so the remote side, once
    it materializes these files under INCLUDES_REMOTE, concatenates them in the
    same way a local run would.
    """
    if not include_dir:
        return {}
    base = pathlib.Path(include_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"--include-dir not found: {include_dir}")
    return {
        str(p.relative_to(base)): p.read_text()
        for p in base.rglob("*.yml")
    }
