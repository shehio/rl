"""Modal runner for GPU training of the Stable Baselines3 Atari agents.

Wraps atari/baselines/atari_baseline_train.py so the exact local CLI runs on an
A10G worker. Checkpoints persist in the "rl-atari-models" Modal volume, and
wandb logging works through the "wandb" Modal secret (set WANDB_API_KEY there).

Usage:
    modal run modal_app.py --algorithm ppo --timesteps 1000000 --env ALE/Pong-v5
"""

import pathlib

import modal

LOCAL_ROOT = pathlib.Path(__file__).parent
REMOTE_ROOT = "/root/rl"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1",
        "stable-baselines3==2.7.0",
        "gymnasium[atari]==1.2.0",
        "ale-py==0.11.1",
        "opencv-python-headless==4.10.0.84",
        "tensorboard==2.17.0",
        "wandb",
    )
    .add_local_dir(
        LOCAL_ROOT / "atari" / "baselines",
        remote_path=f"{REMOTE_ROOT}/atari/baselines",
    )
)

app = modal.App("rl-atari")

models_volume = modal.Volume.from_name("rl-atari-models", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 8,
    secrets=[modal.Secret.from_name("wandb")],
    volumes={f"{REMOTE_ROOT}/atari/models": models_volume},
)
def train(
    algorithm: str = "ppo",
    timesteps: str = "100000",
    env: str = "ALE/Pong-v5",
    n_envs: int = 8,
    seed: int = 0,
    no_wandb: bool = False,
):
    """Run the baseline training script on a GPU worker."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "atari_baseline_train.py",
        "--algorithm",
        algorithm,
        "--timesteps",
        timesteps,
        "--env",
        env,
        "--n_envs",
        str(n_envs),
        "--seed",
        str(seed),
    ]
    if no_wandb:
        cmd.append("--no-wandb")

    subprocess.run(cmd, cwd=f"{REMOTE_ROOT}/atari/baselines", check=True)
    models_volume.commit()


@app.local_entrypoint()
def main(
    algorithm: str = "ppo",
    timesteps: str = "100000",
    env: str = "ALE/Pong-v5",
    n_envs: int = 8,
    seed: int = 0,
    no_wandb: bool = False,
):
    """Mirror of the local atari_baseline_train.py CLI."""
    train.remote(
        algorithm=algorithm,
        timesteps=timesteps,
        env=env,
        n_envs=n_envs,
        seed=seed,
        no_wandb=no_wandb,
    )
