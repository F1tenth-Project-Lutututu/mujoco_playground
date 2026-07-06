import os
import functools
import json
import inspect


# Set necessary environment variables for JAX/MuJoCo performance
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import wandb

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

from experiment_launcher import single_experiment, run_experiment

#DEBUG = len([k for k in os.environ.keys() if "DEBUG" in k.upper()]) > 0
DEBUG = False


def get_ppo_config(env_name: str, impl: str, vision: bool):
    """Returns the tuned MuJoCo Playground PPO config for an environment."""
    if env_name in mujoco_playground.manipulation._envs:
        if vision:
            return manipulation_params.brax_vision_ppo_config(env_name, impl)
        return manipulation_params.brax_ppo_config(env_name, impl)
    if env_name in mujoco_playground.locomotion._envs:
        if vision:
            raise ValueError(f"Vision PPO is not configured for {env_name}.")
        return locomotion_params.brax_ppo_config(env_name, impl)
    if env_name in mujoco_playground.dm_control_suite._envs:
        if vision:
            return dm_control_suite_params.brax_vision_ppo_config(env_name, impl)
        return dm_control_suite_params.brax_ppo_config(env_name, impl)

    raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def filter_train_params(train_params):
    """Drops config keys unsupported by the installed Brax PPO train function."""
    valid_keys = set(inspect.signature(ppo.train).parameters)
    return {k: v for k, v in train_params.items() if k in valid_keys}


@single_experiment
def main(
    env_name: str = "Go1JoystickFlatTerrain",
    impl: str = "jax",
    vision: bool = False,
    domain_randomization: bool = False,
    playground_config_overrides: str = "",
    results_dir: str = "./results",
    seed: int = 1,
):
    # Load default environment config and PPO hyperparameters
    env_cfg = registry.get_default_config(env_name)
    ppo_config = get_ppo_config(env_name, impl, vision)
    ppo_config.num_evals = 200

    env_cfg_overrides = {"impl": impl}
    if vision:
        env_cfg_overrides["vision"] = True
        env_cfg_overrides["vision_config.nworld"] = ppo_config.num_envs
    if playground_config_overrides:
        env_cfg_overrides.update(json.loads(playground_config_overrides))

    ppo_config.seed = seed
    
    # Create a unique experiment name
    group_name = f"{env_name}_ppo_{impl}"
    if domain_randomization:
        group_name += "_dr"
    if vision:
        group_name += "_vision"
    run_name = f"{group_name}_seed{seed}"
    ckpt_dir = os.path.abspath(os.path.join(results_dir, "checkpoints", run_name))
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Starting training run: {run_name}")
    print(f"Environment config overrides: {env_cfg_overrides}")
    print(f"PPO config: {ppo_config}")
    print(f"Checkpoints directory: {ckpt_dir}")

    # 2. Initialize Weights & Biases
    wandb.init(
        project=f"spectral-playground-{env_name}",
        name=run_name,
        group=group_name,
        config={
            "env_config": env_cfg.to_dict(),
            "env_config_overrides": env_cfg_overrides,
            "ppo_config": ppo_config.to_dict(),
            "env_name": env_name,
            "impl": impl,
            "vision": vision,
            "domain_randomization": domain_randomization,
        },
        mode="disabled" if DEBUG else "online",
    )

    # 3. Setup Environment and Randomization
    # Load the environment
    env = registry.load(env_name, config=env_cfg, config_overrides=env_cfg_overrides)
    # Load the evaluation environment (usually same config, but distinct instance)
    eval_env_overrides = dict(env_cfg_overrides)
    if vision:
        eval_env_overrides["vision_config.nworld"] = ppo_config.get("num_eval_envs", 128)
    eval_env = registry.load(
        env_name,
        config=registry.get_default_config(env_name),
        config_overrides=eval_env_overrides,
    )
    
    randomizer_fn = None
    if domain_randomization:
        randomizer_fn = registry.get_domain_randomizer(env_name)

    # 4. Setup Network Factory
    # Helper to create PPO networks based on config
    network_fn = (
        ppo_networks_vision.make_ppo_networks_vision
        if vision
        else ppo_networks.make_ppo_networks
    )
    if hasattr(ppo_config, "network_factory"):
        network_factory = functools.partial(
            network_fn,
            **ppo_config.network_factory
        )
    else:
        network_factory = network_fn

    # 5. Define Callbacks
    
    # Progress callback: logs to wandb and prints to console
    def progress_fn(num_steps, metrics):
        wandb.log(metrics, step=num_steps)
        if "eval/episode_reward" in metrics:
            print(f"Step {num_steps}: Reward = {metrics['eval/episode_reward']:.3f}")
        elif "episode/sum_reward" in metrics:
            print(f"Step {num_steps}: Reward = {metrics['episode/sum_reward']:.3f}")
        else:
            print(f"Step {num_steps}: logged {len(metrics)} metrics")

    # Save config for reproducibility.
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(env_cfg.to_dict(), f, indent=4)

    # 6. Start Training
    # We extract the training arguments from the ConfigDict
    train_params = dict(ppo_config)
    if "network_factory" in train_params:
        del train_params["network_factory"]
    train_params = filter_train_params(train_params)

    print("JIT compiling and starting training loop...")

    train_kwargs = {
        "environment": env,
        "eval_env": eval_env,
        "network_factory": network_factory,
        "progress_fn": progress_fn,
        "wrap_env_fn": wrapper.wrap_for_brax_training,
        **train_params,
    }
    if "save_checkpoint_path" in inspect.signature(ppo.train).parameters:
        train_kwargs["save_checkpoint_path"] = ckpt_dir
    if randomizer_fn is not None:
        train_kwargs["randomization_fn"] = randomizer_fn
    if "vision" in inspect.signature(ppo.train).parameters:
        train_kwargs["vision"] = vision
    
    make_inference_fn, params, _ = ppo.train(**train_kwargs)

    print(f"Training complete. Model saved to {ckpt_dir}")
    wandb.finish()

if __name__ == "__main__":
    run_experiment(main)
