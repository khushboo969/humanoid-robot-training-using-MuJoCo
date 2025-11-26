"""
ppo_mujoco_v5_fixed.py
Corrected beginner-friendly script to train/evaluate PPO on MuJoCo v5 environments:
 - Humanoid-v5       -> walking
 - HumanoidStandup-v5-> stand-up
 - Hopper-v5         -> hopping

Fixes included:
- Wrap evaluation env the same way as training env (VecNormalize) to avoid SB3 sync errors.
- Use SB3 v1.8+ net_arch format (dict(pi=..., vf=...)).
- make_env accepts render_mode to produce matching envs for training/play.
"""

import argparse
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

def make_env(env_id, render_mode=None):
    """
    Helper to create an env factory for DummyVecEnv.
    render_mode=None during training; render_mode="human" for play.
    """
    def _init():
        # pass render_mode explicitly for gymnasium MuJoCo v5
        if render_mode is None:
            env = gym.make(env_id, render_mode=None)
        else:
            env = gym.make(env_id, render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init

def train(env_id: str, total_timesteps: int, save_dir: str, num_envs: int = 1):
    os.makedirs(save_dir, exist_ok=True)

    # Create training vectorized envs and normalize (same wrapper used for training)
    train_vec = DummyVecEnv([make_env(env_id, render_mode=None) for _ in range(num_envs)])
    train_vec = VecNormalize(train_vec, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Policy architecture (SB3 v1.8+ requires dict format)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        train_vec,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048 if num_envs == 1 else 1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.0,
        clip_range=0.2,
        gae_lambda=0.95,
        policy_kwargs=policy_kwargs,
    )

    # Checkpoint callback: save regularly
    checkpoint_cb = CheckpointCallback(save_freq=max(1, total_timesteps // 10),
                                       save_path=save_dir,
                                       name_prefix=f"{env_id.replace('-', '_')}_ckpt")

    # Create eval env that is wrapped the same way as training env to avoid sync errors
    eval_vec = DummyVecEnv([make_env(env_id, render_mode=None)])
    # Wrap eval vector env with VecNormalize; set training=False so stats are not updated
    eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=True, clip_obs=10.)
    eval_vec.training = False  # ensure eval doesn't update running stats

    eval_cb = EvalCallback(eval_vec, best_model_save_path=save_dir,
                           log_path=save_dir, eval_freq=max(1, total_timesteps // 20),
                           deterministic=True, render=False)

    if total_timesteps > 0:
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])

        # Save final model
        model_path = os.path.join(save_dir, f"{env_id.replace('-', '_')}_ppo_{max(1, total_timesteps//1000)}k.zip")
        model.save(model_path)
        print("Saved model to:", model_path)

        # Save VecNormalize statistics (important for proper evaluation / play)
        vecnorm_path = os.path.join(save_dir, "vecnormalize.pkl")
        # train_vec is a VecNormalize wrapper instance, so save it
        train_vec.save(vecnorm_path)
        print("Saved VecNormalize to:", vecnorm_path)
    else:
        print("No training requested (timesteps=0). Skipping training.")

    # close envs
    train_vec.close()
    eval_vec.close()

def play(env_id: str, model_path: str, vecnorm_path: str = None, render: bool = True, play_episodes: int = 5):
    """
    Play a saved model. If vecnormalize stats exist, load them to normalize observations the same way
    they were during training.
    """
    # If vecnormalize exists, load it into a DummyVecEnv (headless) and attach to the model.
    if vecnorm_path is not None and os.path.exists(vecnorm_path):
        # create a dummy vec env (no rendering) to load VecNormalize object
        dummy = DummyVecEnv([make_env(env_id, render_mode=None)])
        vecnorm = VecNormalize.load(vecnorm_path, dummy)
        vecnorm.training = False
        vecnorm.norm_reward = False

        # Load the model and set the normalized env for prediction
        model = PPO.load(model_path, env=vecnorm)
        # For rendering we will use the underlying (unwrapped) env set to human rendering
        render_env = gym.make(env_id, render_mode="human")
        render_env = Monitor(render_env)
    else:
        # No vecnorm — use simple rendering env and load model with a dummy env for SB3 internals
        render_env = gym.make(env_id, render_mode="human")
        render_env = Monitor(render_env)
        wrapped_env = DummyVecEnv([make_env(env_id, render_mode=None)])
        model = PPO.load(model_path, env=wrapped_env)

    for ep in range(play_episodes):
        obs, info = render_env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while True:
            # If model env is VecNormalize we must feed the model observations in the same shape.
            # Using model.predict with the raw obs works because SB3 handles single env arrays internally.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = render_env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                print(f"Episode {ep+1} finished - reward: {total_reward:.2f} steps: {steps}")
                break

    render_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Humanoid-v5",
                        help="Environment id (e.g. Humanoid-v5, HumanoidStandup-v5, Hopper-v5)")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Number of training timesteps (0 to skip training)")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs (1 for beginners/CPU)")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models/checkpoints")
    parser.add_argument("--play", action="store_true", help="Skip training and play a saved model")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model to load for play")
    parser.add_argument("--render", action="store_true", help="Render during play")
    args = parser.parse_args()

    if args.play:
        if args.model is None:
            raise SystemExit("When --play is set, provide --model path to a saved .zip model")
        vecnorm_file = os.path.join(os.path.dirname(args.model), "vecnormalize.pkl")
        play(args.env, args.model, vecnorm_file if os.path.exists(vecnorm_file) else None, render=args.render)
    else:
        print("Training PPO on", args.env)
        train(args.env, args.timesteps, args.save_dir, num_envs=args.num_envs)
