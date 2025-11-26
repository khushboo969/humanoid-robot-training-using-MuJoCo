# PPO MuJoCo Training & Playback (Stable-Baselines3)

A clean, beginner-friendly script to train and evaluate **PPO** agents
on **MuJoCo v5** environments using **Stable-Baselines3**.

This script fixes common pitfalls (VecNormalize mismatch, evaluation
wrapper errors, rendering issues) and provides a plug-and-play setup
for:

-   **Humanoid-v5**
-   **HumanoidStandup-v5**
-   **Hopper-v5**

------------------------------------------------------------------------

## 📁 Features

-   Fully working **PPO training loop**
-   Correct **VecNormalize** usage for training & eval
-   Automated checkpoints
-   Evaluation callbacks
-   Clean playback mode with rendering
-   Flexible CLI interface

------------------------------------------------------------------------

## 📦 Requirements

``` bash
pip install stable-baselines3==2.3.0
pip install gymnasium[mujoco]
pip install mujoco
```

Ubuntu users may also need:

``` bash
sudo apt-get install libosmesa6-dev
```

------------------------------------------------------------------------

## 🚀 Training

Example (200k steps):

``` bash
python humanoid.py --env Humanoid-v5 --timesteps 200000 --save_dir models
```

------------------------------------------------------------------------

## 🎮 Playing a Saved Model

``` bash
python humanoid.py --play --env Humanoid-v5 --model models/Humanoid_v5_ppo_200k.zip --render
```

------------------------------------------------------------------------

## 🛠 Tips

-   Train ≥ **1M timesteps** for real results
-   Keep `vecnormalize.pkl` with the model
-   Use more envs only if CPU allows

------------------------------------------------------------------------

## 📄 License

MIT License.
