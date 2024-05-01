from mtx_to_array import mtx_to_array
from stable_baselines3 import PPO
from Environment import GraphEnv
import os


models_dir = "models/PPO"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


matrix = mtx_to_array("mtx_files/pr02r.mtx")
env = GraphEnv(matrix)


# To load and use a previously educated model
model_path = f"{models_dir}/PPO.zip"
# model = PPO.load(model_path, env=env)


model = PPO("MlpPolicy", env, verbose=1,
            learning_rate=2.5e-4,
            clip_range=0.25,
            ent_coef=0.01)

# model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2048*100)


# env.render()

# To save the model after learning
model.save(f"{models_dir}/PPO")
