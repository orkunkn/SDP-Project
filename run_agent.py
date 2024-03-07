from mtx_to_array import mtx_to_array
from stable_baselines3 import PPO
from Environment_thin import GraphEnv
import os

"""
models_dir = "models/PPO"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
"""

matrix = mtx_to_array("mtx_files/bcsstk17.mtx")
env = GraphEnv(matrix)
env.reset() # Has to be called before every learn

"""
# To load and use a previously educated model
model_path = f"{models_dir}/PPO.zip"
model = PPO.load(model_path, env=env)
"""

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
# env.render()
"""
# To save the model after learning
model.save(f"{models_dir}/PPO")
"""