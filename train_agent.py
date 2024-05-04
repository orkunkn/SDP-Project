from mtx_to_array import mtx_to_array
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from Environment import GraphEnv
import os

def mask_fn(env):
    return env.valid_action_mask()

models_dir = "models/PPO"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

matrix = mtx_to_array("current_model_mtx/gd97_a.mtx")

env = GraphEnv(matrix)

env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# To load and use a previously educated model
model_path = f"{models_dir}/MaskablePPO_medium.zip"
model = MaskablePPO.load(model_path, env=env)

#model = MaskablePPO("MlpPolicy", env, verbose=1, learning_rate=0.00025, n_steps=16)

model.learn(total_timesteps=2048*2)

# env.render()

# To save the model after learning
model.save(f"{models_dir}/MaskablePPO_medium")
