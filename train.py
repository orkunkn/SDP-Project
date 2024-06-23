
from mtx_conversions import mtx_to_array
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from Environment import GraphEnv
import os

def mask_fn(env):
    return env.valid_action_mask()

models_dir = "models"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

mtx_name = "bcsstk17"
matrix = mtx_to_array(f"mtx_files/{mtx_name}.mtx")

env = GraphEnv(matrix)

env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# To load and use a previously educated model
# model_path = f"{models_dir}/MaskablePPO.zip"
# model = MaskablePPO.load(model_path, env=env)

model = MaskablePPO("MlpPolicy", env, verbose=1, learning_rate=0.0001, n_steps=2048)

model.learn(total_timesteps=2048*30)

# To save the model after learning
model.save(f"{models_dir}/MaskablePPO")
