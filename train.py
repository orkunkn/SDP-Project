
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
model_path = f"{models_dir}/MaskablePPO.zip"
#model = MaskablePPO.load(model_path, env=env)

model = MaskablePPO("MlpPolicy", env, verbose=1, learning_rate=0.00005, n_steps=2048)

model.learn(total_timesteps=2048*30)

# env.render()

# To save the model after learning
model.save(f"{models_dir}/MaskablePPO")


# 0.5 dene

"""-----------------------------------------------
|            | Before   | After    | Change   |
-----------------------------------------------
| AIL        | 157      | 499      | +218   % |
| ARL        | 8.24     | 16.8     | +103.7 % |
| ALC        | 306      | 981      | +220.9 % |
| Total Cost | 4.07e+05 | 6.42e+05 | +57.55 % |
| Level      | 1.33e+03 | 654      | -50.9  % |
-----------------------------------------------
-----------------------------------------------
|            | Before   | After    | Change   |
-----------------------------------------------
| AIL        | 111      | 404      | +263.9 % |
| ARL        | 5.07     | 10.9     | +114.7 % |
| ALC        | 217      | 796      | +267.4 % |
| Total Cost | 1.09e+06 | 1.87e+06 | +71.16 % |
| Level      | 5.03e+03 | 2.34e+03 | -53.42 % |
-----------------------------------------------"""