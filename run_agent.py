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

# gd97_a
# ex1
# fs_183_6
matrix = mtx_to_array("new/fs_183_6.mtx")

env = GraphEnv(matrix)

env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# To load and use a previously educated model
model_path = f"{models_dir}/MaskablePPO_medium.zip"
model = MaskablePPO.load(model_path, env=env)

# Initialize the environment and get the starting observation
observation, _ = env.reset()
env.render()
print("AIR", env.AIR, "ARL", env.ARL, "ALC:", env.ALC)
# Number of steps to run
num_steps = 1000
for step in range(num_steps):
    action, _states = model.predict(observation, deterministic=True, action_masks=env.valid_action_mask())
    observation, reward, done, _, info = env.step(action)
    
    # Render the environment
    if done:
        print("AIR:", env.AIR, "ARL", env.ARL, "ALC:", env.ALC)
        env.render()
        break
        # observation, _ = env.reset()
