from mtx_conversions import mtx_to_array, graph_to_mtx
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from Environment import GraphEnv
import os

def mask_fn(env):
    return env.unwrapper.valid_action_mask()

models_dir = "models"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

mtx_name = "bcsstk17"
matrix = mtx_to_array(f"mtx_files/{mtx_name}.mtx")

env = GraphEnv(matrix, mode="run")

env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# To load and use a previously educated model
model_path = f"{models_dir}/MaskablePPO.zip"
model = MaskablePPO.load(model_path, env=env)

# Initialize the environment and get the starting observation
observation, _ = env.reset()
# env.render()

first_ail = env.unwrapped.AIL
first_arl = env.unwrapped.ARL
first_alc = env.unwrapped.ALC
first_air = env.unwrapped.AIR
first_total_cost = sum(env.unwrapped.level_costs)
first_level = len(env.unwrapped.levels)

# Directory containing mtx files
mtx_directory = "mtx_files"

print("Running model for", mtx_name)

# Iterate over all mtx files in the directory
while True:
    action, _states = model.predict(observation, deterministic=True, action_masks=env.unwrapped.valid_action_mask())
    observation, reward, done, _, info = env.step(action)

    if done:
        new_ail = env.unwrapped.AIL
        new_arl = env.unwrapped.ARL
        new_alc = env.unwrapped.ALC
        new_air = env.unwrapped.AIR
        new_total_cost = sum(env.unwrapped.level_costs)
        new_level = len(env.unwrapped.levels)

        total_cost_change = ((new_total_cost - first_total_cost) / first_total_cost) * 100
        ail_change = ((new_ail - first_ail) / first_ail) * 100
        arl_change = ((new_arl - first_arl) / first_arl) * 100
        alc_change = ((new_alc - first_alc) / first_alc) * 100
        air_change = ((new_air - first_air) / first_air) * 100
        level_change = ((new_level - first_level) / first_level) * 100

        # Create a formatted table
        print("-" * 47)
        print("| {:<10} | {:<8} | {:<8} | {:<8} |".format("", "Before", "After", "Change"))
        print("-" * 47)
        print("| {:<10} | {:<8.3g} | {:<8.3g} | {:<+7.4g}% |".format("AIL", first_ail, new_ail, ail_change))
        print("| {:<10} | {:<8.3g} | {:<8.3g} | {:<+7.4g}% |".format("ARL", first_arl, new_arl, arl_change))
        print("| {:<10} | {:<8.3g} | {:<8.3g} | {:<+7.4g}% |".format("ALC", first_alc, new_alc, alc_change))
        print("| {:<10} | {:<8.3g} | {:<8.3g} | {:<+7.4g}% |".format("AIR", first_air, new_air, air_change))
        print("| {:<10} | {:<8.3g} | {:<8.3g} | {:<+7.4g}% |".format("Total Cost", first_total_cost, new_total_cost, total_cost_change))
        print("| {:<10} | {:<8.3g} | {:<8.3g} | {:<+7.4g}% |".format("Level", first_level, new_level, level_change))
        print("-" * 47)

        graph_to_mtx(env.unwrapped.G, f"{mtx_name}")
        # env.render()
        break
        # observation, _ = env.reset()
