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

# Directory containing mtx files
mtx_directory = "mtx_files"

# Iterate over all mtx files in the directory
for filename in os.listdir(mtx_directory):
    if filename.endswith(".mtx"):
        mtx_name = filename[:-4]  # Remove the '.mtx' extension
        matrix = mtx_to_array(f"{mtx_directory}/{filename}")
        env = GraphEnv(matrix)
        env = ActionMasker(env, mask_fn)  # Wrap to enable masking

        if "MaskablePPO.zip" in os.listdir(models_dir):
            print("---- Using existing model. ----")
            model = MaskablePPO.load(f"{models_dir}/MaskablePPO.zip", env=env)
        else:
            print("---- Created new model. ----")
            model = MaskablePPO("MlpPolicy", env, verbose=1, learning_rate=0.00015, n_steps=2048)

        print(f"---- Starting training for {filename} ----")
        model.learn(total_timesteps=2048*100)

        # Save the model after learning
        model.save(f"{models_dir}/MaskablePPO.zip")
        
        print(f"---- Model trained and saved for {filename} ----")
