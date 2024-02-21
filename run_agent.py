from mtx_to_array import mtx_to_array
from stable_baselines3 import PPO
from Environment import GraphEnv

matrix = mtx_to_array("mtx_files/arc130.mtx")
env = GraphEnv(matrix)
env.reset() # Has to be called before every learn
model = PPO("MlpPolicy", env, verbose=1)


model.learn(total_timesteps=2)

