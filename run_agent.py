from mtx_to_array import mtx_to_array
from stable_baselines3 import PPO
from callback import PrintInfoCallback
from Environment import GraphEnv

matrix = mtx_to_array("bcsstk17.mtx")
env = GraphEnv(matrix)
env.reset()
model = PPO("MlpPolicy", env, verbose=1)

callback = PrintInfoCallback()
model.learn(total_timesteps=2, callback=callback)

