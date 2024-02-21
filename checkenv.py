from stable_baselines3.common.env_checker import check_env
from mtx_to_array import mtx_to_array
from Environment import GraphEnv

matrix = mtx_to_array("bcsstk17.mtx")
env = GraphEnv(matrix)
# This will check the environment and output additional warnings if needed
check_env(env)