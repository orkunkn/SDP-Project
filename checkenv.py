from stable_baselines3.common.env_checker import check_env
from mtx_to_array import mtx_to_array
from Environment_thin import GraphEnv

""" This test should be run after every change to environment. no output = good to go """

matrix = mtx_to_array("mtx_files/494_bus.mtx")
env = GraphEnv(matrix)
# This will check the environment and output additional warnings if needed
check_env(env)