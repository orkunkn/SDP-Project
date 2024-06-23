RL Agent Optimizing SpTRSV
==========

This repository contains scripts to optimize a lower triangular
sparse matrix to use in calculations using reinforcement learning.

Overview
--------

This project aims to optimize a lower triangular sparse matrix for
a faster calculation in several areas. This model uses MaskablePPO
from sb3-contrib library which provides the PPO algorithm with an
option to mask the action space.

The idea is determining the thin ideas in the graph using several
conditions and removing the thin levels until a point.

Environment.py creates a custom environment for a graph conversion
of the matrix and taking actions such as moving a node by the agent.
This model uses a fixed action space and environment space length,
so for the bigger matrices the lengths can be changed in this file
but this also extends the training time. Action space length is 
determined by the level count of the graph.

Environment has 2 modes: train, run. In train mode, graph weights
are not used since it slows down the process a lot and the program
does not use it. Weights are only required for the recreation of
the matrix so they are used in run mode.

actions.py, constructor.py and graph.py are the helper functions
for the environment. 

Actions consists of 2 different actions whichare moving to the
next thin level or the next level. Most graph values are updated
here after the action.

Constructor is used for updating the metrics used for finding the
thin levels and then finding the current thin levels. Thin levels
stops updating after reaching a certain percantage of the initial
first thin level count. It can be updated here.

Graph is used only at the beginning. It creates the initial graph
and initial values for the environment metrics.

mtx_conversions.py is used for converting the mtx file into a csr
and converting a networkx graph to mtx file. Also .bin files for
the graph is created for the matrix calculations. Graph to mtx
function is only used in run modes and can be disabled by commenting
out the "graph_to_mtx" function.

train.py and full_train.py is used for training a model.
full_train runs for all matrices in mtx_files.
To train with a single matrix, you can use train.py and modify
the matrix name in the code.

run.py and full_run.py is used for running the model with a 
previously trained model. full_run runs for all matrices in mtx_files.
To run for a single matrix, you can use run.py and modify the matrix
name in the code.

Dependencies
------------

- All code is written in Python 3.
- numpy
- sb3-contrib
- networkx
- scipy
- matplotlib
- tensorboard (Optional for RL metrics, can be disabled by deleting "tensorboard_logs" from MaskablePPO)

Description of files
--------------------

Non-Python files:

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
README.md                         |  Text file (markdown format) description of the project.
mtx_files/*.mtx                   |  Matrix files

Python files:

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
Environment.py                    |  Custom environment for an RL agent.
actions.py                        |  Apply the actions chosen by agent.
constructor.py                    |  Update the graph metrics and thin levels.
graph.py                          |  Initialize the graph and it's metrics.
mtx_conversions.py                |  Convert a mtx file to csr format and a graph to csr and bin format.
run.py                            |  Run a previously trained model.
full_run.py                       |  Run a previously trained model for the all matrices in mtx_files.
train.py                          |  Train a model.
full_train.py                     |  Train a model for the all matrices in mtx_files.