
LEVEL_NUM = 0
TOTAL_EPISODES = 100
LEARNING_RATE = 0.05 # alpha in the literature
DISCOUNT = 0.95 # gamma IN the literature
EPSILON = 0.1
START_EPSILON_DECAYING = 150
END_EPSILON_DECAYING = 600
epsilon_decay_value = EPSILON/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

class QLearningNetwork():
    pass