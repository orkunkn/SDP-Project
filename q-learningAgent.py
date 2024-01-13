

class QLearningAgent():
    pass

def bellman_optimality(V, transitions, gamma):
    for s in range(len(V)):
        V[s] = max(sum(p * (r + gamma * V[next_state]) for p, next_state, r in transitions[s][a]) for a in transitions[s])