import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_dim: int, action_dim: int, learning_rate=0.1, epsilon=0.1, decay_rate=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.q_table = {}
        self.best_action = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.best_reward = -float("inf")

    def choose_action(self, state):
        if random.random() < self.epsilon:
            a = np.random.uniform(0.5, 2.0, self.action_dim)
        else:
            a = self.best_action + np.random.normal(0, 0.1, self.action_dim)
        return np.clip(a, 0.5, 2.0).astype(np.float32)

    def update(self, state, action, reward, next_state):
        if reward > self.best_reward:
            self.best_reward = float(reward)
            self.best_action = action.copy()
        self.epsilon = max(0.01, self.epsilon * self.decay_rate)
