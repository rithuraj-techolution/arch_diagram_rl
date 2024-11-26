import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class SoftmaxActionAgent:
    def __init__(self, env, model_name, logs_dir, model=None):
        print("Setting up Softmax Action Agent.....")
        self.env = env
        self.model = model or self._build_model()
        self.epsilon = 1.0  # Exploration-exploitation trade-off
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.gamma = 0.99  # Discount factor
        self.model_name = model_name
        self.logs_dir = logs_dir

        self.episode_rewards = []
        self.losses = []

    def _build_model(self):
        """Build a simple neural network model."""
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.env.max_state_length,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(len(self.env.action_space) * len(self.env.current_layout["nodes"]), activation="linear")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        return model

    def save_model(self, filepath):
        filepath = os.path.join(filepath, f"{self.model_name}.h5")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def _choose_action(self, state):
        """Choose an action using epsilon-greedy with a robust Softmax policy."""
        # Predict Q-values
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]

        if np.random.rand() < self.epsilon:  # Exploration
            node_index = random.randint(0, len(self.env.current_layout["nodes"]) - 1)
            dx = random.choice(self.env.action_space)
            dy = random.choice(self.env.action_space)
            return node_index, dx, dy

        # Exploitation using Softmax
        # Stabilize Softmax by subtracting max Q-value
        q_values -= np.max(q_values)
        exp_q_values = np.exp(q_values)
        softmax_probs = exp_q_values / (np.sum(exp_q_values) + 1e-10)  # Add small epsilon to avoid division by zero

        if np.any(np.isnan(softmax_probs)):
            print("Warning: Softmax probabilities are NaN. Using uniform distribution.")
            softmax_probs = np.ones_like(q_values) / len(q_values)

        # Sample action based on probabilities
        action_index = np.random.choice(len(q_values), p=softmax_probs)
        node_index = action_index // len(self.env.action_space)
        action_offset = action_index % len(self.env.action_space)
        dx = self.env.action_space[action_offset]
        dy = self.env.action_space[action_offset]

        return node_index, dx, dy

    def _train_model(self, state, action, reward, next_state):
        """Train the model using the Bellman equation."""
        node_index, dx, dy = action
        action_index = (node_index * len(self.env.action_space)) + self.env.action_space.index(dx)

        # Predict current and next Q-values
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        next_q_values = self.model.predict(next_state.reshape(1, -1), verbose=0)

        # Compute target Q-value
        target_q = reward + self.gamma * np.max(next_q_values)
        q_values[0][action_index] = target_q

        # Train the model
        self.model.fit(state.reshape(1, -1), q_values, verbose=0)

    def train(self, episodes=100):
        """Train the agent."""
        for episode in range(episodes):
            self.env.reset()
            total_reward = 0

            for step in range(50):  # Max steps per episode
                state = self.env.state

                try:
                    action = self._choose_action(state)
                except ValueError as e:
                    print("Error in action selection:", e)
                    print("State:", state)
                    continue

                next_state, reward = self.env.step(action)
                total_reward += reward

                try:
                    self._train_model(state, action, reward, next_state)
                except ValueError as e:
                    print("Error in training:", e)
                    print("State:", state)
                    print("Action:", action)
                    print("Next State:", next_state)
                    continue

            # Update epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            # Log episode results
            self.episode_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        self._plot_metrics()

    def _plot_metrics(self):
        """Plot training metrics."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, label="Total Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Total Rewards")
        plt.title("Training Rewards Over Episodes")
        plt.legend()
        plt.grid()
        plt.show()

    def save_model(self):
        """Save the trained model."""
        path = os.path.join(self.logs_dir, f"{self.model_name}.h5")
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load a trained model."""
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
