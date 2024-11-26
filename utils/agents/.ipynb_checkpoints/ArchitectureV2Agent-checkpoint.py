import os
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class ArchitectureV2Agent:
    def __init__(self, env, model_name, logs_dir, model=None):
        self.env = env
        self.model = model or self._build_model()
        self.epsilon = 0.7  # Exploration-exploitation trade-off
        self.epsilon_decay = 0.8
        self.epsilon_min = 0.1
        self.gamma = 0.99
        self.model_name = model_name
        self.logs_dir = logs_dir

        # Initialize lists to track metrics
        self.episode_rewards = []
        self.losses = []
    
    def _choose_action(self, state):
        if np.random.rand() < self.epsilon:
            print("Exploring")
            node_index = random.randint(0, len(self.env.current_layout["nodes"]) - 1)
            dx = random.choice(self.env.action_space)
            dy = random.choice(self.env.action_space)
            return node_index, dx, dy
        else:
            print("Exploiting")
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)
            action_index = np.argmax(q_values)
            node_index = action_index // len(self.env.action_space)
            action_offset = action_index % len(self.env.action_space)
            dx = self.env.action_space[action_offset]
            dy = self.env.action_space[action_offset]
            return node_index, dx, dy

    def _train_model(self, state, action, reward, next_state):
        """Train the neural network using the Bellman equation and return the loss."""
        node_index, dx, dy = action
        action_index = (node_index * len(self.env.action_space)) + self.env.action_space.index(dx)

        # Predict Q-values for current state
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        # Predict Q-values for next state
        next_q_values = self.model.predict(next_state.reshape(1, -1), verbose=0)

        # Update Q-value for the taken action
        target_q = reward + self.gamma * np.max(next_q_values)
        loss = (q_values[0][action_index] - target_q) ** 2
        q_values[0][action_index] = target_q

        # Train the model
        self.model.fit(state.reshape(1, -1), q_values, verbose=0)

        return loss  # Return the loss for logging

    def save_model(self, filepath):
        filepath = os.path.join(filepath, f"{self.model_name}.h5")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def _build_model(self):
        """Build an enhanced convolutional and dense neural network model."""
        # Input dimensions
        input_shape = (self.env.max_state_length,)

        # Define the model
        inputs = tf.keras.layers.Input(shape=input_shape)

        # Reshape input for convolutional processing
        x = tf.keras.layers.Reshape((self.env.max_state_length, 1))(inputs)

        # Convolutional layers to extract spatial relationships
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation="relu", padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)  # Correctly chain the BatchNormalization layer
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

        x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)  # Correctly chain the BatchNormalization layer
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

        # Flatten the output for dense processing
        x = tf.keras.layers.Flatten()(x)

        # Fully connected layers for action-value computation
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)  # Correctly chain the BatchNormalization layer
        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)  # Correctly chain the BatchNormalization layer
        x = tf.keras.layers.Dropout(0.3)(x)

        # Output layer for action values (Q-values)
        outputs = tf.keras.layers.Dense(
            len(self.env.action_space) * len(self.env.current_layout["nodes"]),
            activation="linear"
        )(x)

        # Compile the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.Huber()
        )
        return model


    def train(self, episodes=100):
        """Train the agent over a specified number of episodes."""
        for episode in range(episodes):
            self.env.reset()
            total_reward = 0
            episode_loss = 0

            for step in range(50):  # Max steps per episode
                state = self.env.state
                action = self._choose_action(state)
                next_state, reward = self.env.step(action)
                total_reward += reward

                # Train the model and log loss
                loss = self._train_model(state, action, reward, next_state)
                episode_loss += loss

                # Reduce epsilon to encourage exploitation over exploration
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

            # Log metrics at the end of each episode
            self.episode_rewards.append(total_reward)
            self.losses.append(episode_loss / 50)  # Average loss per step
            
            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Average Loss = {episode_loss / 50}\n")
            # Write episode metrics to a log file
            with open(os.path.join(self.logs_dir, f"{self.model_name}.txt"), "a") as f:
                f.write(f"Episode {episode + 1}: Total Reward = {total_reward}, Average Loss = {episode_loss / 50}\n")
            
            self.env.render()

        # Plot the metrics at the end of training
        # self._plot_metrics()


    def _choose_action(self, state):
        """Choose an action based on the epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:  # Explore
            node_index = random.randint(0, len(self.env.current_layout["nodes"]) - 1)
            dx = random.choice(self.env.action_space)
            dy = random.choice(self.env.action_space)
            return node_index, dx, dy
        else:  # Exploit
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)
            action_index = np.argmax(q_values)
            node_index = action_index // len(self.env.action_space)
            action_offset = action_index % len(self.env.action_space)
            dx = self.env.action_space[action_offset]
            dy = self.env.action_space[action_offset]
            return node_index, dx, dy

    def _train_model(self, state, action, reward, next_state):
        """Train the neural network using the Bellman equation."""
        node_index, dx, dy = action
        action_index = (node_index * len(self.env.action_space)) + self.env.action_space.index(dx)

        # Predict Q-values for current state
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        # Predict Q-values for next state
        next_q_values = self.model.predict(next_state.reshape(1, -1), verbose=0)

        # Update Q-value for the taken action
        target_q = reward + self.gamma * np.max(next_q_values)
        q_values[0][action_index] = target_q

        # Train the model
        trained_model = self.model.fit(state.reshape(1, -1), q_values, verbose=0)

        loss = trained_model.history["loss"][0]

        return loss
        


    def load_model(self, filepath):
        """Load a previously saved model."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


    def _plot_metrics(self):
        """Plot metrics for training evaluation."""
        # Plot total rewards per episode
        plt.figure(figsize=(12, 5))
        plt.plot(self.episode_rewards, label="Total Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Total Rewards")
        plt.title("Total Rewards per Episode")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot average loss per episode
        plt.figure(figsize=(12, 5))
        plt.plot(self.losses, label="Average Loss", color='orange')
        plt.xlabel("Episodes")
        plt.ylabel("Average Loss")
        plt.title("Average Loss per Episode")
        plt.legend()
        plt.grid()
        plt.show()
