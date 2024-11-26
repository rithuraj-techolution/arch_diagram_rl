import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


class CurriculumLearningAgent:
    def __init__(self, env, model_name, logs_dir, actor_lr=0.0001, critic_lr=0.0002, gamma=0.95, curriculum_steps=5):
        print("Setting up Curriculum Learning Agent....")
        self.env = env
        self.gamma = gamma
        self.model_name = model_name
        self.logs_dir = logs_dir

        # Actor and Critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        # Curriculum parameters
        self.curriculum_steps = curriculum_steps
        self.current_level = 0

        # Exploration parameter
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

        # Metrics tracking
        self.episode_rewards = []
        self.losses = []

    def _build_actor(self):
        """Build the actor network."""
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.env.max_state_length,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(len(self.env.action_space) * len(self.env.current_layout["nodes"]), activation="softmax")
        ])
        return model

    def _build_critic(self):
        """Build the critic network."""
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.env.max_state_length,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(1)  # State value
        ])
        return model

    def _choose_action(self, state):
        """Choose an action using epsilon-greedy with the actor's policy."""
        state = (state - np.mean(state)) / (np.std(state) + 1e-10)  # Normalize state
        action_probs = self.actor.predict(state.reshape(1, -1), verbose=0).flatten()

        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(len(action_probs))  # Exploration
        else:
            action_index = np.random.choice(len(action_probs), p=action_probs)  # Exploitation

        node_index = action_index // len(self.env.action_space)
        action_offset = action_index % len(self.env.action_space)
        dx = self.env.action_space[action_offset]
        dy = self.env.action_space[action_offset]
        return node_index, dx, dy

    def _train_step(self, state, action, reward, next_state, done):
        """Perform a single training step for both actor and critic."""
        state = (state - np.mean(state)) / (np.std(state) + 1e-10)  # Normalize state
        next_state = (next_state - np.mean(next_state)) / (np.std(next_state) + 1e-10)

        with tf.GradientTape(persistent=True) as tape:
            # Predict values
            state_value = self.critic(state.reshape(1, -1), training=True)
            next_state_value = self.critic(next_state.reshape(1, -1), training=True)

            # Calculate target and advantage
            target = reward + (1 - done) * self.gamma * next_state_value
            advantage = target - state_value

            # Critic loss (Mean Squared Error)
            critic_loss = tf.reduce_mean(tf.square(advantage))

            # Actor loss (-log_prob * advantage)
            action_probs = self.actor(state.reshape(1, -1), training=True)
            action_index = (action[0] * len(self.env.action_space)) + self.env.action_space.index(action[1])
            log_prob = tf.math.log(action_probs[0, action_index] + 1e-10)
            actor_loss = -log_prob * tf.stop_gradient(advantage)

        # Apply gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in actor_grads]  # Gradient clipping
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in critic_grads]

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return actor_loss.numpy(), critic_loss.numpy()

    def adjust_environment(self):
        """Adjust environment complexity based on the current curriculum level."""
        print(f"Adjusting environment to curriculum level {self.current_level}...")
        # Example adjustments for curriculum levels
        if self.current_level == 0:
            self.env.max_nodes = 3
            self.env.add_constraints = False
        elif self.current_level == 1:
            self.env.max_nodes = 5
            self.env.add_constraints = False
        elif self.current_level == 2:
            self.env.max_nodes = 10
            self.env.add_constraints = True

    def _calculate_threshold(self, rewards):
        """Calculate the threshold based on the rewards."""
        return 0.9 * np.mean(rewards)  # Stricter threshold for curriculum progression

    def train(self, episodes=50):
        """Train the agent with curriculum learning."""
        for level in range(self.curriculum_steps):
            self.current_level = level
            self.adjust_environment()
            total_rewards = []

            for episode in range(episodes):
                self.env.reset()
                total_reward = 0
                state = self.env.state

                for step in range(50):  # Max steps per episode
                    action = self._choose_action(state)
                    next_state, reward = self.env.step(action)
                    done = step == 49  # End of episode condition
                    total_reward += reward

                    # Train step
                    actor_loss, critic_loss = self._train_step(state, action, reward, next_state, done)
                    state = next_state

                total_rewards.append(total_reward)
                print(f"Level {level}, Episode {episode + 1}: Reward = {total_reward:.2f}")

            avg_reward = np.mean(total_rewards)
            print(f"Average Reward for Level {level}: {avg_reward:.2f}")

            threshold = self._calculate_threshold(total_rewards)
            print(f"Calculated Threshold for Level {level}: {threshold:.2f}")

            # Check threshold to decide curriculum progression
            if avg_reward < threshold:
                print(f"Level {level} performance too low, retrying level.")
                self.current_level -= 1  # Retry the level

            self.episode_rewards.extend(total_rewards)  # Store rewards for plotting

            with open(os.path.join(self.logs_dir, f"{self.model_name}_metrics.txt"), "a") as f:
                f.write(f"Level {level}: Average Reward = {avg_reward:.2f}\n")

        print("Curriculum Learning Complete!")
        self._plot_metrics()

    def _plot_metrics(self):
        """Plot metrics after training."""
        rewards = np.array(self.episode_rewards)
        plt.figure(figsize=(12, 5))
        plt.plot(rewards, label="Total Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Total Rewards")
        plt.title("Total Rewards per Episode")
        plt.legend()
        plt.grid()
        plt.show()

    def save_model(self, path):
        """Save the trained model."""
        actor_path = os.path.join(path, f"{self.model_name}_actor.keras")
        print(f"Actor model saved to {actor_path}")
        self.actor.save(actor_path)
        
        critic_path = os.path.join(path, f"{self.model_name}_critic.keras")
        print(f"Critic model saved to {critic_path}")
        self.critic.save(critic_path)
        
    def load_model(self, actor_path, critic_path):
        """Load trained models."""
        self.actor = tf.keras.models.load_model(actor_path)
        self.critic = tf.keras.models.load_model(critic_path)
        print(f"Actor model loaded from {actor_path}")
        print(f"Critic model loaded from {critic_path}")
