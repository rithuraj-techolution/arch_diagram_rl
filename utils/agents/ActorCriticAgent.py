import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class ActorCriticAgent:
    def __init__(self, env, model_name, logs_dir, actor_lr=0.001, critic_lr=0.002, gamma=0.99):
        print("Setting up Actor Critic Agent...")
        self.env = env
        self.model_name = model_name
        self.logs_dir = logs_dir
        self.gamma = gamma

        # Actor: Outputs action probabilities
        self.actor = self._build_actor()

        # Critic: Outputs state value
        self.critic = self._build_critic()

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        # Metrics tracking
        self.episode_rewards = []
        self.losses = []

        # Create logs directory if it doesn't exist
        os.makedirs(self.logs_dir, exist_ok=True)

    def _build_actor(self):
        """Build the actor network."""
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.env.max_state_length,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(len(self.env.action_space) * len(self.env.current_layout["nodes"]), activation="softmax")
        ])
        return model

    def _build_critic(self):
        """Build the critic network."""
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.env.max_state_length,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1)  # State value
        ])
        return model

    def _choose_action(self, state):
        """Choose an action using the actor's policy."""
        action_probs = self.actor.predict(state.reshape(1, -1), verbose=0).flatten()
        action_index = np.random.choice(len(action_probs), p=action_probs)
        node_index = action_index // len(self.env.action_space)
        action_offset = action_index % len(self.env.action_space)
        dx = self.env.action_space[action_offset]
        dy = self.env.action_space[action_offset]
        return node_index, dx, dy

    def _train_step(self, state, action, reward, next_state, done):
        """Perform a single training step for both actor and critic."""
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
            actor_loss = -log_prob * advantage

        # Apply gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return actor_loss.numpy(), critic_loss.numpy()

    def train(self, episodes=100):
        """Train the agent over multiple episodes."""
        for episode in range(episodes):
            self.env.reset()
            total_reward = 0
            state = self.env.state

            episode_actor_loss = 0
            episode_critic_loss = 0

            for step in range(50):  # Max steps per episode
                action = self._choose_action(state)
                next_state, reward = self.env.step(action)
                done = step == 49  # End of episode condition
                total_reward += reward

                # Train step
                actor_loss, critic_loss = self._train_step(state, action, reward, next_state, done)
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss

                state = next_state

            # Log metrics
            self.episode_rewards.append(total_reward)
            self.losses.append((episode_actor_loss / 50, episode_critic_loss / 50))  # Average losses
            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Actor Loss = {episode_actor_loss / 50}, Critic Loss = {episode_critic_loss / 50:.4f}\n")
            
            with open(os.path.join(self.logs_dir, f"{self.model_name}_metrics.txt"), "a") as f:
                f.write(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Actor Loss = {episode_actor_loss / 50}, Critic Loss = {episode_critic_loss / 50:.4f}\n")

        # Plot metrics
        # self._plot_metrics()
        
    def save_model(self, path):
        """Save the trained model."""
        actor_path = os.path.join(path, f"{self.model_name}_actor.keras")
        print(f"Actor model saved to {actor_path}")
        self.actor.save(actor_path)
        
        critic_path = os.path.join(path, f"{self.model_name}_critic.keras")
        print(f"Critic model saved to {critic_path}")
        self.critic.save(critic_path)