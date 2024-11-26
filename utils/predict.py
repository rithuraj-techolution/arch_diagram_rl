import numpy as np

def predict_model(model, env, episodes=50, steps=50):
    print("Predicting using normal model....")
    for episode in range(episodes):
        env.reset()
        for step in range(steps):
            state = env.state
            q_values = model.predict(state.reshape(1, -1), verbose=0)
            action_index = np.argmax(q_values)
            
            node_index = action_index // len(env.action_space)
            action_offset = action_index % len(env.action_space)
            dx = env.action_space[action_offset]
            dy = env.action_space[action_offset]
            
            action = (node_index, dx, dy)
            env.step(action)
            print("Predicting", end="")
            print("...", end="\r")
    
    return env.current_layout
        

def predict_actor(model, env, episodes=50, steps=50):
    print("Predicting using actor model....")
    for episode in range(episodes):
        env.reset()
        for step in range(steps):  # Max steps per episode
            state = env.state
            state = (state - np.mean(state)) / (np.std(state) + 1e-10)  # Normalize state

            # Predict action probabilities from the actor model
            action_probs = model.predict(state.reshape(1, -1), verbose=0).flatten()
            action_index = np.argmax(action_probs)

            # Decode action index into (node_index, dx, dy)
            node_index = action_index // len(env.action_space)
            action_offset = action_index % len(env.action_space)
            dx = env.action_space[action_offset]
            dy = env.action_space[action_offset]

            action = (node_index, dx, dy)
            env.step(action)
            print("Predicting", end="")
            print("...", end="\r")

    return env.current_layout