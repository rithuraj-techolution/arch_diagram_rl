import cv2
import json
import numpy as np
import tensorflow as tf

class DiagramEnvironment:
    def __init__(self, layout_json):
        print("Initializing Diagram Environment...")
        self.original_layout = layout_json
        self.current_layout = json.loads(json.dumps(layout_json))
        self.action_space = [-20, -10, -5, -1, 0, 1, 5, 10, 20]
        self.canvas_size = (720, 405)

        self.max_state_length = 500
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self._fit_tokenizer()
        
        self.state = self._extract_state()

    def _preprocess_layout(self):
        """Optionally preprocess the layout if needed."""
        # This can include standardizing positions, ensuring all nodes and groups have valid attributes, etc.
        pass

    def _fit_tokenizer(self):
        """Fit the tokenizer on node labels, group names, and edge sources/targets."""
        print("Fitting tokenizer...")
        texts = []

        texts.extend(node["data"]["label"] for node in self.current_layout["nodes"])
        texts.extend(self.original_layout["groups"].keys())
        texts.extend(edge["source"] for edge in self.current_layout["edges"])
        texts.extend(edge["target"] for edge in self.current_layout["edges"])

        self.tokenizer.fit_on_texts(texts)

    def _extract_state(self):
        """Extract and tokenize state representation."""
        nodes = self.current_layout["nodes"]
        groups = self.current_layout["groups"]
        edges = self.current_layout["edges"]

        state = []
        for node in nodes:
            node_label_token = self.tokenizer.texts_to_sequences([node["data"]["label"]])[0][:1]

            state.extend(node_label_token)
            state.extend([node["position"]["x"], node["position"]["y"], node["width"], node["height"]])

        for group_name, group in groups.items():
            group_name_token = self.tokenizer.texts_to_sequences([group_name])[0][:1]

            state.extend(group_name_token)
            state.extend([group["x"], group["y"], group["width"], group["height"]])
        
        for edge in edges:
            source_token = self.tokenizer.texts_to_sequences([edge["source"]])[0][:1]
            target_token = self.tokenizer.texts_to_sequences([edge["target"]])[0][:1]

            state.extend(source_token + target_token)
            state.extend([edge["connectionSite_start"], edge["connectionSite_end"]])

        state = state[:self.max_state_length] + [0] * (self.max_state_length - len(state))

        return np.array(state, dtype=np.float32)

    def _calculate_reward(self):
        """Calculate reward based on overlaps, overflow, and group containment."""
        nodes = self.current_layout["nodes"]
        edges = self.current_layout["edges"]

        reward = 0

        # Penalize overlaps
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j and self._check_overlap(node1, node2):
                    reward -= 30

        # Reward for no overlaps
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j and not self._check_overlap(node1, node2):
                    reward += 15

        # Penalize overflow
        for node in nodes:
            if self._check_overflow(node):
                reward -= 50
        
        # Reward for no overflow
        for node in nodes:
            if not self._check_overflow(node):
                reward += 15

        # Reward for nodes staying within their assigned groups
        reward += self._check_nodes_within_groups()

        # Penalize for moving nodes far from their original positions
        for node in nodes:
            original_node = next(n for n in self.original_layout["nodes"] if n["data"]["label"] == node["data"]["label"])
            original_x, original_y = original_node["position"]["x"], original_node["position"]["y"]
            current_x, current_y = node["position"]["x"], node["position"]["y"]

            distance = np.sqrt((current_x - original_x) ** 2 + (current_y - original_y) ** 2)
            reward -= distance * 1.5

        # Penalize for groups and nodes overlapping
        for group in self.current_layout["groups"].values():
            for node in nodes:
                if self._check_group_and_node_overlap(group, node):
                    reward -= 40
        
        # Reward for groups and nodes not overlapping
        for group in self.current_layout["groups"].values():
            for node in nodes:
                if not self._check_group_and_node_overlap(group, node):
                    reward += 15

        return reward
    
    def _check_group_and_node_overlap(self, group, node):
        """Check if a node overlaps with a group."""
        x1, y1 = group["x"], group["y"]
        x2, y2 = node["position"]["x"], node["position"]["y"]
        w1, h1 = group["width"], group["height"]
        w2, h2 = node["width"], node["height"]

        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def _check_overlap(self, node1, node2):
        """Check if two nodes overlap."""
        x1, y1 = node1["position"]["x"], node1["position"]["y"]
        x2, y2 = node2["position"]["x"], node2["position"]["y"]
        w1, h1 = node1["width"], node1["height"]
        w2, h2 = node2["width"], node2["height"]

        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def _check_overflow(self, node):
        """Check if a node is outside the canvas boundaries."""
        x, y = node["position"]["x"], node["position"]["y"]
        w, h = node["width"], node["height"]

        return x < 0 or x + w > self.canvas_size[0] or y < 0 or y + h > self.canvas_size[1]

    def _check_nodes_within_groups(self):
        """Check if nodes are within their respective groups."""
        groups = self.current_layout["groups"]
        nodes = self.current_layout["nodes"]

        reward = 0
        for node in nodes:
            group_name = node["data"].get("group", None)
            if group_name and group_name in groups:
                group = groups[group_name]
                node_x, node_y = node["position"]["x"], node["position"]["y"]
                node_width, node_height = node["width"], node["height"]

                # Check if the node is within the group boundaries
                if (
                    node_x >= group["x"] and
                    node_x + node_width <= group["x"] + group["width"] and
                    node_y >= group["y"] and
                    node_y + node_height <= group["y"] + group["height"]
                ):
                    reward += 15  # Reward for being inside the group
                else:
                    reward -= 40  # Penalty for being outside the group

        return reward
    
    def _calculate_connection_point(self, source_node, target_node, edge):
        start_x, start_y = 0, 0
        end_x, end_y = 0, 0

        connection_start = edge["connectionSite_start"]
        connection_end = edge["connectionSite_end"]

        if connection_start == 0:
            start_x = source_node["position"]["x"] + source_node["width"] // 2
            start_y = source_node["position"]["y"]
        elif connection_start == 1:
            start_x = source_node["position"]["x"]
            start_y = source_node["position"]["y"] + source_node["height"] // 2
        elif connection_start == 2:
            start_x = source_node["position"]["x"] + source_node["width"] // 2
            start_y = source_node["position"]["y"] + source_node["height"]
        elif connection_start == 3:
            start_x = source_node["position"]["x"] + source_node["width"]
            start_y = source_node["position"]["y"] + source_node["height"] // 2

        if connection_end == 0:
            end_x = target_node["position"]["x"] + target_node["width"] // 2
            end_y = target_node["position"]["y"]
        elif connection_end == 1:
            end_x = target_node["position"]["x"]
            end_y = target_node["position"]["y"] + target_node["height"] // 2
        elif connection_end == 2:
            end_x = target_node["position"]["x"] + target_node["width"] // 2
            end_y = target_node["position"]["y"] + target_node["height"]
        elif connection_end == 3:
            end_x = target_node["position"]["x"] + target_node["width"]
            end_y = target_node["position"]["y"] + target_node["height"] // 2

        return start_x, start_y, end_x, end_y
    
    def _check_edges_distance(self):
        """Calculate rewards based on the distance between connected nodes."""
        edges = self.current_layout["edges"]
        reward = 0
        for edge in edges:
            source_node = next(n for n in self.current_layout["nodes"] if n["data"]["label"] == edge["source"])
            target_node = next(n for n in self.current_layout["nodes"] if n["data"]["label"] == edge["target"])

            start_x, start_y, end_x, end_y = self._calculate_connection_point(source_node, target_node, edge)
            distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

            if distance < 100:
                reward += 20
            else:
                reward -= 40
        
        return reward


    def step(self, action):
        """Apply an action to a node and calculate the reward."""
        node_index, dx, dy = action
        node = self.current_layout["nodes"][node_index]

        # Adjust node position
        node["position"]["x"] += dx
        node["position"]["y"] += dy

        # Calculate reward
        reward = self._calculate_reward()

        # Update state
        self.state = self._extract_state()
        return self.state, reward

    def reset(self):
        """Reset the layout to its original state."""
        self.current_layout = json.loads(json.dumps(self.original_layout))
        self.state = self._extract_state()

    def render(self, window_name="Diagram Layout"):
        """Render the current layout using OpenCV."""
        canvas = np.ones((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8) * 255  # White canvas
        nodes = self.current_layout["nodes"]
        edges = self.current_layout["edges"]
        groups = self.current_layout["groups"]

        # Draw groups as rectangles with a light background color
        for group_name, group in groups.items():
            x, y, w, h = group["x"], group["y"], group["width"], group["height"]
            color = (200, 200, 200)  # Light gray
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
            cv2.putText(canvas, group_name, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

        # Draw nodes as rectangles with labels
        for node in nodes:
            x, y, w, h = node["position"]["x"], node["position"]["y"], node["width"], node["height"]
            label = node["data"]["label"]
            color = (100, 100, 255)  # Light blue
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
            cv2.putText(canvas, label, (x + 5, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the canvas
        cv2.imshow(window_name, canvas)
        cv2.waitKey(1)  

    def save_to_json(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.current_layout, f, indent=2)
        print(f"Updated layout saved to {output_path}")
