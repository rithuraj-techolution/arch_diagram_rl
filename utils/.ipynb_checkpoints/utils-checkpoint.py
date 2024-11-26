import os
import cv2
import numpy as np
from utils.Enums import Enums

from utils.agents.ActorCriticAgent import ActorCriticAgent
from utils.agents.ArchitectureV1Agent import ArchitectureV1Agent
from utils.agents.CurriculumLearningAgent import CurriculumLearningAgent
from utils.agents.SoftmaxActionAgent import SoftmaxActionAgent

from utils.environments.DiagramEnvironment import DiagramEnvironment
from utils.environments.DiagramEnvironmentZone import DiagramEnvironmentZone

def map_agent(agent_name, env, model_name):
    agent_mapper = {
        "actor-critic" : ActorCriticAgent(env, model_name, Enums.LOGS_DIRECTORY.value),
        "architecture-v1" : ArchitectureV1Agent(env, model_name, Enums.LOGS_DIRECTORY.value),
        "curriculum" : CurriculumLearningAgent(env, model_name, Enums.LOGS_DIRECTORY.value),
        "softmax" : SoftmaxActionAgent(env, model_name, Enums.LOGS_DIRECTORY.value)
    }
    if agent_name in agent_mapper:
        return agent_mapper[agent_name]
    else:
        return agent_mapper[Enums.DEFAULT_AGENT.value]
    
def map_environment(env_name, layout_data):
    if env_name == "diagram-raw":
        return DiagramEnvironment(layout_data)
    elif env_name == "diagram-zone":
        return DiagramEnvironmentZone(layout_data)
    else:
        return map_environment(Enums.DEFAULT_ENVIRONMENT.value, layout_data)

def map_model(model_name):
    model_names = os.listdir(Enums.MODEL_DIRECTORY.value)
    model_names = [os.path.splitext(model)[0] for model in model_names]
    model_mapper = {}
    for model in model_names:
        h5_path = os.path.join(Enums.MODEL_DIRECTORY.value, model + ".h5")
        keras_path = os.path.join(Enums.MODEL_DIRECTORY.value, model + ".keras")
        if os.path.exists(h5_path):
            model_mapper[model] = h5_path
        elif os.path.exists(keras_path):
            model_mapper[model] = keras_path


    if model_name in model_mapper:
        return model_mapper[model_name]
    else:
        return model_mapper[Enums.DEFAULT_MODEL.value]
    
def get_image(json_data):
    image = 255 * np.ones((405, 720, 3), dtype=np.uint8)
    nodes = json_data["nodes"]
    groups = json_data["groups"]

    for group_name, group in groups.items():
        x, y, w, h = group["x"], group["y"], group["width"], group["height"]
        color = (200, 200, 200)  # Light gray
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
        cv2.putText(image, group_name, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

    for node in nodes:
        x, y, w, h = node["position"]["x"], node["position"]["y"], node["width"], node["height"]
        label = node["data"]["label"]
        color = (100, 100, 255)  # Light blue
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
        cv2.putText(image, label, (x + 5, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image