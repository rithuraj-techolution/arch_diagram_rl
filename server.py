import os
import json
import cv2
import uuid
import tensorflow as tf
from flask import Flask, request, jsonify

from utils.predict import predict_model, predict_actor
from utils.utils import get_image, map_agent, map_environment, map_model
from utils.rlef_utils import upload_to_rlef
from utils.Enums import Enums

app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
    return "I am healthy!"

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    project_name = data.get("project_name", None)
    model_name = data.get("model_name", None)
    env_name = data.get("environment_name", None)
    json_data = data.get("architecture_json", None)
        
    prediction_uid = uuid.uuid4()[:4]
    if json_data is not None and type(json_data) != dict:
        json_data = json.loads(json_data)
    elif type(json_data) != dict:
        return jsonify({"error" : "Architecture JSON data not found"}), 400
    
    original_image = get_image(json_data)
    original_image_temp_path = os.path.join(Enums.TEMP_DATA_DIRECTORY.value, "original_image.jpg")
    cv2.imwrite(original_image_temp_path, original_image)
    print(original_image.shape)

    res = upload_to_rlef(Enums.LLM_GENERATED_MODEL_ID.value, original_image_temp_path, json_data, project_name, model_name, prediction_uid)
    if res == 200:
        print("Original Image uploaded to RLEF successfully")
    else:
        print("Original Image upload to RLEF failed")
    
    model_path = map_model(model_name)
    custom_object = {"mse" : tf.keras.losses.MeanSquaredError}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_object)
    
    env = map_environment(env_name, json_data)
    print("Using model path : ", model_path)
    if "actor" in model_name.lower():
        optimized_json = predict_actor(model, env)

        optimized_image = get_image(optimized_json)
        optimized_image_temp_path = os.path.join(Enums.TEMP_DATA_DIRECTORY.value, "optimized_image.jpg")
        cv2.imwrite(optimized_image_temp_path, optimized_image)
        print(optimized_image.shape)
        res = upload_to_rlef(Enums.RL_OPTIMIZED_MODEL_ID.value, optimized_image_temp_path, optimized_json, project_name, model_name, prediction_uid)

        return jsonify({"optimized_json" : optimized_json}), 200
    else:
        optimized_json = predict_model(model, env)

        optimized_image = get_image(optimized_json)
        optimized_image_temp_path = os.path.join(Enums.TEMP_DATA_DIRECTORY.value, "optimized_image.jpg")
        cv2.imwrite(optimized_image_temp_path, optimized_image)
        print(optimized_image.shape)
        res = upload_to_rlef(Enums.RL_OPTIMIZED_MODEL_ID.value, optimized_image_temp_path, optimized_json, project_name, model_name)
        
        if res == 200:
            print("Optimized Image uploaded to RLEF successfully")
        else:
            print("Optimized Image upload to RLEF failed")

        return jsonify({"optimized_json" : optimized_json}), 200

@app.route("/send_feedback", methods=['POST'])
def send_feedback():
    data = request.get_json()
    project_name = data.get("project_name", None)
    model_name = data.get("model_name", None)
    env_name = data.get("environment_name", None)
    feedback_json = data.get("feedback_json", None)

    prediction_uid = uuid.uuid4()[:4]
    
    feedback_image = get_image(feedback_json)
    feedback_image_temp_path = os.path.join(Enums.TEMP_DATA_DIRECTORY.value, "feedback_image.jpg")
    cv2.imwrite(feedback_image_temp_path, feedback_image)
    res = upload_to_rlef(Enums.FEEDBACK_MODEL_ID.value, feedback_image_temp_path, feedback_json, project_name, model_name, prediction_uid)

    if res == 200:
        return jsonify({"status" : "Feedback sent successfully"}), 200
    else:
        return jsonify({"error" : "Feedback sending failed"}), 400

app.run(host="0.0.0.0", port=5000, debug=True)
        
        
    