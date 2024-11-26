import json
import uuid
import requests

def upload_to_rlef(model_id, file_path, json_data, project_name, model_name):
    url = "https://proposals-rlef.techo.camp/backend/resource/"
    payload = {'model': model_id,
    'status': 'backlog',
    'csv': json.dumps(json_data),
    'label': 'ai_generated',
    'tag': f"{model_name}_{project_name}",
    'prediction': 'ai_generated',
    'confidence_score': '100',
    }
    
    file_name = f"{project_name}_{model_name}_{str(uuid.uuid4())[:3]}.png"
    
    files=[
    ('resource',(file_name, open(file_path,'rb'),'image/jpeg'))
    ]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    # print(response.text)
    return response.status_code