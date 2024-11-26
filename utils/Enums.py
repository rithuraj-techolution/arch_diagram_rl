from enum import Enum

class Enums(Enum):
#     Default Models
    DEFAULT_MODEL = "diagramzone-v2"
    DEFAULT_ENVIRONMENT = "diagram-zone"
    DEFAULT_AGENT = "architecture-v1"

#     Default Paths
    MODEL_DIRECTORY = "models/"
    ENVIRONMENT_DIRECTORY = "utils/environments"
    AGENTS_DIRECTORY = "utils/agents"
    LOGS_DIRECTORY = "logs"
    TEMP_DATA_DIRECTORY = "temp"
    
#     RLEF 
    LLM_GENERATED_MODEL_ID = "67447097fa869acd8eca75b6" 
    RL_OPTIMIZED_MODEL_ID = "674470ddfa869acd8eca7a38"
    FEEDBACK_MODEL_ID = "6745969ffa869acd8ecb1481"  
