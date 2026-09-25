import ollama


class Agent:
    
    def __init__(self, model_name:str, system_prompt:str):
        self.model_name = model_name
        self.system_prompt = system_prompt