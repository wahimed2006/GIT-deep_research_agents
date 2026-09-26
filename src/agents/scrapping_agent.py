from typing import Any, Dict, List

from tools_calling_agent import ToolCallingAgent

class ScrappingAgent(ToolCallingAgent):
    
    def __init__(self, model_name: str, tools: List[Dict[str, Any]], system_prompt: str = ""):
        super().__init__(model_name, tools, system_prompt)