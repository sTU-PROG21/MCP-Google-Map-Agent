from praisonaiagents import Agent, MCP
import os
from ollama_manager import OllamaManager


class InferlessPythonModel:
    def initialize(self):
        maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        manager = OllamaManager()
        manager.start_server()
        model_id = 'llama3.3:70b'
        models = manager.list_models()
        print(f"Available models: {models}")
        
        if not any(model['name'] == model_id for model in models):
            manager.download_model(model_id)
        self.maps_agent = Agent(
            instructions="""You are a helpful assistant that can interact with Google Maps.
            Use the available tools when relevant to handle location-based queries.""",
            llm=f"ollama/{model_id}",
            tools=MCP("npx -y @modelcontextprotocol/server-google-maps",
                    env={"GOOGLE_MAPS_API_KEY": maps_api_key})
        )
    def infer(self,inputs):
        result = self.maps_agent.start(inputs.prompt)
        return result
    def finalize(self):
        pass
