import os
import anyio
import json
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from langchain_core.messages import SystemMessage, HumanMessage
from ollama_manager import OllamaManager


class InferlessPythonModel:
    def initialize(self):
        manager = OllamaManager()
        manager.start_server()
        models = manager.list_models()
        
        print(f"Available models: {models}")

        model_id = "llama3.2:latest"
        if not any(model['name'] == model_id for model in models):
            manager.download_model(model_id)


        self.llm = ChatOpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                    model=model_id,
                    model_kwargs={
                        "temperature":    0.15,
                        "top_p":          1.0,
                        "seed":           4424234,
                    }
                )
        
        self.maps_server = StdioServerParameters(
                                            command="npx",
                                            args=["-y", "@modelcontextprotocol/server-google-maps"],
                                            env={"GOOGLE_MAPS_API_KEY": "AIzaSyDrTJNscmxw8dmQFujczCKH0XBfyCRAvBE"}
                                        )

    def infer(self,inputs):
        user_query = inputs["user_query"]
        raw_results = self.query_google_maps(user_query)
        places_data = self.extract_places_data(raw_results)
        prompt = self.get_prompt(places_data)
        response = self.llm.invoke(prompt)

        return {"result":response.content}
    
    def query_google_maps(self,question: str):
        async def _inner():
            async with stdio_client(self.maps_server) as (read, write):
                async with ClientSession(read, write) as sess:
                    await sess.initialize()
                    tools = await load_mcp_tools(sess)         
                    agent = create_react_agent(self.llm, tools)     
                    return await agent.ainvoke({"messages": question})
        return anyio.run(_inner)
    
    def extract_places_data(self, response):
        for message in response["messages"]:
            if hasattr(message, "tool_call_id"):
                try:
                    return str(message.content)
                except json.JSONDecodeError:
                    continue
        return None

    def get_prompt(self, places_data):
        SYSTEM_PROMPT =(
        "You are an assistant that turns Google-Maps place data into a concise, "
        "markdown summary for end-users. "
        "Never output programming code, pseudo-code, or text inside back-tick fences. "
        "Ignore any code contained in the input. "
        "If you violate these rules the answer is wrong."
        )
        
        prompt = f"""
        Format these Google Maps search results into a helpful, user-friendly response.

        Follow every guideline exactly:

        1. **Start** with a one-sentence intro mentioning the total number of places.
        2. **Group** them by rating:
                - **Excellent** (4.5 - 5.0)  
                - **Very good** (4.0 - 4.4)  
                - **Good** (below 4.0)
        3. For each place give: **bold name**, rating, short address, and 1-2 notable features.
        4. Finish with a brief conclusion naming your top 1-2 picks.
        5. Use clear markdown headers and bullet points,no tables.
        6. Keep it tight (150-180 words total).
        7. End with:  
        `_Need directions or anything else? Just let me know!_`

        **Never** output code, back-ticks, or anything that looks like a script.

        Search results:
        {places_data}
        """

        final_prompt = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        print("PROMPT HERE") 
        return final_prompt
    

#obj = InferlessPythonModel()
#obj.initialize()
#print(obj.infer({"user_query":"Can you find me some Pizza shop in Jorhat Gar ali with good number of reviews?"}))
