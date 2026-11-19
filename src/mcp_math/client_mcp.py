import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from langchain.chat_models import init_chat_model
import asyncio
from pydantic import BaseModel,Field
class weatheroutput(BaseModel):
    location: str
    temperature: float
    condition: str
    tool: list 
client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            # Make sure to update to the full absolute path to your math_server.py file
            "args": [r"C:\Users\Lenovo\Downloads\mcp_exercise\mcp-math\src\mcp_math\math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            # Make sure you start your weather server on port 8000
            "url": "http://localhost:8000/mcp",
            "transport": "http",
        }
    }
)
async def main():
    tools = await client.get_tools()
    #model=init_chat_model("openrouter:openai/gpt-4o-mini",url="https://api.openrouter.ai/v1",api_key=os.environ.get("OPENROUTER_API_KEY"))
    agent = create_agent( model="openrouter:openai/gpt-4o-mini", tools=tools,
                         response_format=weatheroutput)
    #math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
   
    weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"})
    #print_tool_calls("Math", math_response)
    #print("Math Response:", math_response["messages"][-1].content)
    print_tool_calls("Weather", weather_response)
    print("Weather Response:", weather_response["messages"][-1].content)


def print_tool_calls(label, response):
    """Print every tool the agent called, with its arguments and result."""
    results = {
        m.tool_call_id: m.content
        for m in response["messages"]
        if getattr(m, "type", None) == "tool"
    }
    for m in response["messages"]:
        for call in getattr(m, "tool_calls", None) or []:
            print(f"[{label}] Tool called: {call['name']}({call['args']})")
            print(f"[{label}]   -> {results.get(call['id'])}")


asyncio.run(main())