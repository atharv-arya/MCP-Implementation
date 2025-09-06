from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient
import os

async def run_memory_chat():
    "Run a chat using MCPAgent's built-in conversation memory."
    # loading env variables 
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    # config file path
    config_file_path = "browser_mcp.json"

    # creating MCP client and agent with memory enabled
    mcp_client = MCPClient.from_config_file(config_file_path)
    llm = ChatGroq(model="qwen-qwq-32b")

    # creating the agent with the memory_enabled = True
    agent = MCPAgent(
        llm = llm,
        client = mcp_client,
        max_steps = 15,
        memory_enabled = True, # enabling conversation memory 
    )

    print("\n-------Chat Session Started-------")
    print("Type 'quit' to end the session")
    print("Type 'clear' to clear the conversation history")
    print("-----------------------------------")

    