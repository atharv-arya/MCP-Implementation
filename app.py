import asyncio

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
    llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")

    # creating the agent with the memory_enabled = True
    agent = MCPAgent(
        llm = llm,
        client = mcp_client,
        max_steps = 15,
        memory_enabled = True, # enabling conversation memory 
    )

    print("\n-------Chat Session Started-------")
    print("Type 'quit' to end the session.")
    print("Type 'clear' to clear the conversation history.")
    print("-----------------------------------")

    try: 
        # Main chat loop
        while True:
            user_input = input("\nYou:")

            # check for exit command
            if user_input.lower() == "quit":
                print("Chat session ending...")
                break

            # check for clear command
            if user_input.lower() == 'clear':
                print("Conversation history has been cleared.")
                continue

            # Get response from the agent
            print("\nAssistant: ", end = "", flush = True)

            try: 
                # Run the agent with the user input (memory handling is automatic)
                response = await agent.run(user_input)
                print(response)

            except Exception as e:
                print(f"\nError: {e}")

    finally:
        # Clean up
        if mcp_client and mcp_client.sessions:
            await mcp_client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_memory_chat())
