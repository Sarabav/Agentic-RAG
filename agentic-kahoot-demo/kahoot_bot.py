#!/usr/bin/env python3
"""
Minimal CrewAI Flow-based Kahoot Bot using Browser MCP
A persistent agent that plays Kahoot quizzes automatically
"""

import os
import time
from typing import Dict
from pydantic import BaseModel
from crewai import Agent, LLM, Task
from crewai.flow.flow import Flow, listen, start
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from dotenv import load_dotenv
from rag_tool import RAGTool

# Load environment variables
load_dotenv()

# Flow state model
class KahootBotState(BaseModel):
    game_pin: str = os.getenv("KAHOOT_PIN", "299268")
    nickname: str = os.getenv("KAHOOT_NICKNAME", "CrewAI_Bot")
    is_connected: bool = False

class KahootBotFlow(Flow[KahootBotState]):
    """Persistent Kahoot bot using CrewAI Flow and Browser MCP"""
    
    def __init__(self):
        super().__init__()
        
        # Configure LLM
        self.llm = LLM(
            model="openai/gpt-4o-mini",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Browser MCP server parameters - use the actual @browsermcp/mcp package
        env = os.environ.copy()
        env["SKIP_PORT_KILL"] = "1"
        self.browser_server_params = StdioServerParameters(
            command="npx",
            args=["@browsermcp/mcp@latest"],
            env=env
        )
        
    @start()
    def initialize_bot(self) -> KahootBotState:
        """Initialize the Kahoot bot"""
        print("🤖 Starting Kahoot Bot...")
        
        # Debug environment variables
        print(f"🐛 Debug - KAHOOT_PIN env var: '{os.getenv('KAHOOT_PIN')}'")
        print(f"🐛 Debug - KAHOOT_NICKNAME env var: '{os.getenv('KAHOOT_NICKNAME')}'")
        
        state = KahootBotState()
        print(f"🔧 Using PIN: {state.game_pin}")
        print(f"🔧 Using Nickname: {state.nickname}")
        
        return state
    
    @listen(initialize_bot)
    def play_quiz_actively(self, state: KahootBotState) -> KahootBotState:
        """Keep bot alive and answer questions one by one with user confirmation"""
        print(f"🎮 Starting persistent quiz bot for {state.nickname}")
        print("📋 ASSUMPTION: You are already in the Kahoot waiting room")

        # Prompt user to select a collection at the start of the session
        from chromadb import PersistentClient

        chroma_client = PersistentClient(path="./chroma_db")
        collections = chroma_client.list_collections()
        if not collections:
            print("❌ No collections found in ChromaDB. Please create one using chromadb_manager.py first.")
            return state

        print("📚 Available ChromaDB collections:")
        for idx, col in enumerate(collections):
            print(f"{idx+1}. {col.name}")

        selected_collection_name = None
        while selected_collection_name is None:
            try:
                choice = int(input("Select a collection by number for RAGTool: ")) - 1
                if 0 <= choice < len(collections):
                    selected_collection_name = collections[choice].name
                    print(f"✅ Selected collection for RAGTool: {selected_collection_name}")
                else:
                    print("❌ Invalid selection. Try again.")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")

        try:
            rag_tool = RAGTool(collection_name=selected_collection_name, db_path="./chroma_db/chroma.sqlite3")
        except Exception as e:
            print(f"❌ Failed to initialize RAGTool: {e}")
            rag_tool = None
        if not rag_tool:
            print("❌ RAGTool initialization Failed.")
            return state

        try:
            with MCPServerAdapter(self.browser_server_params) as browser_tools:
                print("🚀 Browser MCP server started!")
                print("➡️  In Chrome, click the Browser-MCP extension icon, then 'Connect'.")
                print("   After it shows the green badge, come back here and press Enter...")
                
                # Wait for user to connect extension
                input("🔄  Press <Enter> when the extension is connected...")
                
                # Test connection
                while True:
                    try:
                        result = browser_tools.browser_snapshot(random_string="ping")
                        print(f"✅ Connection test successful: {len(str(result))} chars returned")
                        break
                    except Exception as e:
                        print(f"❌ Connection test failed: {str(e)}")
                        try:
                            tool_names = [tool.name for tool in browser_tools]
                            print(f"ℹ️  Tools available: {tool_names}")
                            if 'browser_navigate' in tool_names:
                                print("✅ Browser tools detected, proceeding anyway...")
                                break
                        except Exception as e2:
                            print(f"❌ Tools check also failed: {str(e2)}")
                        
                        print("Make sure the Browser MCP extension is connected (green badge) and press Enter...")
                        input()

                print("✅ Browser MCP connected!")
                print(f"Available tools: {[tool.name for tool in browser_tools]}")
                
                # Simple answer instructions
                answer_instructions = f"""
                SYSTEM: For every quiz question, you MUST use the RAGTool to retrieve knowledge from the selected ChromaDB collection before answering, even if you think you know the answer. Do NOT answer from your own knowledge. Always call the RAGTool first, then use its output to select the answer.

                1. Take a browser_snapshot to see the current page
                2. If you see a quiz question with colored answer buttons:
                   - Read the question carefully
                   - Use the RAGTool to search for the answer (ALWAYS do this first)
                   - Click the most logical answer using browser_click, based on the RAGTool's output
                3. If no question is visible, report what you see

                Be quick and decisive with your answer choice.
                """

                # Continuous quiz loop
                question_count = 0
                print("\n🎯 Starting continuous quiz mode!")
                print("💡 Bot will answer one question, then wait for your input to continue")
                print("❌ Press Ctrl+C to stop the bot\n")
                
                while True:
                    try:
                        question_count += 1
                        print(f"🔍 Looking for question #{question_count}...")
                        
                        # Create fresh agent for each question (no memory persistence)
                        # Add RAGTool to the list of tools if available
                        tools_list = list(browser_tools)
                        if rag_tool:
                            tools_list.append(rag_tool)

                        quiz_agent = Agent(
                            role="Kahoot Quiz Player",
                            goal="For every quiz question, you MUST use the RAGTool to retrieve knowledge from the selected ChromaDB collection before answering. Do NOT answer from your own knowledge. Always call the RAGTool first, then use its output to select the answer.",
                            backstory="I always use the RAGTool to search for the answer to every quiz question, even if I think I know it. I never answer from my own knowledge. I analyze quiz questions, use the RAGTool, and click the most logical answer choice based on its output.",
                            llm=self.llm,
                            tools=tools_list,
                            verbose=True,
                            memory=False,  # No memory to avoid cached browser state
                        )
                        
                        quiz_task = Task(
                            description="Look at the browser page and click the best answer if there's a quiz question. If you need extra knowledge, use the RAGTool.",
                            expected_output="Answer clicked or report of current page state",
                            agent=quiz_agent,
                            tools=tools_list,
                            prompt=answer_instructions,
                        )

                        result = quiz_task.execute_sync()
                        print(f"✅ Question #{question_count} Result: {result}")
                        
                        # Wait for user to press Enter before continuing to next question
                        print(f"\n⏳ Question #{question_count} completed!")
                        print("📋 Navigate to the next question in Kahoot (if any)")
                        next_action = input("🔄 Press <Enter> to look for next question, or type 'quit' to stop: ").strip().lower()
                        
                        if next_action == 'quit':
                            print("🛑 Stopping quiz bot as requested")
                            break
                            
                        print(f"\n{'='*50}")
                        
                    except KeyboardInterrupt:
                        print("\n🛑 Quiz bot stopped by user (Ctrl+C)")
                        break
                    except Exception as e:
                        print(f"❌ Error in question #{question_count}: {e}")
                        retry = input("🔄 Press <Enter> to retry, or type 'quit' to stop: ").strip().lower()
                        if retry == 'quit':
                            break

        except Exception as e:
            print(f"❌ Quiz monitoring failed: {e}")
            
        return state

def main():
    """Main entry point"""
    print("🚀 CrewAI Kahoot Bot Starting...")
    
    # Create and run the flow
    bot_flow = KahootBotFlow()
    result = bot_flow.kickoff()
    
    print(f"\n🎯 Final Result: {result}")

if __name__ == "__main__":
    main()
