# backend.py


import sys
import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool


import sqlite3
# from pyparsing import Optional
import requests
from exception.custom_exception import DocumentPortalException
from src.document_chat.retrieval import ConversationalRAG
from pathlib import Path
from logger.custom_logger import CustomLogger
from typing import Annotated, Optional
from utils.model_loader import ModelLoader
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    


  
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")  # <--- keep consistent with save_local()

class LangGraphToolBackend:
    """
    LLangGraph defining chatbot with tools.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, index_name="index")
        answer = rag.invoke("What is ...?", chat_history=[])
    """
    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id

            # Load LLM and prompts once
            self.model_loader = ModelLoader()
            self.graph = StateGraph(ChatState)

            self.log.info("LangGraphToolBackend initialized", session_id=self.session_id)
        except Exception as e:
            self.log.error("Failed to initialize LangGraphToolBackend", error=str(e))
            raise DocumentPortalException("Initialization error in LangGraphToolBackend", sys)


    # -------------------
    # 0. State
    # -------------------

    # -------------------
    # 1. LLM
    # -------------------
    # llm = ChatOpenAI()

    # -------------------
    # 2. Tools
    # -------------------
    # Tools
    

    @tool
    def calculator(self,first_num: float, second_num: float, operation: str) -> dict:
        """
        Perform a basic arithmetic operation on two numbers.
        Supported operations: add, sub, mul, div
        """
        try:
            if operation == "add":
                result = first_num + second_num
            elif operation == "sub":
                result = first_num - second_num
            elif operation == "mul":
                result = first_num * second_num
            elif operation == "div":
                if second_num == 0:
                    return {"error": "Division by zero is not allowed"}
                result = first_num / second_num
            else:
                return {"error": f"Unsupported operation '{operation}'"}
            
            return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
        except Exception as e:
            return {"error": str(e)}




    @tool
    def get_stock_price(self,symbol: str) -> dict:
        """
        Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
        using Alpha Vantage with API key in the URL.
        """
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
        r = requests.get(url)
        return r.json()
    
    @tool
    def get_faiss_data(self, state: ChatState) -> dict:
        """
        Fetch last two relevant documents from FAISS index and answer the question.
        """
        session_id = self.session_id
        question = state.message[-1].content
        
        
        k = 3
        if not session_id:
            raise Exception(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE) 
        if not os.path.isdir(index_dir):
            return "No Data in the index"

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)  # build retriever + chain
        response = rag.invoke(question, chat_history=[])
       
        response = rag.invoke(state.question, chat_history=[])
        return response
       


    # -------------------
    # 4. Nodes
    # -------------------
    def chat_node(self,state: ChatState):
        """LLM node that may answer or request a tool call."""
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def define_graph(self):
        self.search_tool = DuckDuckGoSearchRun(region="us-en")
        # self.tools = [self.search_tool, self.get_stock_price, self.calculator, self.get_faiss_data]
        self.tools = [self.search_tool, self.get_stock_price, self.calculator, self.get_faiss_data]
        self.llm = ChatOpenAI () #self.model_loader.load_llm()
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # self.tool_node = ToolNode(self.tools)
        self.tool_node = ToolNode(self.tools)

        # -------------------
        # 5. Checkpointer
        # -------------------
        conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
        self.checkpointer = SqliteSaver(conn=conn)

        # -------------------
        # 6. Graph
        # -------------------
       
        self.graph.add_node("chat_node", self.chat_node)
        self.graph.add_node("tools", self.tool_node)

        self.graph.add_edge(START, "chat_node")

        self.graph.add_conditional_edges("chat_node",tools_condition)
        self.graph.add_edge('tools', 'chat_node')

        self.graph.add_edge('chat_node', END)

        chatbot = self.graph.compile(checkpointer=self.checkpointer)
        return chatbot

    # -------------------
    # 7. Helper
    # -------------------
    def retrieve_all_threads(self):
        all_threads = set()
        for checkpoint in self.checkpointer.list(None):
            thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
            if thread_id:
                all_threads.add(thread_id)

        return list(all_threads)
    
if __name__ == "__main__":
    langraph = LangGraphToolBackend(session_id="default")
    chatbot = langraph.define_graph()
    state = {'messages': [HumanMessage(content="What is the stock price of AAPL?")]}
    for event in chatbot.stream(state, config={'configurable': {'thread_id': 'default'}}):
        print(event)