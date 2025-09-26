# backend.py
from asyncio import log
import os
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
from src.document_chat.retrieval import ConversationalRAG
from utils.document_ops import load_documents

load_dotenv()

# -------------------
# 1. LLM
# -------------------
llm = ChatOpenAI()

# -------------------
# 2. Tools
# -------------------
# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
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
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

#----------------------------
# adding FAISESS tool

#---------------------
@tool
def get_faiss_data(question: str) -> dict:
    """
    Fetch relevant documents from FAISS index and answer the question.
    If no relevant answer is found, increase top-k retrieval by 5 each time.
    """

    from src.document_chat.retrieval import ConversationalRAG

    FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")
    FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
    session_id = "default"
    use_session_dirs=True
    index_dir = os.path.abspath(os.path.join(FAISS_BASE, session_id)) if use_session_dirs else FAISS_BASE

    if not os.path.isdir(index_dir):
        return {"error": f"No FAISS index found at {index_dir}"}

    try:
        rag = ConversationalRAG(session_id=session_id)
        max_k = 50
        k = 10
        increment = 5

        while k <= max_k:
            print(f"Trying retrieval with top-k = {k}")
            rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)
            response = rag.invoke(question, chat_history=[])

            # Customize this check based on your response format
            if response and isinstance(response, str) and "no relevant" not in response.lower():
                return {"answer": response, "retrieval_k": k}

            k += increment

        return {
            "error": "No relevant documents found after increasing k",
            "max_k_tried": max_k
        }

    except Exception as e:
        return {"error": str(e)}

tools = [search_tool, get_stock_price, calculator, get_faiss_data]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}



tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

if __name__ == "__main__":
   
    state = {'messages': [HumanMessage(content="What is the stock price of Apple?")]}
    for event in chatbot.stream(state, config={'configurable': {'thread_id': 'default'}}):
        print(event)