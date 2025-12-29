
from langgraph.graph import START , END , StateGraph
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint , HuggingFaceEndpointEmbeddings
from langgraph.prebuilt import ToolNode , tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_chroma import Chroma
from dotenv import load_dotenv
from typing import TypedDict , Annotated ,List , Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage , SystemMessage , HumanMessage , ToolMessage , AIMessage
from langchain_community.utilities import GoogleScholarAPIWrapper
from langchain_core.documents import Document
from query_retriver import get_multiquery_documents
import os
import requests
import re

load_dotenv()

"""# define model"""

llm = HuggingFaceEndpoint(
    repo_id='allenai/Olmo-3-7B-Instruct',
    # repo_id='Qwen/Qwen3-4B-Instruct-2507',
    huggingfacehub_api_token = os.environ['HF_TOKEN'],
)

model = ChatHuggingFace(llm=llm)


# 1. Initialize Embeddings
embedding_model = HuggingFaceEndpointEmbeddings(
    model='sentence-transformers/all-MiniLM-L6-v2',
    huggingfacehub_api_token=os.environ['HF_TOKEN']
)

# 2. Load Vector Store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    collection_name="collection_research_guide"
)

"""# create tools"""

search_tool = DuckDuckGoSearchRun()

@tool
def google_scholar_search(name:str) -> dict:
    """
    Search for the academic publications and citations of a specific professor
    affiliated with IIT ISM Dhanbad using Google Scholar.

    STRICT USAGE GUIDELINES:
    - Use this tool ONLY when a user provides a specific person's name and
      asks about their research papers, h-index, citations, or academic output.
    - DO NOT use this tool for general queries about IIT ISM, department info,
      or broad research topics.
    - DO NOT use this tool if a professor's name is not explicitly mentioned.

    Args:
        name: The full name of the professor (e.g., 'Prof. Shalivahan' or 'Dr. Ajit Kumar').
              The tool automatically handles name cleaning and university-specific filtering.
    """

    try :
      name = name.lower()
      name = re.sub(r'\b(iit|ism|dhanbad)\b', '', name)
      name = ' '.join(name.split())

      google_scholar_search_tool = GoogleScholarAPIWrapper(
        serp_api_key = os.environ['SERP_API_KEY'],
        google_scholar_engine= "google_scholar",
      )

      result = google_scholar_search_tool.run(f"{name} iit ism ")

      return {"Professor": name.title(), "results": result}

    except Exception as e:
          return {"error": str(e)}



@tool
def rag_tool(query: str) -> Dict[str, Any]:
    """
    Primary search tool for all inquiries regarding IIT ISM Dhanbad projects,
    faculty research papers, and institutional academic initiatives.

    Use this tool ONLY when the user's request pertains to:
    - Ongoing or completed research projects at IIT ISM.
    - Specific research domains (e.g., Mining, Petroleum, Earth Sciences, AI).
    - Faculty publications, lab facilities, or interdisciplinary research centers.

    Args:
        query: A refined, standalone search query. If the user's input is vague
               (e.g., "tell me more"), rewrite it to be specific (e.g., "Current
               research projects in the Mining Engineering department at IIT ISM").
    """
    try:

        # print("calling the rag tool")
        # Fetch documents using your multi-query retriever logic
        # results = get_multiquery_documents(query=query)

        retriever = vectorstore.as_retriever(
            search_type = "similarity",
            search_kwargs={"k": 5}
        )

        results = retriever.invoke(query)

        # print("got all documents")

        if not results:
            return {
                "query": query,
                "context": [],
                "metadata": [],
                "status": "success",
                "message": "No relevant documents found for this query."
            }

        # Extract content and metadata
        context = [doc.page_content for doc in results]
        # metadata = [doc.metadata for doc in results]

        return {
            "query": query,
            "context": context,
            # "metadata": metadata,
            "status": "success"
        }

    except Exception as e:
        # Returning the error as a string allows the LLM to understand
        # the failure rather than having the entire graph crash.
        return {
            "query": query,
            "error": f"An error occurred during retrieval: {str(e)}",
            "status": "error"
        }

tools = [search_tool , google_scholar_search , rag_tool]
model_with_tools = model.bind_tools(tools)

"""# create graph state"""

class ChatState(TypedDict):
    messages: List[BaseMessage]
    query : str
    response : str

"""# system instruction"""

system_instruction = SystemMessage(content="""
You are the Official IIT ISM Dhanbad Research & Faculty Assistant.

CORE SCOPE:
1. All research-related queries you handle MUST be specific to IIT ISM Dhanbad.
2. If a user asks about general research, projects, or academic publications, you must pivot the answer to focus solely on the contributions, labs, and faculty of IIT ISM Dhanbad.
3. You have deep knowledge of IIT ISM faculty profiles, ongoing institutional projects, and student research initiatives.

DOMAIN GUARDRAILS:
- POLITE GREETINGS: You can engage in basic pleasantries (e.g., "Hello," "How can I help you today?").
- STRICT LIMITATION: For any query unrelated to IIT ISM Dhanbad (e.g., stock market prices, general world news, sports, or non-IIT ISM research), politely decline by saying: "I am specialized specifically in IIT ISM Dhanbad's research and projects. I cannot assist with [topic], but I can tell you about the research happening at IIT ISM!"

TONE: Professional, academic, and helpful.
""")

"""# create nodes"""

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

def chat_node(state: ChatState):
    messages = state.get("messages", [])

    print("at chat node")
    completion_guard = SystemMessage(content=(
        "You are a professional research assistant with a focus on IIT ISM. "
        "Every query is related to research or faculty at IIT ISM. "
        "Try to answer on your own , if not found then go for the given tools. "
        "IMPORTANT: Please provide a clear, **point-wise** response to every query. "
        "Your answers should be **concise and under 500 words**. "
        "Ensure that each point is well explained but stays within the word limit. "
        "If you are summarizing research projects or faculty work, ensure each point is fully addressed without leaving any gaps. "
        "Do not end your response until every aspect of the query has been covered in detail."
    ))

    if messages and isinstance(messages[-1], ToolMessage):
        llm_inputs = [completion_guard] + messages
    else:
        llm_inputs = [completion_guard, HumanMessage(content=state['query'])]

    ai_message = model_with_tools.invoke(llm_inputs)

    print(ai_message)
    return {
        "messages": llm_inputs[1:] + [ai_message],
        "response": ai_message.content
    }

tool_node = ToolNode(tools)

"""# create graph"""

graph = StateGraph(ChatState)

graph.add_node("chat_node" ,chat_node )
graph.add_node("tools" , tool_node)

graph.add_edge(START , "chat_node")
graph.add_conditional_edges("chat_node" , tools_condition)
graph.add_edge("tools" , "chat_node")

chatbot = graph.compile()

# print(chatbot)

def get_answer(query) -> str:

    if not query:
        return "Please provide a query."

    user_input = query

    print("user input : " , query) 

    output = chatbot.invoke({"query": user_input})

    print(output["response"])
    
    return output["response"]


# print(model.invoke("what is google brain ?"))
# print(get_answer("tunnel construction research projects"))