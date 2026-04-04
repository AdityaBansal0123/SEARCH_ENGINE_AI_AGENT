import math
import re
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="Smart Search Agent", page_icon="🔎", layout="centered")
st.title("🔎 Smart Search Agent")
st.caption("Powered by Groq · LangChain · Wikipedia · Arxiv · DuckDuckGo")

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("Groq API Key", type="password", placeholder="gsk_...")
model_name = st.sidebar.selectbox(
    "Model",
    ["llama-3.3-70b-versatile"],
    index=0,  # Highly recommended to stick with Llama 3 70B for ReAct agents
)
max_iterations = st.sidebar.slider("Max Agent Iterations", 3, 15, 7)
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi 👋 Ask me anything!"}
    ]
    st.rerun()

# ──────────────────────────────────────────────
# TOOLS
# ──────────────────────────────────────────────
wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=800),
)
wiki_tool.description = (
    "Use Wikipedia for factual, encyclopedic information about people, places, "
    "history, science, and concepts. Input: a concise search term."
)

arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=600),
)
arxiv_tool.description = (
    "Use Arxiv to find academic/research papers. Input: a short topic or paper title."
)

search_tool = DuckDuckGoSearchRun(name="DuckDuckGo")
search_tool.description = (
    "Use DuckDuckGo for current events, news, or recent information. Input: a search query."
)

tools = [search_tool, arxiv_tool, wiki_tool]

# ──────────────────────────────────────────────
# REACT PROMPT
# Required variables: {tools}, {tool_names}, {input}, {agent_scratchpad}
# ──────────────────────────────────────────────
# FIX APPLIED: Added explicit instruction to always output Action Input
REACT_TEMPLATE = """You are a helpful, accurate assistant with access to tools.

Tools available:
{tools}

RULES:
- Use Calculator for ANY math or numerical computation — never calculate mentally.
- Use Wikipedia for encyclopedic facts.
- Use Arxiv for research papers.
- Use DuckDuckGo for recent/news/web info.
- YOU MUST ALWAYS provide an 'Action Input:' when using a tool. Do not just output the 'Action:'.

Strictly follow this format — no deviations:

Question: the input question you must answer
Thought: reasoning about what to do
Action: one of [{tool_names}]
Action Input: input to the tool (plain text, no quotes, no JSON)
Observation: result from the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

react_prompt = PromptTemplate.from_template(REACT_TEMPLATE)

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi 👋 Ask me anything — I can search the web, Wikipedia, Arxiv, and solve math!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ──────────────────────────────────────────────
# CHAT INPUT
# ──────────────────────────────────────────────
if user_input := st.chat_input("Ask anything…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    if not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar.")
        st.stop()

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        streaming=True,
        temperature=0,
    )

    react_agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

    # FIX APPLIED: early_stopping_method changed to "force"
    executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
        early_stopping_method="force", 
        return_intermediate_steps=False,
    )

    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            result = executor.invoke(
                {"input": user_input},
                config={"callbacks": [cb]},
            )
            response = result.get("output", "No response generated.")
        except Exception as ex:
            response = f"⚠️ Agent error: {ex}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
