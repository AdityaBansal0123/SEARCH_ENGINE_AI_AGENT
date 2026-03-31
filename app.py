import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.callbacks.streamlit import StreamlitCallbackHandler
from langchain_classic.tools import Tool
load_dotenv()

# ---------------- UI ----------------
st.set_page_config(page_title="LangChain Search Agent", page_icon="🔎")
st.title("🔎 Smart Search Agent (Groq + LangChain)")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# ---------------- Tools ----------------

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500),
    description="Use for general knowledge"
)

arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300),
    description="Use for research papers"
)

search = DuckDuckGoSearchRun(
    name="Search",
    description="Use for latest information or current events"
)

def calculator_tool(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid math expression"

calculator = Tool(
    name="Calculator",
    func=calculator_tool,
    description="Useful for solving math problems. Input should be a valid mathematical expression."
)

tools = [search, arxiv, wiki, calculator]

# ---------------- Session Memory ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi 👋 I can search the web, Arxiv, and Wikipedia. Ask me anything!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------- User Input ----------------
if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.warning("Please enter your Groq API key.")
        st.stop()

    # ---------------- LLM ----------------
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        streaming=True,
        temperature=0
    )

    # ---------------- Agent ----------------
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,   # ✅ prevents crash
        verbose=True,
        max_iterations=9            # ✅ prevents infinite loops
    )

    # ---------------- Response ----------------
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        try:
            response = agent.run(f"""
                You are a strict ReAct agent.

                Follow this EXACT format:

                Thought: think step by step
                Action: one of [{', '.join([tool.name for tool in tools])}]
                Action Input: input to the tool

                OR

                Final Answer: your final answer

                User Question: {prompt}
                """,callbacks=[st_cb])

        except Exception as e:
            response = f"⚠️ Error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
