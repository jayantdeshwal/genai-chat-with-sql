import streamlit as st
from pathlib import Path
# --- We are using the "zero-shot-react-description" agent ---
# This is the text-based agent that "thinks" out loud.
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLLite 3 Database- Student.db", "Connect to your SQL Database"]
selected_opt = st.sidebar.radio(
    label="Choose the DB which you want to chat", options=radio_opt
)


if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MYSQL User")
    mysql_password = st.sidebar.text_input("MYSQL password", type="password")
    mysql_db = st.sidebar.text_input("MySQL database")
else:
    db_uri = LOCALDB

api_key = st.sidebar.text_input(label="Groq API Key", type="password")


if not api_key:
    st.info("Please add the groq api key")
    st.stop()

## LLM model
# --- FIX 1: Use the "smarter" 8k model ---
# The 'llama-3.1-8b-instant' model is NOT smart enough for this agent.
# 'Llama3-8b-8192' is the only one that works reliably.
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="Llama3-8b-8192", # <-- CRITICAL CHANGE
    streaming=True
)

@st.cache_resource(ttl="2h")
def configure_db(
    db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None
):
    if db_uri == LOCALDB:
        # Assume student.db is in the same directory as the app.py
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        if not dbfilepath.exists():
            st.error(f"Database file not found at: {dbfilepath}")
            st.stop()
        # Connect in read-only mode
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(
            create_engine(
                f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
            )
        )

if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)


# --- FIX 2: Use the ReAct Agent that works with this model ---
# This is the text-based agent that "thinks" out loud.
agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="zero-shot-react-description", # <-- CRITICAL CHANGE
    verbose=True,
    handle_parsing_errors=True,
    # This suffix is CRITICAL to stop the "iteration limit" error
    suffix="""Begin!

Question: {input}
Thought: {agent_scratchpad}""",
)


if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        # --- FIX 3: This handler will now work! ---
        # It can read the text "Thoughts" from the ReAct agent.
        streamlit_callback = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=True, # <-- THIS WILL NOW WORK
            collapse_completed_thoughts=False,
        )
        
        try:
            response = agent.invoke(
                {"input": user_query},
                callbacks=[streamlit_callback]
            )
            # The output of this agent is in the "output" key
            content = response["output"]
            
            st.session_state.messages.append({"role": "assistant", "content": content})
            st.write(content)

        except Exception as e:
            # Catch any other errors
            st.error("An error occurred. Please try a different query.")
            st.error(f"Details: {e}")

