import streamlit as st
from pathlib import Path
from langchain_community.utilities import SQLDatabase
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# This is the correct 1.0+ import for the SQL agent
from langchain_community.agent_toolkits import create_sql_agent
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
    st.info("Please add the Groq API key")
    st.stop()

# --- THE FIX (1/2) ---
# We must use the modern, "smart" 70B model.
# The 8B model is NOT powerful enough for this SQL agent.
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile", # <-- THIS IS THE CORRECT MODEL
    streaming=True
)

@st.cache_resource(ttl="2h")
def configure_db(
    db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None
):
    if db_uri == LOCALDB:
        # Create a read-only connection to the local SQLite DB
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        print(dbfilepath)
        # Check if the file exists before connecting
        if not dbfilepath.exists():
            st.error(
                f"Database file not found at {dbfilepath}. Please make sure 'student.db' is in the same folder as 'app.py'."
            )
            st.stop()
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

try:
    if db_uri == MYSQL:
        db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
    else:
        db = configure_db(db_uri)
except Exception as e:
    st.error(f"Failed to connect to the database: {e}")
    st.stop()


# --- THE FIX (2/2) ---
# We pair the smart 70B model with the modern "tool-calling" agent.
agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="tool-calling", # This is the modern, correct 1.0+ agent type
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
        # NOTE: expand_new_thoughts=True will NOT work with a "tool-calling"
        # agent, as it does not produce text-based "thoughts".
        streamlit_callback = StreamlitCallbackHandler(
            st.container(), expand_new_thoughts=False
        )

        try:
            # Use .invoke() and {"input": ...} for 1.0+
            response = agent.invoke(
                {"input": user_query}, callbacks=[streamlit_callback]
            )

            # Read the final answer from the "output" key
            final_answer = response["output"]
            st.session_state.messages.append(
                {"role": "assistant", "content": final_answer}
            )
            st.write(final_answer)

        except Exception as e:
            st.error(f"An error occurred: {e}")

