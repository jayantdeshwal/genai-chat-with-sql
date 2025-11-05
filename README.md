🦜 LangChain: Chat with SQL DB

This is a Streamlit web application that uses a modern LangChain 1.0+ SQL Agent to answer natural language questions about a database. It is powered by the Groq Llama 3.1 8B model.

Users can ask questions like "How many students are in the database?" or "List the students in the 'Physics' department," and the agent will generate and execute the correct SQL query to get the answer.

Tech Stack

Python: The core programming language.

Streamlit: For building the interactive web UI.

LangChain (1.0+): For the agent framework (create_sql_agent, langchain-community).

Groq: For serving the Llama 3.1 8B model.

SQLAlchemy: The library used by LangChain to connect to the SQL database.