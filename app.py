import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# --- LOAD ENVIRONMENT VARIABLES ---
# Create a .env file in your project directory and add your LANGCHAIN_API_KEY
# LANGCHAIN_API_KEY="your_api_key_here"
load_dotenv()

# --- PAGE CONFIGURATION ---
# Sets the title and icon of the browser tab, and the layout of the page.
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")

# --- CUSTOM CSS FOR STYLING ---
# Injects custom CSS to style the chat interface.
st.markdown("""
<style>
    /* General styles for the main app container */
    .stApp {
        background-color: #f0f2f6; /* Light grey background */
    }

    /* Styles for the chat messages */
    .stChatMessage {
        border-radius: 20px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 1px solid transparent;
        max-width: 85%;
        word-wrap: break-word;
    }

    /* Styles for user messages (on the right) */
    [data-testid="stChatMessageContent"] {
        background-color: #dcf8c6; /* Light green for user messages */
        color: #333;
    }

    /* Align user messages to the right */
    div[data-testid="stChatMessage"][class*="user"] {
        display: flex;
        justify-content: flex-end;
    }
    
    /* Styles for assistant messages (on the left) */
    div[data-testid="stChatMessage"][class*="assistant"] [data-testid="stChatMessageContent"] {
        background-color: #ffffff; /* White for assistant messages */
        color: #333;
        border: 1px solid #e0e0e0;
    }

    /* Header styling */
    .st-emotion-cache-18ni7ap {
        background-color: #ffffff;
        border-bottom: 2px solid #f0f2f6;
    }
    
    /* Input box styling */
    [data-testid="stChatInput"] {
        background-color: #ffffff;
        border-top: 2px solid #f0f2f6;
    }
    
    [data-testid="stTextInput"] > div > div > input {
        border-radius: 20px;
        border: 1px solid #ccc;
        padding: 0.75rem 1rem;
    }
    
    [data-testid="stButton"] > button {
        border-radius: 50%;
        width: 40px;
        height: 40px;
    }
</style>
""", unsafe_allow_html=True)


# --- LANGCHAIN SETUP ---
# Set up LangChain tracking and API key from environment variables.
# This helps in monitoring and debugging your LangChain applications.
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Define the prompt template for the chat.
# This structures the input to the language model.
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful and friendly AI assistant. Please respond to user queries in a clear and concise way."),
        ("user", "Question: {question}")
    ]
)

# Initialize the Ollama model (ensure you have Ollama running with the llama3 model).
try:
    llama_model = Ollama(model="llama3")
    # Define the processing chain: prompt -> model -> output parser.
    chain = prompt_template | llama_model | StrOutputParser()
except Exception as e:
    st.error(f"Failed to initialize Ollama model. Please ensure Ollama is running and the 'llama3' model is available. Error: {e}")
    st.stop() # Stop the app if the model can't be loaded.

# --- STREAMLIT APP LAYOUT ---

# App header
st.title("🤖 AI Assistant")
st.caption("Your friendly neighborhood chatbot, powered by Llama3")


# Initialize chat history in session state if it doesn't exist.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
    ]

# Display existing chat messages from history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input from the chat interface.
if user_input := st.chat_input("Ask your questions..."):
    # Add user's message to chat history and display it.
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display the assistant's response.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the LangChain chain to get the response.
                response = chain.invoke({"question": user_input})
                st.markdown(response)
                # Add assistant's response to chat history.
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"Sorry, I encountered an error. Please try again. Details: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
