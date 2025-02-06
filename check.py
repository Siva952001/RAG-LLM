import streamlit as st
from streamlit_option_menu import option_menu
import base64
from googletrans import Translator
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.ollama import OllamaEmbeddings
import pandas as pd
import os
from database import get_mongo_client, get_database, get_collection1, get_collection2,get_collection3, save_chat_to_db
from hashlib import sha256
from datetime import datetime
from chat_retrieval import load_chat_retrieval,get_categories_from_db,add_category_to_db
from file_process import process_uploaded_files

logo_path = "C:/Users/DEV-037/Desktop/test-mesop/images/logo.png"


with open("C:/Users/DEV-037/Desktop/test-mesop/images/logo.png", "rb") as image_file:
    logo_base64 = base64.b64encode(image_file.read()).decode("utf-8")


st.markdown(f"""
    <style>
    .sidebar .sidebar-content {{
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }}
    .sidebar .sidebar-content h1 {{
        font-size: 24px;
        color: #1f77b4;
        font-weight: bold;
    }}
    .stTextInput>div>div>input {{
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 12px;
        font-size: 16px;
    }}
    .stButton>button {{
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: #155a8a;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }}
    .chat-container {{
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        background-color: #f9f9f9;
        margin-bottom: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .message {{
        margin-bottom: 10px;
        font-size: 16px;
    }}
    .message.user {{
        color: #1f77b4;
        font-weight: bold;
    }}
    .message.assistant {{
        color: #e75f5f;
        font-weight: bold;
    }}
    .message.translated {{
        font-style: italic;
        color: #6c757d;
        font-size: 14px;
    }}
    .stDateInput>div>div {{
        font-size: 16px;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #f0f2f6;
    }}
    .stContainer {{
        padding: 15px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .small-button {{
        font-size: 12px;
        padding: 5px 10px;
        margin: 5px 0;
        border-radius: 5px;
    }}
    .center-text {{
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        color: #333333;
        margin-top: 20px;
        margin-bottom: 20px;
        line-height: 1.2;
    }}
    .center-text p {{
        margin: 0;
    }}
    .bottom-logo {{
       
        display:flex;
        justify-content:center;
        align-items:center;
        text-align: center;
        font-size: 14px;
        color: #333333;
        position:fixed;
        bottom:0 !important;
        right:10% !important;
    }}
    .bottom-logo span{{
        font-weight:bold !important;
        
    }}
      .bottom-logo img{{
        margin-bottom:10px;
        
    }}
  
    </style>

    <div class="bottom-logo">
        <span class="powered-by">Powered by</span>
        <img src="data:image/png;base64,{logo_base64}" alt="Logo"/>
    </div>
""", unsafe_allow_html=True)


# Mongo setup
client = get_mongo_client()
db = get_database(client)
collection = get_collection1(db)
credentials_collection = get_collection2(db)
knowledge_base = get_collection3(db)



def check_credentials(username, password):
    hashed_password = sha256(password.encode()).hexdigest()
    user = credentials_collection.find_one({"username": username, "password": hashed_password})
    return user is not None

def create_user(username, password):
    hashed_password = sha256(password.encode()).hexdigest()
    credentials_collection.insert_one({"username": username, "password": hashed_password})
    
def list_models(model_dir):
    return [model for model in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, model))]



def is_admin(username, password):
    return username == "admin" and password == "123"

# Initialize session state if not already present
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""
if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}

MODEL_DIR = "C:/Users/DEV-037/.ollama/models/manifests/registry.ollama.ai/library"
FILE_DIR = 'files'
DATA_DIR = 'data'
translator = Translator()

os.makedirs(FILE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

available_models = list_models(MODEL_DIR)

# Session management UI
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

     
# Login Form
def login():
    with st.form(key='login_form', clear_on_submit=True):
        username = st.text_input("Username", key="login_username", help="Enter your username")
        password = st.text_input("Password", type="password", key="login_password", help="Enter your password")
        submit_button = st.form_submit_button("Login", help="Click to login")
        
        if submit_button:
            if not username or not password:
                st.error("Please enter both username and password.")
            elif is_admin(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.is_admin = True
                reset_session_state()
                st.success("Admin login successful!")
                st.experimental_rerun()
            elif check_credentials(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                reset_session_state()
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
                
# Session State Reset           
def reset_session_state():
    st.session_state.chat_history = [] 
    st.session_state.processed_files = {} 
    st.session_state.vectorstore = None  
    # st.session_state.qa_chain = None  
            
            
# Sign-up Form
def signup():
    with st.form(key='signup_form', clear_on_submit=True):
        username = st.text_input("Username", key="signup_username", help="Choose a new username")
        password = st.text_input("Password", type="password", key="signup_password", help="Set your password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password", help="Re-enter your password")
        
        
        submit_button = st.form_submit_button("Sign Up", help="Create a new account")
        
        if submit_button:
            if not username or not password or not confirm_password:
                st.error("Please enter both username and password.")
            elif password != confirm_password:
                st.error("Passwords do not match. Please try again.")
            elif check_credentials(username, password):
                st.error("Username already exists. Please choose a different username.")
            else:
                create_user(username, password)
                st.success("User created successfully!")
                st.session_state.logged_in = True
                st.session_state.username = username
                st.experimental_rerun()

# Main logic to toggle between login and signup form
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    # Use option_menu from streamlit-option-menu
    selected = option_menu(
        menu_title=None, 
        options=["Login", "Sign Up"],  
        icons=["box-arrow-in-right", "person-plus"], 
        menu_icon="cast",  
        default_index=0,  
        orientation="horizontal",  
        styles = {
    "container": {
        "padding": "0!important", 
        "background-color": "#ffffff",  
        "transition": "background-color 0.3s ease"  
    },
    "icon": {
        "color": "#007bff",  
        "font-size": "25px",
        "transition": "color 0.3s ease"  
    },
    "nav-link": {
        "font-size": "16px", 
        "text-align": "center", 
        "margin": "0px", 
        "--hover-color": "#f0f0f0",  
        "color": "#333",  
        "transition": "color 0.3s ease, background-color 0.3s ease"  
    },
    "nav-link-hover": {
        "background-color": "#e5e5e5",  
        "color": "#0056b3",  
        "transition": "background-color 0.3s ease, color 0.3s ease"  
    },
    "nav-link-selected": {
        "background-color": "#0056b3",  
        "color": "white", 
        "font-weight": "bold",  
        "transition": "background-color 0.3s ease, color 0.3s ease"  
    }
 }
)

    if selected == "Login":
        login()
    elif selected == "Sign Up":
        signup()
else:
    st.sidebar.markdown("You are now logged in.")
    
    

    def initiate_chat():
    # Check if 'selected_model' is set in the session state, if not, set to default
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = available_models[0] if available_models else None

        # Initialize LLM and embedding model if not already set
        if 'llm' not in st.session_state:
            st.session_state.llm = Ollama(
                base_url="http://localhost:11434",
                model=st.session_state.selected_model,
                verbose=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )
            st.session_state.embedding_model = OllamaEmbeddings(model=st.session_state.selected_model)

        # Initialize vectorstore, qa_chain, retriever, and chat history if not already set
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = None
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = None
        if 'retriever' not in st.session_state:
            st.session_state.retriever = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []  # Initialize chat history if not present

        # Add custom CSS to style the select box
        st.markdown(
            """
            <style>
            /* Style adjustments */
            .row-widget.stSelectbox {
                width: 700px !important;
            }
            .css-1x8cf1d edgvbvh10 {
                border-radius: 1.25rem;
            }
            .css-po3vlj {
                padding: 10rem !important;
                border-radius: 9.25rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Category selection
        category_options = get_categories_from_db()

        # Add option for creating a new category if admin
        if st.session_state.is_admin:
            category_options.append("+ Create New Category")

        # Check if files have been uploaded, stored in session state
        if 'files_uploaded' not in st.session_state:
            st.session_state.files_uploaded = False
            st.session_state.uploaded_file_names = []

        # If files have not been uploaded, show category selection and file uploader
        if not st.session_state.files_uploaded:
            selected_category = st.selectbox(
                "Select Category and Upload File", category_options, help="Choose the category for file upload"
            )

            # Store the selected category in session state
            st.session_state.selected_category = selected_category

            # Handle new category creation
            if selected_category == "+ Create New Category" and st.session_state.is_admin:
                new_category = st.sidebar.text_input("Enter New Category Name")
                if st.sidebar.button("OK") and new_category:
                    if db['knowledge_base'].find_one({"category_name": new_category}):
                        st.sidebar.error("Category already exists.")
                    else:
                        add_category_to_db(new_category)
                        st.sidebar.success(f"'{new_category}' added to categories.")
                        st.experimental_rerun()
            else:
                category_dir = os.path.join(DATA_DIR, selected_category)
                os.makedirs(category_dir, exist_ok=True)
                st.markdown("""
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">

    <style>
    /* Container with subtle colors and smoother borders */
    .css-po3vlj {
        background: linear-gradient(135deg, #e8f5e9, #f9fbe7);
        border: 1px solid #a5d6a7;
        padding: 30px;
        text-align: center;
        border-radius: 12px;
        box-shadow: 0px 3px 12px rgba(0, 0, 0, 0.05);
        transition: background 0.4s ease, transform 0.2s ease-in-out;
    }

    .css-po3vlj:hover {
        background: linear-gradient(135deg, #d1c4e9, #f3e5f5);
        transform: scale(1.04);
        border-color: #b39ddb;
        box-shadow: 0px 5px 16px rgba(0, 0, 0, 0.08);
    }

    /* Icon bounce effect with subtle colors */
    .css-nwtri svg {
        fill: #81c784;
        width: 45px;
        height: 45px;
        transition: fill 0.3s ease-in-out, transform 0.3s ease;
        animation: icon-bounce 1.2s infinite;
    }

    .css-po3vlj:hover .css-nwtri svg {
        fill: #b39ddb;
        transform: rotate(10deg);
    }

    @keyframes icon-bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-6px);
        }
    }

    /* File upload content with a modern feel */
    .file-upload-content {
        font-size: 16px;
        font-weight: bold;
        color: #388e3c;
    }

    .css-1aehpvj {
        font-size: 13px;
        color: #757575;
        margin-top: 8px;
    }

    /* Button styling using Bootstrap principles */
    .css-1x8cf1d {
        background-color: #66bb6a;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        cursor: pointer;
        transition: background-color 0.2s ease-in-out, box-shadow 0.3s;
    }

    .css-1x8cf1d:hover {
        background-color: #43a047;
        box-shadow: 0px 3px 12px rgba(0, 0, 0, 0.1);
    }

    /* Compact and subtle dropdown */
    .row-widget.stSelectbox {
        background: linear-gradient(135deg, #f1f8e9, #e8f5e9);
        border: 1px solid #a5d6a7;
        border-radius: 8px;
        padding: 8px;
        width: 180px; /* Adjusted width for compactness */
        font-size: 14px; /* Slightly smaller font size */
        box-shadow: 0px 3px 8px rgba(0, 0, 0, 0.08);
        transition: background 0.4s ease, transform 0.2s ease-in-out;
    }

    .row-widget.stSelectbox:hover {
        background: linear-gradient(135deg, #d1c4e9, #e8eaf6);
        transform: scale(1.03);
        border-color: #b39ddb;
        box-shadow: 0px 5px 14px rgba(0, 0, 0, 0.1);
    }

    /* Adjusted font size for dropdown label */
    .st-af {
        font-size: 14px;
        font-weight: bold;
        color: #388e3c;
    }

    .css-81oif8 {
        font-size: 16px;
        font-weight: bold;
        color: #388e3c;
    }

    </style>
    """, unsafe_allow_html=True)

                uploaded_files = st.file_uploader("", type=['xlsx', 'xls', 'pdf', 'docx'], accept_multiple_files=True)

                if uploaded_files:
                    # Store the uploaded file names in session state
                    for uploaded_file in uploaded_files:
                        with open(os.path.join(category_dir, uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.session_state.uploaded_file_names.append(uploaded_file.name)

                    st.success(f"Uploaded {len(uploaded_files)} files to '{selected_category}' category.")

                    # Process uploaded files
                    process_uploaded_files(uploaded_files, category_dir)

                    # Mark files as uploaded
                    st.session_state.files_uploaded = True
                    st.experimental_rerun()  # Reload the app to hide file uploader and category selection
        else:
            # Files are uploaded, show uploaded file list with a cancel option
            st.sidebar.write("### Uploaded Files:")
            files_to_delete = []

            for idx, file_name in enumerate(st.session_state.uploaded_file_names):
                col1, col2 = st.sidebar.columns([4, 1])
                with col1:
                    st.sidebar.write(file_name)
                with col2:
                    if st.sidebar.button("❌", key=f"delete_{idx}"):
                        files_to_delete.append(file_name)

            # Remove files marked for deletion
            if files_to_delete:
                for file_name in files_to_delete:
                    st.session_state.uploaded_file_names.remove(file_name)
                    # Optionally remove from directory too
                    file_path = os.path.join(category_dir, file_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                st.experimental_rerun()

            # Display model selection and chat interface
            selected_model = st.sidebar.selectbox("Select Model", available_models, index=available_models.index(st.session_state.selected_model))

            # Update LLM if the selected model has changed
            if st.session_state.selected_model != selected_model:
                st.session_state.selected_model = selected_model
                st.session_state.memory.clear()
                st.session_state.chat_history = []

                # Update LLM and embedding model
                st.session_state.llm = Ollama(
                    base_url="http://localhost:11434",
                    model=selected_model,
                    verbose=True,
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                )
                st.session_state.embedding_model = OllamaEmbeddings(model=selected_model)

                # Initialize QA chain if vectorstore is set
                if st.session_state.vectorstore:
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=st.session_state.llm,
                        chain_type='stuff',
                        retriever=st.session_state.vectorstore.as_retriever(),
                        verbose=True,
                        chain_type_kwargs={
                            "verbose": True,
                            "prompt": st.session_state.prompt,
                            "memory": st.session_state.memory,
                        }
                    )
                st.success(f"Model updated to '{selected_model}'")

            # If QA chain is not set, initialize it
            if st.session_state.qa_chain is None and st.session_state.vectorstore:
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=st.session_state.llm,
                    chain_type='stuff',
                    retriever=st.session_state.vectorstore.as_retriever(),
                    verbose=True,
                    chain_type_kwargs={
                        "verbose": True,
                        "prompt": st.session_state.prompt,
                        "memory": st.session_state.memory,
                    }
                )

            # Chat functionality will now be available only after files are uploaded
            st.write("---")

            # Layout containers
            chat_container = st.container()

            # Display Chat History
            with chat_container:
                st.subheader("💬 Chat")
                for message in st.session_state.chat_history:
                    if message['role'] == 'user':
                        st.markdown(
                            f"<div class='chat-container'><p class='message user'>👤 {message['message']}</p></div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='chat-container'><p class='message assistant'>🧠 {message['message']}</p></div>",
                            unsafe_allow_html=True
                        )

            # Input Field at the Bottom (conditional on file upload)
            
            st.markdown("""
    <style>
    /* Style for the text input container */
    .css-pb6fr7.edfmue0 {
        display: flex;
        align-items: center;
        border: 2px solid #80deea;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0px 3px 10px rgba(0, 0, 0, 0.1);
        background: linear-gradient(135deg, #e0f7fa, #f0f4c3);
    }

    /* Style for the input field */
    .css-pb6fr7.edfmue0 input[type="text"] {
        border: none;
        outline: none;
        flex-grow: 1;
        padding: 10px;
        border-radius: 10px;
        font-size: 16px;
        background: transparent;
        color: #00796b;
    }

    /* Placeholder text styling */
    .css-pb6fr7.edfmue0 input::placeholder {
        color: #9e9e9e;
    }

    /* Style for the send icon */
    .send-icon {
        cursor: pointer;
        color: #4CAF50;
        margin-left: 10px;
        font-size: 24px;
        transition: color 0.3s ease-in-out;
    }

    .send-icon:hover {
        color: #388E3C;
    }

    /* Style for the voice recognition icon */
    .voice-icon {
        cursor: pointer;
        color: #4CAF50;
        margin-right: 10px;
        font-size: 24px;
        transition: color 0.3s ease-in-out;
    }

    .voice-icon:hover {
        color: #388E3C;
    }
    </style>
    """, unsafe_allow_html=True)
            
            input_container = st.container()
            with input_container:
                st.write("---")
                user_input = st.text_input(
                    "💬 Ask a question:",
                    key="user_input",
                    placeholder="Type your question here...",
                    label_visibility='collapsed'
                )

            # Translation
            def safe_translate(text, dest_lang):
                try:
                    translated = translator.translate(text, dest=dest_lang)
                    return translated.text
                except Exception as e:
                    st.error(f"Translation failed: {str(e)}")
                    return text
                
            # Overall MongoDB Saving Function
            def save_to_mongo(user_input, response_text, translated_response):
                current_time = datetime.now().strftime('%d-%m-%Y')
                chat_data = {
                    "username": st.session_state.username,
                    "user_query": user_input,
                    "assistant_response": response_text,
                    "translated_response": translated_response,
                    "timestamp": current_time,
                    "model_name": st.session_state.selected_model,
                    "category": st.session_state.get('selected_category', None)
                }
                save_chat_to_db(collection, chat_data)
            

            if st.button("Submit"):
                if user_input.strip():
                    user_message = {"role": "user", "message": user_input}
                    st.session_state.chat_history.append(user_message)
                    st.markdown(
                        f"<div class='chat-container'><p class='message user'>👤 {user_input}</p></div>",
                        unsafe_allow_html=True
                    )

                    with st.spinner("Assistant is typing..."):
                        if st.session_state.qa_chain:
                            response = st.session_state.qa_chain({"query": user_input})
                            response_text = response['result']

                            translated_response = safe_translate(response_text, 'ta')

                            # Save to MongoDB
                            save_to_mongo(user_input, response_text, translated_response)

                            assistant_message = {"role": "assistant", "message": response_text}
                            st.session_state.chat_history.append(assistant_message)

                            st.markdown(
                                f"<div class='chat-container'><p class='message assistant'>🧠 {response_text}</p></div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.error("QA chain not initialized. Please upload a document to start the chat.")
                else:
                    st.warning("Please enter a question before submitting.")         
                
    with st.sidebar:
        selected_option = option_menu(
            menu_title="",  
            options=["Home", "Filter Chats", "Chat","Models" , "Logout"],  
            icons=["house", "chat-dots", "chat-left-text","box" ,"box-arrow-right"],  
            menu_icon="cast",  
            default_index=0,  
            orientation="vertical", 
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "#ffffff",  
                },
                "icon": {
                    "color": "#1a73e8",  
                    "font-size": "16px",  
                    "margin-right": "10px",  
                },
                "nav-link": {
                    "font-size": "16px",  
                    "margin": "10px 0",  
                    "color": "#000000",  
                },
                "nav-link-selected": {
                    "background-color": "#f0f0f0",  
                    "color": "#1a73e8",  
                },
            },
        )
        
    # Logout 
    def logout():
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.is_admin = False
        st.session_state.show_signup = False 
        st.success("You have been logged out.")
        st.experimental_rerun()
             
    # Display 
    if selected_option == "Home":
        st.markdown(
        """
        <style>
        /* Full height layout with horizontal alignment */
        body, html {
            overflow: hidden;  /* Disable scrolling */
            margin: 0;  /* Remove default margin */
            padding: 0;  /* Remove default padding */
            font-family: Arial, sans-serif;  /* Smooth font */
            background-color: #ffffff;  /* White background for the body */
            display: flex
        }

        /* Sidebar styles */
        .sidebar {
            width: 300px;  /* Fixed width for sidebar */
            height: 100vh;  /* Full height */
            background-color: #f8f9fa;  /* Light gray background for sidebar */
            position: fixed;  /* Fixed position */
            left: 0;  /* Align to left */
            top: 0;  /* Align to top */
            padding: 20px;  /* Padding for content */
            box-shadow: 2px 0px 5px rgba(0, 0, 0, 0.1);  /* Shadow for depth */
        }

        /* Main content styles */
        .main-content {
            width:100%;
            padding: 20px;  /* Padding for main content */
            display: flex;  /* Flexbox for horizontal alignment */
            align-items: center;  /* Center vertically */
            justify-content: center;  /* Center horizontally */
        }

        /* Welcome box with smooth design */
        .welcome-box {
            background-color: white;  /* White box */
            border-radius: 15px;
            padding: 40px;
            max-width: 800px;
            width: 100%;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .welcome-box:hover {
            transform: translateY(-5px);  /* Soft hover effect */
            box-shadow: 0px 12px 25px rgba(0, 0, 0, 0.2);  /* Enhance shadow on hover */
        }

        /* Welcome header */
        .welcome-header {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 20px;
            color: #1a73e8;  /* Match navbar icon color */
        }

        /* Welcome message */
        .welcome-message {
            font-size: 1.2em;
            line-height: 1.6;
            color: #555;  /* Slightly lighter gray for readability */
        }

        /* Highlight key text */
        .highlight {
            color: #1a73e8;  /* Bright color for highlights, matching navbar */
            font-weight: bold;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .welcome-header {
                font-size: 2em;
            }
            .welcome-message {
                font-size: 1em;
            }
            .sidebar {
                width: 200px;  /* Adjust sidebar width for smaller screens */
            }
            .main-content {
                margin-left: 210px;  /* Adjust margin for smaller screens */
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

        # Main content area for welcome message
        st.markdown(
            """
            <div class='main-content'>
                <div class='welcome-box'>
                    <div class='welcome-header'>
                        Welcome, {username}!
                    </div>
                    <div class='welcome-message'>
                        👋 We're excited to have you here! Ready to dive into some AI-powered conversations?<br><br>
                        Simply <span class='highlight'>click on "Chat"</span> in the menu to start interacting with our intelligent models.<br><br>
                        🔍 Want to explore specific topics? Select a <span class='highlight'>category</span> and <span class='highlight'>upload your file</span>, and watch the AI bring your data to life!<br><br>
                        Let the conversation begin! 🚀
                    </div>
                </div>
            </div>
            """.format(username=st.session_state.username.capitalize()),
            unsafe_allow_html=True
        )
            
    elif selected_option =="Filter Chats":
        load_chat_retrieval()
        
    elif selected_option == "Chat":
        initiate_chat()
        
    elif selected_option == "Models":   
        available_models = list_models(MODEL_DIR)
        if available_models:
            st.subheader(" Available Models:")
            
            # Create a DataFrame to hold the model names with serial numbers
            model_data = pd.DataFrame({
                "S.No.": range(1, len(available_models) + 1),
                "Model Name": available_models
            })
            
            # Render the table using styled HTML
            st.markdown(model_data.to_html(index=False, classes="model-table"), unsafe_allow_html=True)
        else:
            st.warning("⚠️ No models available in the specified directory.")
            
    elif selected_option == "Logout":
        logout()
        