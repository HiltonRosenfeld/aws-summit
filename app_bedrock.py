import os
from pathlib import Path
import boto3

from langchain_openai import ChatOpenAI
from langchain_community.chat_models.bedrock import BedrockChat
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import BedrockEmbeddings

from langchain_community.vectorstores.astradb import AstraDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory, AstraDBChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.runnable import RunnableMap
from langchain.callbacks.base import BaseCallbackHandler

import streamlit as st
import tempfile


ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]

ASTRA_VECTOR_ENDPOINT = st.secrets["ASTRA_VECTOR_ENDPOINT"]
ASTRA_DB_KEYSPACE = "default_keyspace"
ASTRA_DB_COLLECTION = "awssummit"

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = st.secrets["AWS_DEFAULT_REGION"]

os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION

os.environ["LANGCHAIN_PROJECT"] = "awssummit"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


print("Started")


# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

#################
### Constants ###
#################

# Define the number of docs to retrieve from the vectorstore and memory
top_k_vectorstore = 4
top_k_memory = 3

###############
### Globals ###
###############

global lang_dict
global rails_dict
global embedding
global vectorstore
global retriever
global model
global chat_history
global memory
global bedrock_runtime


#######################
### Resources Cache ###
#######################

# Cache boto3 session for future runs
@st.cache_resource(show_spinner='Getting the Boto Session...')
def load_boto_client():
    print("load_boto_client")
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )
    return boto3.client("bedrock-runtime")

# Cache OpenAI Embedding for future runs
@st.cache_resource(show_spinner='Getting the Embedding Model...')
def load_embedding():
    print("load_embedding")
    # Get the OpenAI Embedding
    #return OpenAIEmbeddings(model="text-embedding-3-small")
    # Get the Bedrock Embedding
    return BedrockEmbeddings(
        client=bedrock_runtime,
        #region_name="us-east-1"
    )
    

# Cache Vector Store for future runs
@st.cache_resource(show_spinner='Getting the Vector Store from Astra DB...')
def load_vectorstore():
    print(f"load_vectorstore: {ASTRA_DB_KEYSPACE} / {ASTRA_DB_COLLECTION}")
    # Get the load_vectorstore store from Astra DB
    return AstraDB(
        embedding=embedding,
        namespace=ASTRA_DB_KEYSPACE,
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_VECTOR_ENDPOINT,
    )
    
# Cache Retriever for future runs
@st.cache_resource(show_spinner='Getting the retriever...')
def load_retriever():
    print("load_retriever")
    # Get the Retriever from the Vectorstore
    return vectorstore.as_retriever(
        search_kwargs={"k": top_k_vectorstore}
    )

# Cache Chat Model for future runs
@st.cache_resource(show_spinner='Getting the Chat Model...')
def load_model(model_id="anthropic.claude-v2"):
    print(f"load_model: {model_id}")
    # if model_id contains 'openai' then use OpenAI model
    if 'openai' in model_id:
        if '3.5' in model_id:
            gpt_version = 'gpt-3.5-turbo'
        else:
            gpt_version = 'gpt-4-turbo-preview'
        return ChatOpenAI(
            temperature=0.2,
            model=gpt_version,
            streaming=True,
            verbose=False
            )
    # else use Bedrock model
    return BedrockChat(
        client=bedrock_runtime,
        model_id=model_id,
        streaming=True,
        #callbacks=[StreamingStdOutCallbackHandler()],
        model_kwargs={"temperature": 0.1},
    )
# Anthropic Claude - NOT WORKING - required keys prompt, max_tokens_to_sample
# Amazon Titan - WORKING
# Meta Lllama - WORKING




# Cache Chat History for future runs
@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_chat_history():
    print("load_chat_history")
    return AstraDBChatMessageHistory(
        session_id=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_VECTOR_ENDPOINT,
        namespace=ASTRA_DB_KEYSPACE,
    )

@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_memory():
    print("load_memory")
    return ConversationBufferWindowMemory(
        chat_memory=chat_history,
        return_messages=True,
        k=top_k_memory,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
    )

# Cache prompt
# 
#Include the price of the product if found in the context.
#You're a helpful AI assistant tasked to answer the user's questions
#You're friendly and you answer extensively with multiple sentences. 
#You prefer to use bulletpoints to summarize.
@st.cache_data()
def load_prompt():
    print("load_prompt")
    template = """You are Jerry Seinfeld speaking at a conference for technology professionals.
Focus on the user's needs and provide the best possible answer.
Do not include any information other than what is provied in the context below.
Do not include images in your response.
If you don't know the answer, just say 'I do not know the answer'.

Use the following context to answer the question:
{context}

Use the previous chat history to answer the question:
{chat_history}

Question:
{question}

Answer in English"""

    return ChatPromptTemplate.from_messages([("system", template)])


#################
### Functions ###
#################

# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_files):
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            
            # Write to temporary file
            temp_dir = tempfile.TemporaryDirectory()
            file = uploaded_file
            print(f"""Processing: {file.name}""")
            temp_filepath = os.path.join(temp_dir.name, file.name)
            print(f"""Processing: {temp_filepath}""")
            with open(temp_filepath, 'wb') as f:
                f.write(file.getvalue())

            # Create the text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1500,
                chunk_overlap  = 100
            )

            if uploaded_file.name.endswith('txt'):
                file = [uploaded_file.read().decode()]
                texts = text_splitter.create_documents(file, [{'source': uploaded_file.name}])
                vectorstore.add_documents(texts)
                st.info(f"{len(texts)} chunks loaded into Astra DB")            

            if uploaded_file.name.endswith('pdf'):
                # Read PDF
                docs = []
                loader = PyPDFLoader(temp_filepath)
                docs.extend(loader.load())

                pages = text_splitter.split_documents(docs)
                vectorstore.add_documents(pages)  
                st.info(f"{len(pages)} pages loaded into Astra DB")



#####################
### Session state ###
#####################

# Start with empty messages, stored in session state
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="How may I help you today?")]



############
### Main ###
############

# Write the welcome text
st.markdown(Path('welcome.md').read_text())

# DataStax logo
with st.sidebar:
    st.image('./public/logo.svg')
    st.text('')

# Initialize
with st.sidebar:
    bedrock_runtime = load_boto_client()
    embedding = load_embedding()
    vectorstore = load_vectorstore()
    retriever = load_retriever()
    model = load_model()
    chat_history = load_chat_history()
    memory = load_memory()
    prompt = load_prompt()


# Sidebar
with st.sidebar:
    # Add Data Input options
    with st.container(border=True):
        st.caption('Load data using one of the methods below.')
        with st.form('load_files'):
            uploaded_files = st.file_uploader("Upload a file", type=["txt", "pdf"], accept_multiple_files=True)
            submitted = st.form_submit_button('Upload')
            if submitted:
                with st.spinner('Chunking, Embedding, and Uploading to Astra'):
                    vectorize_text(uploaded_files)
        
        #st.button("Upload file to Astra DB", on_click=vectorize_text(uploaded_files))

        with st.form('load_url'):
            # option 2: enter URL
            url = st.text_input("Enter URL", "")
            # option 3: enter text
            text = st.text_area("Enter Text", "")
            
            submitted = st.form_submit_button('Embed')
            if submitted:
                #st.write(embedding.embed_query(text))
                print("Submitted")


    # Add a drop down to choose the LLM model
    with st.container(border=True):
        model_id = st.selectbox('Choose the LLM model', [
            'meta.llama2-13b-chat-v1',
            'meta.llama2-70b-chat-v1',
            'amazon.titan-text-express-v1',
            'anthropic.claude-v2', 
            'anthropic.claude-3-sonnet-20240229-v1:0',
            #'openai.gpt-3.5',
            #'openai.gpt-4'
            ])
        model = load_model(model_id)


    # Drop the Chat History
    with st.form('delete_memory'):
        st.caption('Delete the conversational memory.')
        submitted = st.form_submit_button('Delete conversational memory')
        if submitted:
            with st.spinner('Delete chat history'):
                memory.clear()

    # Delete Context
    with st.form('delete_context'):
        st.caption("Delete the context and conversational history.")
        submitted = st.form_submit_button("Delete context")
        if submitted:
            with st.spinner("Removing context and history..."):
                vectorstore.clear()
                chat_history.clear()
                st.session_state.messages = [AIMessage(content="How may I help you today?")]


# Draw all messages, both user and agent so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message.type).markdown(message.content)

# Now get a prompt from a user
if question := st.chat_input("What's up?"):
    print(f"Got question: \"{question}\"")

    # Add the prompt to messages, stored in session state
    st.session_state.messages.append(HumanMessage(content=question))

    # Draw the prompt on the page
    print("Display user prompt")
    with st.chat_message("user"):
        st.markdown(question)

    # Get the results from Langchain
    print("Get AI response")
    with st.chat_message("assistant"):
        # UI placeholder to start filling with agent response
        response_placeholder = st.empty()

        history = memory.load_memory_variables({})
        print(f"Using memory: {history}")

        inputs = RunnableMap({
            'context': lambda x: retriever.get_relevant_documents(x['question']),
            'chat_history': lambda x: x['chat_history'],
            'question': lambda x: x['question']
        })
        print(f"Using inputs: {inputs}")

        chain = inputs | prompt | model
        print(f"Using chain: {chain}")

        # Call the chain and stream the results into the UI
        response = chain.invoke({'question': question, 'chat_history': history}, config={'callbacks': [StreamHandler(response_placeholder)]})
        print(f"Response: {response}")
        #print(embedding.embed_query(question))
        content = response.content

        # Write the sources used
        relevant_documents = retriever.get_relevant_documents(question)
        content += f"""

*{"The following context was used for this answer:"}:*  
"""
        sources = []
        for doc in relevant_documents:
            source = doc.metadata['source']
            page_content = doc.page_content
            #title = doc.metadata['title']
            if source not in sources:
                content += f"""📙 :orange[{os.path.basename(os.path.normpath(source))}]  
"""
                sources.append(source)
        print(f"Used sources: {sources}")

        # Write the final answer without the cursor
        response_placeholder.markdown(content)


        # Add the result to memory
        memory.save_context({'question': question}, {'answer': content})

        # Add the answer to the messages session state
        st.session_state.messages.append(AIMessage(content=content))

