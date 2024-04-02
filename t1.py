import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import BedrockChat
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
