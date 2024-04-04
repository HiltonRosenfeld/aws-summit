import os

from langchain_community.chat_models.bedrock import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
#ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]

ASTRA_VECTOR_ENDPOINT = os.getenv("ASTRA_VECTOR_ENDPOINT")
#ASTRA_VECTOR_ENDPOINT = st.secrets["ASTRA_VECTOR_ENDPOINT"]
ASTRA_DB_KEYSPACE = "default_keyspace"
ASTRA_DB_COLLECTION = "awssummit"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
#AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
#AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
#AWS_DEFAULT_REGION = st.secrets["AWS_DEFAULT_REGION"]


#os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
#os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
#os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION

#os.environ["LANGCHAIN_PROJECT"] = "awssummit"
#os.environ["LANGCHAIN_TRACING_V2"] = "true"

print("Started")


###
### Use a Boto3 Client to authenticate with Bedrock Runtime
###
import boto3

session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)
bedrock_runtime = boto3.client("bedrock-runtime")

###
### Bedrock Embeddings
### WORKING
###
bedrock_embeddings = BedrockEmbeddings(
    client=bedrock_runtime,
    model_id="cohere.embed-english-v3", #1024
    #model_id="amazon.titan-embed-text-v1", #1536
    region_name="us-east-1",
    )

embedding = bedrock_embeddings.embed_query("This is a content of the document")


###
### Bedrock Chat
###      Claude-v2 - working
###      Claude-v3-Sonnet - working
###      Amazon Titan - working
###      Meta Llama2 13b - working
###      Meta Llama2 70b - working

###
chat = BedrockChat(
    client=bedrock_runtime,
    #model_id="anthropic.claude-v2", 
    #model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    #model_id="amazon.titan-text-express-v1",
    #model_id="meta.llama2-13b-chat-v1",
    #model_id="meta.llama2-70b-chat-v1",
    model_kwargs={"temperature": 0.1}
    )


messages = [
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    )
]

result = chat(messages)
print(result.content)
