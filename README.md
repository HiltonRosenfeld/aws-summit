# RAG on Bedrock

This demo is available to run on [Streamlit Cloud](https://aws-summit-sydney-2024-demo.streamlit.app), or you can run the demo yourself using the instrcutions below

The demo will ask each user to provide a username. That is used to keep seperate each user's vector context and conversation history.



### Setup to run your own instance

1. Create Streamlit Secrets
The following attributes must be provided in your secrets.toml.
[See Streamlit docs fro more help](https://docs.streamlit.io/develop/concepts/connections/secrets-management)

    - ASTRA_VECTOR_ENDPOINT
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_DEFAULT_REGION

2. Run the app

    `streamlit run app_bedrock.py`



## Using the demo

1. Provide content

    - Use the sidebar to provide content that will be referenced in the RAG workflow.
    - You can choose from:
        - Uploading a file
        - Providing a URL 
        - Providing text

2. Select your preferred LLM from the available Bedrock Foundation Models

3. Ask a question