#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：LLMStreamlit
# @IDE     ：PyCharm
# @Author  ：Huang Andy Hong Hua
# @Email   ：
# @Date    ：2024/3/20 10:33
# ====================================
import streamlit as st
import os, json
import embed_pdf
from utils.sagemaker_endpoint import SagemakerEndpointEmbeddings
from handlers.content import ContentHandler, ContentHandlerQA
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

global own_llm, own_embeddings
region = 'cn-northwest-1'
EMBEDDING_ENDPOINT_NAME = "cmlm-bge-g4dn-endpoint"
content_handler = ContentHandler()
content_handler_qa = ContentHandlerQA()
own_embeddings = SagemakerEndpointEmbeddings(
    endpoint_name=EMBEDDING_ENDPOINT_NAME,
    region_name=region,
    content_handler=content_handler,
)
chatbot_config = json.load(open('./configs/config.json'))
own_llm = ChatOpenAI(api_key=chatbot_config["chatbot"]["moonshot_api_key"],
                     base_url=chatbot_config["chatbot"]["moonshot_api_base"],
                     model=chatbot_config["chatbot"]["moonshot_deployment_name"])
content_handler_qa = ContentHandlerQA()
# kwargs = {"temperature": 0.01}
kwargs ={"parameters":{"temperature": 0.01},"messages":[]}
llm = SagemakerEndpoint(endpoint_name=chatglm_endpoint_name, region_name=("cn-north-1"),
                        content_handler=content_handler_qa, model_kwargs=kwargs)

# messages = [
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="hi")
# ]
# print(own_llm(messages))
# print(own_embeddings.embed_query("test"))

# create sidebar and ask for openai api key if not set in secrets
secrets_file_path = os.path.join(".streamlit", "secrets.toml")
if os.path.exists(secrets_file_path):
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        else:
            print("OpenAI API Key not found in environment variables")
    except FileNotFoundError:
        print('Secrets file not found')
else:
    print('Secrets file not found')

if not os.getenv('OPENAI_API_KEY', '').startswith("sk-"):
    os.environ["OPENAI_API_KEY"] = st.sidebar.text_input(
        "OpenAI API Key", type="password"
    )
else:
    if st.sidebar.button("Embed Documents"):
        st.sidebar.info("Embedding documents...")
        try:
            embed_pdf.embed_all_pdf_docs(own_embeddings)
            st.sidebar.info("Done!")
        except Exception as e:
            st.sidebar.error(e)
            st.sidebar.error("Failed to embed documents.")

# create the app
st.title("Welcome to NimaGPT")

chosen_file = st.radio(
    "Choose a file to search", embed_pdf.get_all_index_files(), index=0
)

# check if openai api key is set
if not os.getenv('OPENAI_API_KEY', '').startswith("sk-"):
    st.warning("Please enter your OpenAI API key!", icon="⚠")
    st.stop()

# load the agent
from llm_helper import convert_message, get_rag_chain, get_rag_fusion_chain

rag_method_map = {
    'Basic RAG': get_rag_chain,
    'RAG Fusion': get_rag_fusion_chain
}
chosen_rag_method = st.radio(
    "Choose a RAG method", rag_method_map.keys(), index=0
)
get_rag_chain_func = rag_method_map[chosen_rag_method]
## get the chain WITHOUT the retrieval callback (not used)
# custom_chain = get_rag_chain_func(chosen_file)

# create the message history state
if "messages" not in st.session_state:
    st.session_state.messages = []

# render older messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# render the chat input
prompt = st.chat_input("Enter your message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # render the user's new message
    with st.chat_message("user"):
        st.markdown(prompt)

    # render the assistant's response
    with st.chat_message("assistant"):
        retrival_container = st.container()
        message_placeholder = st.empty()

        retrieval_status = retrival_container.status("**Context Retrieval**")
        queried_questions = []
        rendered_questions = set()


        def update_retrieval_status():
            for q in queried_questions:
                if q in rendered_questions:
                    continue
                rendered_questions.add(q)
                retrieval_status.markdown(f"\n\n`- {q}`")


        def retrieval_cb(qs):
            for q in qs:
                if q not in queried_questions:
                    queried_questions.append(q)
            return qs


        # get the chain with the retrieval callback
        custom_chain = get_rag_chain_func(chosen_file, retrieval_cb=retrieval_cb)

        if "messages" in st.session_state:
            chat_history = [convert_message(m) for m in st.session_state.messages[:-1]]
        else:
            chat_history = []

        full_response = ""
        for response in custom_chain.stream(
                {"input": prompt, "chat_history": chat_history}
        ):
            if "output" in response:
                full_response += response["output"]
            else:
                full_response += response.content

            message_placeholder.markdown(full_response + "▌")
            update_retrieval_status()

        retrieval_status.update(state="complete")
        message_placeholder.markdown(full_response)

    # add the full response to the message history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
