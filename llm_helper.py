#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：LLMStreamlit
# @IDE     ：PyCharm
# @Author  ：Huang Andy Hong Hua
# @Email   ：
# @Date    ：2024/3/20 10:33
# ====================================
from typing import Optional

# langchain imports
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter
# from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
import json
from utils.sagemaker_endpoint import SagemakerEndpointEmbeddings
from handlers.content import ContentHandler, ContentHandlerQA
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from .handlers.stream import SagemakerStreamContentHandler

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
# messages = [
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="hi")
# ]
# print(own_llm(messages))

A_Role_en = "user"
RESET = '/rs'
STOP = [f"\n{A_Role_en}"]
stream =True
region = 'cn-northwest-1'
stream_content_handler = SagemakerStreamContentHandler(
            callbacks=StreamingStdOutCallbackHandler(), stop=STOP
        )
# content_handler_qa = LLMContentHandler()
parameters = {"top_k": 1, "top_p": 0}
model_kwargs = {'parameters': parameters, 'history': [], 'stream': True}
sagemaker_llm = SagemakerEndpoint(endpoint_name="baichuan2-7b-4bits-test-djl-p3-endpoint", region_name=region,
                        content_handler=stream_content_handler, model_kwargs=model_kwargs)


def format_docs(docs):
    res = ""
    # res = str(docs)
    for doc in docs:
        escaped_page_content = doc.page_content.replace("\n", "\\n")
        res += "<doc>\n"
        res += f"  <content>{escaped_page_content}</content>\n"
        for m in doc.metadata:
            res += f"  <{m}>{doc.metadata[m]}</{m}>\n"
        res += "</doc>\n"
    return res


def get_search_index(file_name="Mahmoudi_Nima_202202_PhD.pdf", index_folder="index"):
    global own_embeddings
    # load embeddings
    from langchain.vectorstores import FAISS
    from langchain.embeddings.openai import OpenAIEmbeddings

    search_index = FAISS.load_local(
        folder_path=index_folder,
        index_name=file_name + ".index",
        embeddings=own_embeddings,
    )
    return search_index


def convert_message(m):
    if m["role"] == "user":
        return HumanMessage(content=m["content"])
    elif m["role"] == "assistant":
        return AIMessage(content=m["content"])
    elif m["role"] == "system":
        return SystemMessage(content=m["content"])
    else:
        raise ValueError(f"Unknown role {m['role']}")


_condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {input}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_condense_template)

_rag_template = """Answer the question based only on the following context, citing the page number(s) of the document(s) you used to answer the question:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(_rag_template)


def _format_chat_history(chat_history):
    def format_single_chat_message(m):
        if type(m) is HumanMessage:
            return "Human: " + m.content
        elif type(m) is AIMessage:
            return "Assistant: " + m.content
        elif type(m) is SystemMessage:
            return "System: " + m.content
        else:
            raise ValueError(f"Unknown role {m['role']}")

    return "\n".join([format_single_chat_message(m) for m in chat_history])


def get_standalone_question_from_chat_history_chain():
    global own_llm
    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
                            | CONDENSE_QUESTION_PROMPT
                            | own_llm  # ChatOpenAI(temperature=0)
                            | StrOutputParser(),
    )
    return _inputs


def get_rag_chain(file_name="test.pdf", index_folder="index", retrieval_cb=None):
    global own_llm
    vectorstore = get_search_index(file_name, index_folder)
    retriever = vectorstore.as_retriever()

    if retrieval_cb is None:
        retrieval_cb = lambda x: x

    def context_update_fn(q):
        retrieval_cb([q])
        return q

    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
                            | CONDENSE_QUESTION_PROMPT
                            | own_llm  # ChatOpenAI(temperature=0)
                            | StrOutputParser(),
    )
    _context = {
        "context": itemgetter("standalone_question") | RunnablePassthrough(context_update_fn) | retriever | format_docs,
        "question": lambda x: x["standalone_question"],
    }
    conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | own_llm  # ChatOpenAI()
    return conversational_qa_chain


# RAG fusion chain
# source1: https://youtu.be/GchC5WxeXGc?si=6i7J0rPZI7SNwFYZ
# source2: https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1
def reciprocal_rank_fusion(results: list[list], k=60):
    from langchain.load import dumps, loads
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


def get_search_query_generation_chain():
    global own_llm
    from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
    prompt = ChatPromptTemplate(
        input_variables=['original_query'],
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=[],
                    template='You are a helpful assistant that generates multiple search queries based on a single input query.'
                )
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=['original_query'],
                    template='Generate multiple search queries related to: {original_query} \n OUTPUT (4 queries):'
                )
            )
        ]
    )

    generate_queries = (
            prompt |
            own_llm |  # ChatOpenAI(temperature=0)
            StrOutputParser() |
            (lambda x: x.split("\n"))
    )

    return generate_queries


def get_rag_fusion_chain(file_name="Mahmoudi_Nima_202202_PhD.pdf", index_folder="index", retrieval_cb=None):
    global own_llm
    vectorstore = get_search_index(file_name, index_folder)
    retriever = vectorstore.as_retriever()
    query_generation_chain = get_search_query_generation_chain()
    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
                            | CONDENSE_QUESTION_PROMPT
                            | own_llm  # ChatOpenAI(temperature=0) 使用自己的模型
                            | StrOutputParser(),
    )

    if retrieval_cb is None:
        retrieval_cb = lambda x: x

    _context = {
        "context":
            RunnablePassthrough.assign(
                original_query=lambda x: x["standalone_question"]
            )
            | query_generation_chain
            | retrieval_cb
            | retriever.map()
            | reciprocal_rank_fusion
            | (lambda x: [item[0] for item in x])
            | format_docs,
        "question": lambda x: x["standalone_question"],
    }
    # conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()
    conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | own_llm
    return conversational_qa_chain


####################
# Adding agent chain with OpenAI function calling

def get_search_tool_from_index(search_index, st_cb: Optional[StreamlitCallbackHandler] = None, ):
    from langchain.agents import tool
    from agent_helper import retry_and_streamlit_callback

    @tool
    @retry_and_streamlit_callback(st_cb=st_cb, tool_name="Content Seach Tool")
    def search(query: str) -> str:
        """Search the contents of the source document for the queries."""

        docs = search_index.similarity_search(query, k=5)
        return format_docs(docs)

    return search


def get_lc_oai_tools(file_name: str = "Mahmoudi_Nima_202202_PhD.pdf", index_folder: str = "index",
                     st_cb: Optional[StreamlitCallbackHandler] = None, ):
    from langchain.tools.render import format_tool_to_openai_tool
    search_index = get_search_index(file_name, index_folder)
    lc_tools = [get_search_tool_from_index(search_index=search_index, st_cb=st_cb)]
    oai_tools = [format_tool_to_openai_tool(t) for t in lc_tools]
    return lc_tools, oai_tools


def get_agent_chain(file_name="Mahmoudi_Nima_202202_PhD.pdf", index_folder="index", callbacks=None,
                    st_cb: Optional[StreamlitCallbackHandler] = None, ):
    global own_llm
    if callbacks is None:
        callbacks = []

    from langchain.agents import initialize_agent, AgentType
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.agents.format_scratchpad.openai_tools import (
        format_to_openai_tool_messages,
    )
    from langchain.agents import AgentExecutor
    from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

    lc_tools, oai_tools = get_lc_oai_tools(file_name, index_folder, st_cb)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful assistant, use the search tool to answer the user's question and cite only the page number when you use information coming (like [p1]) from the source document.\nchat history: {chat_history}"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
    llm = own_llm  # 使用自己的模型

    agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: _format_chat_history(x["chat_history"]),
            }
            | prompt
            | llm.bind(tools=oai_tools)
            | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=lc_tools, verbose=True, callbacks=callbacks)
    return agent_executor


if __name__ == "__main__":
    question_generation_chain = get_search_query_generation_chain()
    print('=' * 50)
    print('RAG Chain')
    chain = get_rag_chain()
    print(chain.invoke({'input': 'serverless computing', 'chat_history': []}))

    print('=' * 50)
    print('Question Generation Chain')
    print(question_generation_chain.invoke({'original_query': 'serverless computing'}))

    print('-' * 50)
    print('RAG Fusion Chain')
    chain = get_rag_fusion_chain()
    print(chain.invoke({'input': 'serverless computing', 'chat_history': []}))

    agent_executor = get_agent_chain()
    print(
        agent_executor.invoke({
            "input": "based on the source document, compare FaaS with BaaS??",
            "chat_history": [],
        })
    )
