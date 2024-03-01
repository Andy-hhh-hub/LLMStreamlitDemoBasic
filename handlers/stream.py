#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：llm-stream-reponse-lambda
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/11/3 15:12 
# ====================================
import json
import logging
import time
import boto3
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.pydantic_v1 import Extra, root_validator
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, Dict, List, Union, Mapping, Optional, TypeVar, Union
from langchain.schema import LLMResult
from langchain.llms.base import LLM
import io

logger = logging.getLogger()


def format_reference(recall_knowledge):
    text = '\n```json\n#Reference\n'
    for sn, item in enumerate(recall_knowledge):
        displaydata = {"doc": item['doc'], "score": item['score']}
        doc_category = item['doc_category']
        doc_title = item['doc_title']
        text += f'Doc[{sn + 1}]:["{doc_title}"]-["{doc_category}"]\n{json.dumps(displaydata, ensure_ascii=False)}\n'
    text += '\n```'
    return text


class StreamScanner:
    def __init__(self):
        self.buff = io.BytesIO()
        self.read_pos = 0

    def write(self, content):
        self.buff.seek(0, io.SEEK_END)
        self.buff.write(content)

    def readlines(self):
        self.buff.seek(self.read_pos)
        for line in self.buff.readlines():
            if line[-1] != b'\n':
                self.read_pos += len(line)
                yield line[:-1]

    def reset(self):
        self.read_pos = 0


class CustomStreamingOutCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def __init__(self, wsclient: str, messageId: str, connectionId: str, model_name: str, stream: bool,

                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wsclient = wsclient
        self.connectionId = connectionId
        self.msgid = messageId
        self.model_name = model_name
        self.recall_knowledge = []
        self.stream = stream

    def add_recall_knowledge(self, recall_knowledge):

        self.recall_knowledge = recall_knowledge

    def postMessage(self, data):
        try:
            self.wsclient.post_to_connection(Data=data.encode('utf-8'), ConnectionId=self.connectionId)
            logger.info(f"stream chunk:{data}")
        except Exception as e:
            logger.error(f'post {data} to_wsconnection error:{str(e)}')

    def message_format(self, messages):
        """Format messages as ChatGPT who only accepts roles of ['system', 'assistant', 'user']"""
        return [
            {'role': 'assistant', 'content': msg['content']}
            if msg['role'] == 'assistant'
            else {'role': 'user', 'content': msg['content']}
            for msg in messages
        ]

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if self.stream:
            data = json.dumps(
                {
                    "message": "chunk",
                    "object": "chat.completion.chunk",
                    "created": time.time(),
                    'msgid': self.msgid,
                    'role': "assistant",
                    "body": token,
                    "finish_reason": None,
                    'connectionId': self.connectionId
                },
                ensure_ascii=False)
            self.postMessage(data)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        pass
        # if (not self.hide_ref) and self.use_stream:
        #     text = format_reference(self.recall_knowledge)
        #     data = json.dumps(
        #         {
        #             "message": "chunk",
        #             "object": "chat.completion.chunk",
        #             "created": time.time(),
        #             'msgid': self.msgid,
        #             'role': "assistant",
        #             'text': {'content': f'{text}'},
        #             "finish_reason": "stop",
        #             'connectionId': self.connectionId
        #         },
        #         ensure_ascii=False)
        #     self.postMessage(data)

    def on_chain_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        data = json.dumps(
            {
                "message": "chunk",
                "object": "chat.completion.chunk",
                "created": time.time(),
                'msgid': self.msgid,
                'role': "assistant",
                "body": str(error[0]),
                'connectionId': self.connectionId
            },
            ensure_ascii=False)
        self.postMessage(data)


class SagemakerStreamContentHandler(LLMContentHandler):
    content_type: Optional[str] = "application/json"
    accepts: Optional[str] = "application/json"
    callbacks: BaseCallbackHandler

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def __init__(self, callbacks: BaseCallbackHandler, stop, frequency: int = 5, **kwargs) -> None:

        super().__init__(**kwargs)
        self.callbacks = callbacks
        self.stop = stop
        self.frequency = frequency

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        print(prompt, model_kwargs)
        input_str = json.dumps({'inputs': prompt, **model_kwargs})
        # logger.info(f'transform_input:{input_str}')
        return input_str.encode('utf-8')

    def transform_output(self, event_stream: Any) -> str:
        scanner = StreamScanner()
        text = ''
        count = 0
        for event in event_stream:
            scanner.write(event['PayloadPart']['Bytes'])
            for line in scanner.readlines():
                try:
                    resp = json.loads(line)
                    token = resp.get("outputs")['outputs']
                    text += token
                    for stop in self.stop:  ##如果碰到STOP截断
                        if text.endswith(stop):
                            self.callbacks.on_llm_end(None)
                            text = text.rstrip(stop)
                            return text
                    # self.callbacks.on_llm_new_token(token)
                    count += 1
                    if count % self.frequency == 0:
                        self.callbacks.on_llm_new_token(text)
                    # print(token, end='')
                except Exception as e:
                    # print(line)
                    continue
        self.callbacks.on_llm_end(None)
        return text


class SagemakerStreamEndpoint(LLM):
    endpoint_name: str = ""
    region_name: str = ""
    content_handler: LLMContentHandler
    model_kwargs: Optional[Dict] = None
    endpoint_kwargs: Optional[Dict] = None
    streaming: bool = False

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that AWS credentials to and python package exists in environment."""
        try:
            session = boto3.Session()
            values["client"] = session.client(
                "sagemaker-runtime", region_name=values["region_name"]
            )
        except Exception as e:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                "profile name are valid."
            ) from e
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_name": self.endpoint_name},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "sagemaker_stream_endpoint"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        _model_kwargs = self.model_kwargs or {}
        _model_kwargs = {**_model_kwargs, **kwargs}
        _endpoint_kwargs = self.endpoint_kwargs or {}

        body = self.content_handler.transform_input(prompt, _model_kwargs)
        content_type = self.content_handler.content_type
        accepts = self.content_handler.accepts
        if self.streaming:
            # send request
            try:
                response = self.client.invoke_endpoint_with_response_stream(
                    EndpointName=self.endpoint_name,
                    Body=body,
                    ContentType=content_type,
                    Accept=accepts,
                    **_endpoint_kwargs,
                )
            except Exception as e:
                raise ValueError(f"Error raised by inference endpoint: {e}")
        else:
            try:
                response = self.client.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    Body=body,
                    ContentType=content_type,
                    Accept=accepts,
                    **_endpoint_kwargs,
                )
            except Exception as e:
                raise ValueError(f"Error raised by inference endpoint: {e}")

        text = self.content_handler.transform_output(response["Body"])
        return text


class ContentHandler(EmbeddingsContentHandler):
    parameters = {
        "max_new_tokens": 50,
        "temperature": 0,
        "min_length": 10,
        "no_repeat_ngram_size": 2,
    }

    def transform_input(self, inputs: list[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": inputs, **model_kwargs})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["sentence_embeddings"]


class llmContentHandler(LLMContentHandler):
    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({'inputs': prompt, **model_kwargs})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["outputs"]
