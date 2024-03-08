import json
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.pydantic_v1 import Extra, root_validator
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List, Union, Mapping, Optional, TypeVar, Union
import io


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


class SagemakerStreamContentHandler(LLMContentHandler):
    content_type: Optional[str] = "application/json"
    accepts: Optional[str] = "application/json"
    callbacks: BaseCallbackHandler

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def __init__(self, callbacks: BaseCallbackHandler, stop, stream, frequency: int = 5, **kwargs) -> None:

        super().__init__(**kwargs)
        self.callbacks = callbacks
        self.stop = stop
        self.frequency = frequency
        self.stream = stream

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        print(prompt, model_kwargs)
        input_str = json.dumps({'inputs': prompt, **model_kwargs})
        return input_str.encode('utf-8')

    def transform_output(self, event_stream: Any) -> str:
        if self.stream:
            scanner = StreamScanner()
            text = ''
            count = 0
            for event in event_stream:
                scanner.write(event)
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
        else:
            response_json = json.loads(event_stream.read().decode("utf-8"))
            return response_json["answers"]
