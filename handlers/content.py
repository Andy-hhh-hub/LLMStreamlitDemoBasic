#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：llm-stream-reponse-lambda
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/12/5 9:41
# ====================================
import json
from langchain.llms.sagemaker_endpoint import LLMContentHandler


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, **model_kwargs})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))  # [0][0]
        '''
        if len(response_json)==1:  # 单独一个
            response_json = response_json[0][0]
        elif len(response_json)>1:  # 多个推理时
            response_json = response_json
#         print(np.array(response_json).shape())
        '''
        return response_json


class ContentHandlerQA(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        print(prompt, model_kwargs)
        input_str = json.dumps({"inputs": prompt, **model_kwargs})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["answers"]
