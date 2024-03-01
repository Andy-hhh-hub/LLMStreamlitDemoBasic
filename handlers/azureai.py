#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：llm-stream-reponse-lambda
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/11/7 16:18 
# ====================================
import json
import logging
import time
import openai

logger = logging.getLogger()


class OpenAIChatStreamClient:

    def __init__(self, wsclient, messageId, connectionId, stream, frequency=5):
        """
        初始化 OpenAIChatStreamClient。

        Args:
            wsclient: WebSocket 客户端，用于向客户端推送数据
            messageId: 消息的唯一标识符
            connectionId: WebSocket 连接的唯一标识符
        """
        self.wsclient = wsclient
        self.messageId = messageId
        self.connectionId = connectionId
        self.stream = stream
        self.frequency = frequency

    def postMessage(self, data):
        """
        向 WebSocket 连接的客户端发送数据。

        Args:
            data (str): 要发送的数据
        """
        try:
            self.wsclient.post_to_connection(Data=data.encode('utf-8'), ConnectionId=self.connectionId)
            logger.info(f"stream chunk:{data}")
        except Exception as e:
            logger.error(f'post {data} to_wsconnection error:{str(e)}')

    def construct_data(self, token):
        """
        构造要发送到 WebSocket 的数据。

        Args:
            token (str): 生成的文本

        Returns:
            str: 要发送到 WebSocket 的 JSON 格式数据
        """
        data = json.dumps(
            {
                "message": "chunk",
                "object": "chat.completion.chunk",
                "created": time.time(),
                'msgid': self.messageId,
                'role': "AI",
                "body": token,
                "finish_reason": None,
                'connectionId': self.connectionId
            },
            ensure_ascii=False)
        return data

    def stream_chat(self,
                    deployment_name,
                    messages,
                    temperature=0,
                    max_tokens=1024,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None):
        """
        使用 OpenAI ChatCompletion API 以流式方式进行对话。

        Args:
            deployment_name (str): 部署的 OpenAI 引擎的名称
            messages (list): 对话消息列表
            temperature (float): 温度参数

        Returns:
            str: 完整的响应文本
        """
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=True
        )

        collected_chunks = []
        collected_messages = []

        full_reply_content = ""
        count = 0
        for chunk in response:
            collected_chunks.append(chunk)
            count += 1

            if chunk['choices']:
                chunk_message = chunk['choices'][0]['delta']
                collected_messages.append(chunk_message)
                curr_chunk = chunk_message.get('content', '')

                full_reply_content += curr_chunk

                if self.stream and count % self.frequency == 0:
                    self.postMessage(self.construct_data(full_reply_content))

        # full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
        return full_reply_content
