#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：llm-stream-reponse-lambda
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：zhhao@deloitte.com.cn
# @Date    ：2023/11/7 16:58 
# ====================================
import json
import time


class WebSocketResponse:

    def __init__(self, messageId, connectionId):
        """
        初始化 Response 对象。

        Args:
            messageId (str): 消息的唯一标识符
            connectionId (str): WebSocket 连接的唯一标识符
        """
        self.messageId = messageId
        self.connectionId = connectionId

    def __call__(self, answer):
        """
        生成 WebSocket 响应数据并返回。

        Args:
            answer (list): 要发送给 WebSocket 连接的生成的文本

        Returns:
            dict: 包含 WebSocket 响应信息的字典
        """
        response_data = {
            "message": "completion",
            "object": "chat.completion",
            "created": time.time(),
            'msgid': self.messageId,
            'role': "AI",
            "body": answer,
            "finish_reason": "stop",
            'connectionId': self.connectionId
        }

        # 构建WebSocket响应
        response = {
            "statusCode": 200,  # 状态码（可根据需要自定义）
            "headers": {
                "Content-Type": "application/json"  # 指定响应内容类型
            },
            "isBase64Encoded": False,  # 是否使用Base64编码
            "body": json.dumps(response_data, ensure_ascii=False)  # 将消息内容编码为JSON字符串
        }
        return response


class LambdaResponse:

    def __call__(self, answer):
        """
        生成 WebSocket 响应数据并返回。

        Args:
            answer (list): 要发送给 WebSocket 连接的生成的文本

        Returns:
            dict: 包含 WebSocket 响应信息的字典
        """

        # 构建WebSocket响应
        response = {
            "statusCode": 200,  # 状态码（可根据需要自定义）
            "headers": {
                "Content-Type": "application/json"  # 指定响应内容类型
            },
            "isBase64Encoded": False,  # 是否使用Base64编码
            "body": json.dumps(answer, ensure_ascii=False)  # 将消息内容编码为JSON字符串
        }
        return response
