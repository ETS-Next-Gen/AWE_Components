#!/usr/bin/env python3
# Copyright 2022, Educational Testing Service

import asyncio
import websockets
import json
from websocket import create_connection


class WordseqProbClient:

    uri = None

    def __init__(self):
        self.uri = "ws://localhost:8775"

    def set_uri(self, uri):
        self.uri = uri

    def send(self, message: list):
        if message is None:
            print('no message!')
            return None
        try:
            ws = create_connection(self.uri, timeout=None)
            ws.send(json.dumps(message))
            result = ws.recv()
            ws.close()
            ws.on_message = None
            ws.on_open = None
            ws.close = None
            del ws
            ws = None
            return json.loads(result)
        except Exception as e:
            print(e)
            return None


if __name__ == '__main__':
    wsc = WordseqProbClient()

