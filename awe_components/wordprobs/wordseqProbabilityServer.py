#!/usr/bin/env python3
# Copyright 2022, Educational Testing Service

import asyncio
import enum
import json
import sys
import websockets

from distutils.sysconfig import get_python_lib

from awe_components.wordprobs.wordProbabilityInContext \
    import WordProbabilityInContext

if sys.version_info[0] == 3:
    xrange = range

transformer_types = enum.Enum('Transformer Type', "NONE BERT NEUSPELL")


class wordseqProbabilityServer:

    wpic = None

    def __init__(self):

        self.wpic = WordProbabilityInContext()

        asyncio.get_event_loop().run_until_complete(
            websockets.serve(self.run_wordseqprobs, 'localhost', 8775))
        print('running')
        asyncio.get_event_loop().run_forever()
        print('died')

    async def kill(self, websocket):
        await websocket.close()
        exit()

    async def run_wordseqprobs(self, websocket, path):
        async for message in websocket:

            messagelist = json.loads(message)
            print(messagelist)
            print(len(messagelist))
            if messagelist == ['kill()']:
                await self.kill(websocket)
            elif len(messagelist) == 3:
                print('processing')
                word = messagelist[0]
                context = ''.join(messagelist[1]) \
                          + ' [MASK] ' + \
                          ''.join(messagelist[2])

                print(word, context)
                probability = self.wpic.probabilityInContext(word, context)
                await websocket.send(json.dumps([probability]))
            else:
                print('error')
                await websocket.send(json.dumps([0]))


if __name__ == '__main__':
    wsps = wordseqProbabilityServer()
