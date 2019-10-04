import asyncio
import json
import os
import sys

import websockets
from client import Action, GustClient


class GustRemote:
    def __init__(
        self,
        wind_url=None,
        model_id=None,
        request_id=None,
        gust_id=None,
        gust_mode=None,
    ):
        self.wind_url = wind_url
        self.model_id = model_id
        self.request_id = request_id
        self.gust_id = gust_id
        self.gust_mode = gust_mode
        self._inference_function = None
        self._load_function = None
        self._preprocess_function = None
        self._train_function = None
        self._client = None
        self._websocket = None

    @classmethod
    def from_env(cls):
        wind_url = os.environ.get("WIND_URL")
        model_id = os.environ.get("MODEL_ID")
        gust_id = os.environ.get("GUST_ID")
        gust_mode = os.environ.get("GUST_MODE")
        request_id = os.environ.get("REQUEST_ID")

        if not wind_url:
            raise Exception("WIND_URL env variable not set")
        if not model_id:
            raise Exception("MODEL_ID env variable not set")
        if not gust_id:
            raise Exception("GUST_ID env variable not set")
        if not gust_mode:
            raise Exception("GUST_MODE env variable not set")
        if not request_id:
            raise Exception("REQUEST_ID env variable not set")

        return cls(
            wind_url=wind_url,
            model_id=model_id,
            gust_id=gust_id,
            gust_mode=gust_mode,
            request_id=request_id,
        )

    def load(self, func):
        self._load_function = func

    def inference(self, func):
        self._inference_function = func

    def preprocess(self, func):
        self._preprocess_function = func

    def train(self, func):
        self._train_function = func

    def _reply(self, message):
        return self._loop.create_task(self.send_json(self._websocket, message))

    def start(self):
        self._loop = asyncio.get_event_loop()
        print(f"Starting gust instance {self.gust_id}")

        def replier(message):
            self._loop.call_soon_threadsafe(self._reply, message)

        self._client = GustClient(
            replier,
            load_function=self._load_function,
            preprocess_function=self._preprocess_function,
            train_function=self._train_function,
            inference_function=self._inference_function,
        )
        self._client.start()
        self._loop.run_until_complete(self._main_loop())

    async def send_json(self, websocket, dict):
        await websocket.send(json.dumps(dict))

    async def _main_loop(self):
        uri = f"ws://{self.wind_url}/ws/gust/container/{self.gust_id}/{self.model_id}/{self.gust_mode}"
        try:
            async with websockets.connect(uri) as websocket:
                self._websocket = websocket
                self._client.run_action(
                    Action(
                        command="load",
                        request_id=self.request_id,
                        payload={"model_id": self.model_id},
                    )
                )
                self._client.run_action(
                    Action(
                        command="initiate",
                        request_id=self.request_id,
                        payload={"model_id": self.model_id},
                    )
                )
                while True:
                    if not self._client.is_alive():
                        print("Shutting down gust, client thread died")
                        sys.exit(0)
                    try:
                        payload = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        payload = json.loads(payload)
                        action = Action(
                            command=payload["command"],
                            request_id=payload["request_id"],
                            payload=payload,
                        )
                        self._client.run_action(action)
                    except asyncio.TimeoutError:
                        pass
        except ConnectionRefusedError:
            print("Unable to connect to host")
        except websockets.exceptions.ConnectionClosedError:
            print("Connection closed")
