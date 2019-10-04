import base64
import io
import threading
from dataclasses import dataclass
from queue import Queue

import darwin
from PIL import Image


@dataclass
class Action:
    command: str
    request_id: str
    payload: dict


class GustClient(threading.Thread):
    def __init__(
        self,
        replier,
        load_function,
        preprocess_function,
        train_function,
        inference_function,
    ):
        super().__init__()

        self._load_function = load_function
        self._preprocess_function = preprocess_function
        self._train_function = train_function
        self._inference_function = inference_function
        self._action_queue = Queue()
        self._replier = replier

    def run(self):
        while True:
            action = self._action_queue.get()
            print(f"Handling action: '{action.command}'")
            if action.command == "initiate":
                self.handle_initiate(action)
            elif action.command == "load":
                self.handle_load(action)
            elif action.command == "inference":
                self.handle_inference(action)
            elif action.command == "token":
                self.handle_token(action)
            elif action.command == "train":
                self.handle_train(action)
            elif action.command == "stop":
                self.handle_stop(action)
                return
            else:
                self.handle_unknown(action)

    def run_action(self, action):
        self._action_queue.put(action)

    def reply(self, action, message):
        message["request_id"] = action.request_id
        self._replier(message)

    def fetch_image(self, image_payload):
        if "base64" in image_payload:
            return io.BytesIO(
                base64.decodebytes(image_payload["base64"].encode("ascii"))
            )
        else:
            raise Exception("Unknown image format")

    def handle_initiate(self, action):
        self.reply(action, {"command": "init", "model_id": action.payload["model_id"]})

    def handle_load(self, action):
        if not self._load_function:
            raise Exception("no load function registered")
        self._load_function()

    def handle_inference(self, action):
        if not self._inference_function:
            raise Exception("no inference function registered")

        image = self.fetch_image(action.payload["image"])
        image = Image.open(image)
        if self._preprocess_function:
            image = self._preprocess_function(image)
        result = self._inference_function(action.payload)
        self.reply(action, {"result": result, "command": "inference_result"})

    def handle_train(self, action):
        if not self._train_function:
            raise Exception("no train function registered")
        self.reply(
            action,
            {
                "command": "request_token",
                "request_id": action.payload["request_id"],
                "dataset_id": action.payload["dataset_id"],
            },
        )

    def handle_token(self, action):
        token = action.payload["token"]
        dataset_id = action.payload["dataset_id"]

        client = darwin.Client.from_token(token)
        dataset = client.get_remote_dataset(dataset_id=dataset_id)
        progress, _count = dataset.pull()
        for _ in progress():
            pass
        # darwin.export_dataset(token, dataset_id)

        result = self._train_function(dataset_id)
        self.reply(action, {"result": result, "command": "train_result"})

    def handle_unknown(self, action):
        self.reply(
            action,
            {
                "error": {
                    "code": "unknown.command",
                    "message": f"Unknown command '{action.command}'.",
                }
            },
        )
