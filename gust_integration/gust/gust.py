import os

from local import GustLocal
from remote import GustRemote


class Gust:
    def __init__(self):
        self._inference_function = None
        self._load_function = None
        self._preprocess_function = None
        self._train_function = None

    def load(self, func):
        self._load_function = func

    def inference(self, func):
        self._inference_function = func

    def preprocess(self, func):
        self._preprocess_function = func

    def train(self, func):
        self._train_function = func

    def _remote_mode(self):
        return os.environ.get("GUST_ID") is not None

    def start(self):
        if self._remote_mode():
            handler = GustRemote.from_env()
        else:
            handler = GustLocal.from_parser()

        handler.load(self._load_function)
        handler.inference(self._inference_function)
        handler.preprocess(self._preprocess_function)
        handler.train(self._train_function)

        return handler.start()

########################################################################################################################
if __name__ == "__main__":
    gust = Gust()
    gust.start()
