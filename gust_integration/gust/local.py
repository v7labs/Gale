import argparse
import PIL


class GustLocal:
    def __init__(self, image_path):
        self.image_path = image_path
        self._inference_function = None
        self._load_function = None
        self._preprocess_function = None
        self._train_function = None

    @classmethod
    def from_parser(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("image_path")
        args = parser.parse_args()
        return cls(image_path=args.image_path)

    def load(self, func):
        self._load_function = func

    def inference(self, func):
        self._inference_function = func

    def preprocess(self, func):
        self._preprocess_function = func

    def train(self, func):
        self._train_function = func

    def start(self):
        self._load_function()
        try:
            image = PIL.Image.open(self.image_path)
            if self.preprocess:
                image = self._preprocess_function(image)
            result = self._inference_function(image)
            print(result)
        except FileNotFoundError:
            print(f"Failed to load '{self.image_path}'")
