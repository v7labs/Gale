# DeepDIVA
from template.runner.base import BaseInference
from template.runner.image_classification.setup import ImageClassificationSetup


class ImageClassificationInference(BaseInference):

    def __init__(self):
        super().__init__()
        self.setup =  ImageClassificationSetup()
