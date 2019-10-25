# DeepDIVA
from template.runner.base import BaseInference
from template.runner.image_classification.setup import ImageClassificationSetup


class ImageClassificationInference(BaseInference):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup = ImageClassificationSetup()
