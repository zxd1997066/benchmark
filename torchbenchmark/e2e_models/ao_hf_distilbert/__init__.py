from torchbenchmark.util.e2emodel import E2EBenchmarkModel
from torchbenchmark.tasks import NLP


class Model(E2EBenchmarkModel):
    task = NLP.LANGUAGE_MODELING
    DEFAULT_TRAIN_BSIZE: int = 32
    DEFAULT_EVAL_BSIZE: int = 1

    def __init__(self, test, batch_size=None, extra_args=[]):
        super().__init__(test=test, batch_size=batch_size, extra_args=extra_args)
        # TODO: currently only support 1 GPU device
        model_name = "distilbert-base-uncased-distilled-squad"
        self.device = "cuda"

    def train(self):
        pass

    def eval(self):
        pass
