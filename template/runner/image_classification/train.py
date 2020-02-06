# DeepDIVA
from evaluation.metrics import accuracy
from template.runner.base.base_routine import BaseRoutine
from util.metric_logger import MetricLogger


class ImageClassificationTrain(BaseRoutine):

    @classmethod
    def start_of_the_epoch(cls, model, **kwargs):
        """See parent method for documentation

        Extra-Parameters
        ----------
        model : torch.nn.module
            The network model being used.
        """
        model.train()

        MetricLogger().add_scalar_meter(tag=cls.main_metric())
        MetricLogger().add_scalar_meter(tag='loss')

    @classmethod
    def run_one_mini_batch(cls, model, criterion, optimizer, input, target, **kwargs):
        """See parent method for documentation

        Extra-Parameters
        ----------
        optimizer : torch.optim
            The optimizer used to perform the weight update.
        """
        # Compute output
        output = model(input)

        # Unpack the target
        target = target['category_id']

        # Compute and record the loss
        loss = criterion(output, target)
        MetricLogger().update(key='loss', value=loss.item(), n=len(input))

        # Compute and record the accuracy
        acc = accuracy(output.data, target.data, topk=(1,))[0]
        MetricLogger().update(key='accuracy', value=acc[0], n=len(input))

        # Reset gradient
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Perform a step by updating the weights
        optimizer.step()

    @classmethod
    def main_metric(cls) -> str:
        """See parent method for documentation"""
        return "accuracy"