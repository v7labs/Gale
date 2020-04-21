# Utils
import time
from tqdm import tqdm
import torch
import os

# Gale
from evaluation.metrics import accuracy
from template.runner.base.base_routine import BaseRoutine
from util.metric_logger import MetricLogger, ScalarValue
from util.TB_writer import TBWriter


class GraphClassificationTrain(BaseRoutine):
    @classmethod
    def run(cls, data_loader, epoch, log_interval, logging_label, batch_lr_schedulers, run=None,
            **kwargs):
        """
        Training routine

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The dataloader of the current set.
        epoch : int
            Number of the epoch (for logging purposes).
        log_interval : int
            Interval limiting the logging of mini-batches.
        logging_label : string
            Label for logging purposes. Typically 'train', 'test' or 'val'.
            It's prepended to the logging output path and messages.
        run : int
            Number of run, used in multi-run context to discriminate the different runs
        batch_lr_schedulers : list(torch.optim.lr_scheduler)
            List of lr schedulers to call step() on after every batch

        Returns
        ----------
        Main metric : float
            Main metric of the model on the evaluated split
        """
        # 'run' is injected in kwargs at runtime in RunMe.py IFF it is a multi-run event
        multi_run_label = f"_{run}" if run is not None else ""

        # Instantiate the counter
        MetricLogger().reset(postfix=multi_run_label)

        # Custom routine to run at the start of the epoch
        cls.start_of_the_epoch(data_loader=data_loader,
                               epoch=epoch,
                               logging_label=logging_label,
                               multi_run_label=multi_run_label,
                               **kwargs)

        # Iterate over whole training set
        end = time.time()
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=130, leave=False)
        for batch_idx, data in pbar:
            input = data
            target = data.y

            # Measure data loading time
            data_time = time.time() - end

            # Moving data to GPU
            input, target = cls.move_to_device(input=input, target=target, **kwargs)

            cls.run_one_mini_batch(input=input,
                                   target=target,
                                   multi_run_label=multi_run_label,
                                   **kwargs)

            # Update the LR according to the scheduler
            for lr_scheduler in batch_lr_schedulers:
                lr_scheduler.step()

            # Add metrics to Tensorboard for the last mini-batch value
            for tag, meter in MetricLogger():
                if isinstance(meter, ScalarValue):
                    TBWriter().add_scalar(tag=logging_label + '/mb_' + tag,
                                          scalar_value=meter.value,
                                          global_step=epoch * len(data_loader) + batch_idx)

            # Measure elapsed time for a mini-batch
            batch_time = time.time() - end
            end = time.time()

            # Log to console
            if batch_idx % log_interval == 0 and len(MetricLogger()) > 0:
                if cls.main_metric()+multi_run_label in MetricLogger():
                    mlogger = MetricLogger()[cls.main_metric()]
                elif "loss"+multi_run_label in MetricLogger():
                    mlogger = MetricLogger()["loss"]
                else:
                    raise AttributeError
                pbar.set_description(f'{logging_label} epoch [{epoch}][{batch_idx}/{len(data_loader)}]')
                pbar.set_postfix(Metric=f'{mlogger.global_avg:.3f}',
                                 Time=f'{batch_time:.3f}',
                                 Data=f'{data_time:.3f}')

        # Custom routine to run at the end of the epoch
        cls.end_of_the_epoch(data_loader=data_loader,
                             epoch=epoch,
                             logging_label=logging_label,
                             multi_run_label=multi_run_label,
                             **kwargs)

        # Add metrics to Tensorboard for the full-epoch value
        for tag, meter in MetricLogger():
            if isinstance(meter, ScalarValue):
                TBWriter().add_scalar(tag=logging_label + '/' + tag, scalar_value=meter.global_avg, global_step=epoch)

        if cls.main_metric()+multi_run_label in MetricLogger():
            return MetricLogger()[cls.main_metric()].global_avg
        else:
            return 0

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
        output = model(input, target_size=target.shape[0])

        # Compute and record the loss
        loss = criterion(output, target)
        MetricLogger().update(key='loss', value=loss.item(), n=len(input))

        # Compute and record the accuracy
        acc = accuracy(output.data, target.data, topk=(1,))[0]
        MetricLogger().update(key='accuracy', value=acc[0], n=len(input))
        # TODO check if n is correct

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

    @classmethod
    def move_to_device(cls, input=None, target=None, no_cuda=False, **kwargs):
        """Move the input and the target on the device that shall be used e.g. GPU

        Parameters
        ----------
        input : torch
        target : torch
           The input and target data for the mini-batch
        no_cuda : boolean
            Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
        Returns
        -------
        input : torch
        target : torch
           The input and target data for the mini-batch loaded on the GPU
        """
        def move_to_cuda(elem):
            if elem is not None:
                if isinstance(elem, dict):
                    for k, v in elem.items():
                        elem[k] = move_to_cuda(v)
                else:
                    elem = elem.to(torch.device('cuda:{}'.format(os.environ['CUDA_VISIBLE_DEVICES'])))
            return elem

        if not no_cuda:
            input = move_to_cuda(input)
            target = move_to_cuda(target)
        return input, target

