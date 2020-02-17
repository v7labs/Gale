# Utils
import time
from abc import abstractmethod
from tqdm import tqdm

# DeepDIVA
from util.metric_logger import MetricLogger, ScalarValue
from util.TB_writer import TBWriter


class BaseRoutine:

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
            Label for logging purposes. Typically 'train', 'test' or 'valid'.
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
        for batch_idx, (input, target) in pbar:

            # Measure data loading time
            data_time = time.time() - end

            # Moving data to GPU
            input, target = cls.move_to_device(input=input, target=target, **kwargs)

            cls.run_one_mini_batch(input=input,
                                   target=target,
                                   multi_run_label=multi_run_label,
                                   **kwargs)

            # Update the LR according to the scheduler, only during training
            if not 'val' in logging_label and not 'test' in logging_label:
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
                if batch_idx % log_interval == 0 and len(MetricLogger()) > 0:
                    if cls.main_metric() + multi_run_label in MetricLogger():
                        mlogger = MetricLogger()[cls.main_metric()]
                    elif "loss" + multi_run_label in MetricLogger():
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

        if cls.main_metric() in MetricLogger():
            return MetricLogger()[cls.main_metric()].global_avg
        else:
            return 0

    @classmethod
    def move_to_device(cls, input=None, target=None, no_cuda=False, **kwargs):
        """Move the input and the target on the device that shall be used e.g. GPU

        Parameters
        ----------
        input : torch.autograd.Variable
        target : torch.autograd.Variable
           The input and target data for the mini-batch
        no_cuda : boolean
            Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
        Returns
        -------
        input : torch.autograd.Variable
        target : torch.autograd.Variable
           The input and target data for the mini-batch loaded on the GPU
        """
        def move_to_cuda(elem):
            if elem is not None:
                if isinstance(elem, dict):
                    for k, v in elem.items():
                        elem[k] = move_to_cuda(v)
                elif isinstance(elem, (list, tuple)):
                    elem = [move_to_cuda(e) for e in elem]
                else:
                    elem = elem.cuda(non_blocking=True)
            return elem

        if not no_cuda:
            input = move_to_cuda(input)
            target = move_to_cuda(target)
        return input, target

    @classmethod
    def start_of_the_epoch(cls, **kwargs):
        """
        Custom routine to run at the start of the epoch.
        (e.g. for model.train() / model.eval() or to setup meters in MetricLogger(), or setup a confusion matrix)
        NOTE: this is not abstract because this method might not be used by all use cases. It can be left empty.
        """
        pass

    @classmethod
    @abstractmethod
    def run_one_mini_batch(cls, model, criterion, input, target, multi_run_label, **kwargs):
        """Train the model passed as parameter for one mini-batch

        Parameters
        ----------
        model : torch.nn.module
            The network model being used.
        criterion : torch.nn.loss
            The loss function used to compute the loss of the model.
        input : torch.autograd.Variable
            The input data for the mini-batch
        target : torch.autograd.Variable
            The target data (labels) for the mini-batch
        multi_run_label : str
            Label to append in case of multi run
        """
        raise NotImplementedError

    @classmethod
    def end_of_the_epoch(cls, **kwargs):
        """
        Custom routine to run at the end of the epoch.
        (e.g  it can be used to save the confusion matrix to file)
        NOTE: this is not abstract because this method might not be used by all use cases. It can be left empty.
        """
        pass

    @classmethod
    @abstractmethod
    def main_metric(cls) -> str:
        """Return a string with the exact tag for the main metric used to evaluate the model e.g. 'accuracy' """
        raise NotImplementedError
