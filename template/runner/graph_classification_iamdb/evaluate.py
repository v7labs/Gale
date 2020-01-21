import torch

# Gale
import numpy as np
from evaluation.metrics import accuracy
from util.metric_logger import MetricLogger
from util.TB_writer import TBWriter
from .train import GraphClassificationTrain


class GraphClassificationEvaluate(GraphClassificationTrain):
    @classmethod
    def start_of_the_epoch(cls, model, num_classes, data_loader, **kwargs):
        """See parent method for documentation

        Extra-Parameters
        ----------
        model : torch.nn.module
            The network model being used.
        num_classes : int
            Number of classes in the dataset
        """
        model.eval()
        MetricLogger().add_scalar_meter(tag=cls.main_metric())
        MetricLogger().add_scalar_meter(tag='loss')
        MetricLogger().add_confusion_matrix_meter(tag='confusion_matrix', num_classes=num_classes)
        MetricLogger().add_classification_results_meter(tag='classification_results', file_list=data_loader.dataset.config['file_names'])


    @classmethod
    def run_one_mini_batch(cls, model, criterion, input, target, **kwargs):
        """See parent method for documentation"""
        # Compute output
        output = model(input)

        # Compute and record the loss
        loss = criterion(output, target)
        MetricLogger().update(key='loss', value=loss.item(), n=len(input))

        # Compute and record the accuracy
        acc = accuracy(output.data, target.data, topk=(1,))[0]
        MetricLogger().update(key='accuracy', value=acc[0], n=len(input))

        # Update the confusion matrix
        MetricLogger().update(key='confusion_matrix', p=np.argmax(output.data.cpu().numpy(), axis=1), t=target.cpu().numpy())
        # Update the classification results
        MetricLogger().update(key='classification_results', p=np.argmax(output.data.cpu().numpy(), axis=1), t=target.cpu().numpy(),
                            f_ind=input.file_name_ind.cpu().numpy())

    @classmethod
    def end_of_the_epoch(cls, data_loader, epoch, logging_label, current_log_folder=None, **kwargs):
        """See parent method for documentation

        Extra-Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The dataloader of the current set.
        epoch : int
            Number of the epoch (for logging purposes).
        logging_label : string
            Label for logging purposes. Typically 'train', 'test' or 'valid'.
            It's prepended to the logging output path and messages.
        """
        classes = data_loader.dataset.config['classes']

        # Make and log to TB the confusion matrix
        cm = MetricLogger()['confusion_matrix'].make_heatmap(classes)
        TBWriter().save_image(tag=logging_label + '/confusion_matrix', image=cm, global_step=epoch)

        # Generate a classification report for each epoch
        cr = MetricLogger()['confusion_matrix'].get_classification_report(classes)
        TBWriter().add_text(tag='Classification Report for epoch {}\n'.format(epoch),
                            text_string='\n' + cr,
                            global_step=epoch)

        if current_log_folder:
            # save the clasification output as a csv
            MetricLogger()['classification_results'].save_csv(output_folder=current_log_folder)
