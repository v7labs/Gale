"""
This file is the template for runner classes
"""

# Utils
from abc import abstractmethod, ABCMeta


class SafeSingleton(ABCMeta):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class AbstractRunner(metaclass=SafeSingleton):

    @abstractmethod
    def single_run(self, **kwargs):
        """
        This is the main routine where train(), validate() and test() are called.

        Returns
        -------
        train_value : ndarray[floats] of size (1, `epochs`)
            Accuracy values for train split
        val_value : ndarray[floats] of size (1, `epochs`+1)
            Accuracy values for validation split
        test_value : float
            Accuracy value for test split
        """
        pass

    ####################################################################################################################
    def prepare(self, **kwargs):
        """ Loads and prepares the data, the optimizer and the criterion """
        pass

    def train_routine(self, **kwargs):
        """ Performs the training and validation routines """
        pass

    def test_routine(self, **kwargs):
        """ Load the best model according to the validation score (early stopping) and runs the test routine """
        pass

    ####################################################################################################################
    """
    These methods delegate their function to other classes in this package.
    It is useful because sub-classes can selectively change the logic of certain parts only.
    """
    def _train(self, **kwargs):
        pass

    def _validate(self, **kwargs):
        pass

    def _test(self, **kwargs):
        pass
