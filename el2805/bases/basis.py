import numpy as np
from abc import ABC, abstractmethod


class Basis(ABC):
    """Generic basis function for linear function approximators."""

    @abstractmethod
    def __call__(self, state: np.ndarray):
        """Transform the vector to the given basis.

        :param state: state to transform
        :type state: 1D ndarray
        :return: transform (coefficients under the new basis)
        :rtype: 1D array
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def output_size(self):
        raise NotImplementedError
