import numpy as np
import itertools as it
from el2805.bases.basis import Basis


class FourierBasis(Basis):
    def __init__(self, input_size: int, output_size: int, p: int = None, basis: np.ndarray | None = None):
        super().__init__(input_size, output_size)

        if basis is None:
            assert p is not None
            self.eta = np.asarray(list(it.product([np.arange(p + 1) for _ in range(input_size)])))
        else:
            assert basis is not None
            assert basis.shape == (input_size, output_size)
            self.eta = basis

    def __call__(self, state: np.ndarray):
        return np.cos(np.pi * self.eta.T @ state)

    @property
    def output_size(self):
        return self.eta.shape[1]
