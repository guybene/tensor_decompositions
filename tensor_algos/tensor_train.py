from typing import List

from numpy import ndarray
import numpy as np

from tensor_algos.tensor_I_algorithms import TensorAlgo
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor


class TensorTrain(TensorAlgo):


    def __init__(self, rank: List[int]):
        """
        Tensor train object initializer
        :param rank: A list of rank of the form:
            If the list is a single int, takes that int as the rank for all factors.
            Else, is sent as a rank for each factor
        """
        self._rank = rank if len(rank) > 1 else rank[0]
        self._algo = tensor_train


    def decompose(self, tensor: ndarray) -> List[ndarray]:
        train = self._algo(tensor, rank=self._rank)
        return train

    def __str__(self):
        return f"Algo: Tensor Train, Ranks: {self._rank}"

    @staticmethod
    def compose(decomposed_tensor) -> ndarray:
        """
        Assumes the input to be a tensor train object
        """
        return tt_to_tensor(decomposed_tensor)


if __name__ == "__main__":
    data = np.random.randn(3, 4, 5, 6, 7)
    algo = TensorTrain(rank=[50])
    decomp_data = algo.decompose(data)
    composed_data = algo.compose(decomp_data)
    print(np.sum(data-composed_data))