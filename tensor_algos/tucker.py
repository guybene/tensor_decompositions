from typing import List

from numpy import ndarray
import numpy as np

from tensor_algos.tensor_I_algorithms import TensorAlgo
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import TuckerTensor


class TuckerHOI(TensorAlgo):


    def __init__(self, rank: List[int], n_iter_max = None, tol= None):
        """
        Tucker object initializer
        :param rank: A list of rank of the form:
            If the list is a single int, takes that int as the rank for all factors.
            Else, is sent as a rank for each factor
        :param n_iter_max: Num of iterations
        :param tol: The tolerance for the algorithm when it comes to errors
        """
        self._rank = rank if len(rank) > 1 else rank[0]
        self._n_iter_max = n_iter_max if n_iter_max is not None else 100
        self._tol = tol if tol is not None else 1e-8
        self._algo = tucker


    def decompose(self, tensor: ndarray) -> List[ndarray]:
        core, factors = self._algo(tensor, rank=self._rank, n_iter_max=self._n_iter_max, tol=self._tol)
        return [core] + factors

    def __str__(self):
        return f"Algo: Tucker, Ranks: {self._rank}, n_iter: {self._n_iter_max}"

    @staticmethod
    def compose(decomposed_tensor: List[ndarray]) -> ndarray:
        """
        Assumes the input is a list of [core_tensor, U_1,..,U_N]
        """
        core = decomposed_tensor[0]
        factors = decomposed_tensor[1:]
        tucker_obj = TuckerTensor((core, factors))
        return tucker_obj.to_tensor()




if __name__ == "__main__":
    data = np.random.randn(3, 4, 5, 6, 7)
    algo = TuckerHOI(rank=[20])
    decomp_data = algo.decompose(data)
    composed_data = algo.compose(decomp_data)
    print(np.sum(data-composed_data))