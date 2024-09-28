from typing import List

from numpy import ndarray
import numpy as np

from tensor_algos.tensor_I_algorithms import TensorAlgo
from tensorly.decomposition import parafac


class CpAls(TensorAlgo):


    def __init__(self, rank, n_iter_max = None, tol= None):
        self._rank = rank
        self._n_iter_max = n_iter_max if n_iter_max is not None else 100
        self._tol = tol if tol is not None else 1e-8
        self._algo = parafac


    def decompose(self, tensor: ndarray) -> List[ndarray]:
        weights, factors = self._algo(tensor, rank=self._rank, n_iter_max=self._n_iter_max, tol=self._tol)
        return factors

    def __str__(self):
        return f"Algo: CP_ALS, Rank: {self._rank}, n_iter: {self._n_iter_max}"

    #TODO: Perhaps revert from static method
    @staticmethod
    def compose(decomposed_tensor: List[ndarray]) -> ndarray:
        """
        Assumes the input is a list on nd.array
        :param decomposed_tensor:
        :return: A composed single tensor
        """
        cp_rank = decomposed_tensor[0].shape[1]
        tensor_shape = [factor.shape[0] for factor in decomposed_tensor]
        composed_data = np.zeros(tensor_shape)

        einsum_input = ','.join([chr(97 + i) for i in range(len(tensor_shape))])
        einsum_output = ''.join([chr(97 + i) for i in range(len(tensor_shape))])
        for i in range(cp_rank):
            curr_factors = [factor.T[i] for factor in decomposed_tensor]
            composed_data += np.einsum(f"{einsum_input}->{einsum_output}", *curr_factors)
        return composed_data

if __name__ == "__main__":
    data = np.random.randn(3, 4, 5, 6, 7)
    factors = parafac(data, rank=100).factors
    comp_data = CpAls.compose(factors)
    print(np.sum(data-comp_data))