from typing import List

from numpy import ndarray
import numpy as np

from tensor_algos.tensor_I_algorithms import TensorAlgo


class TubalSVD(TensorAlgo):


    def __init__(self, rank: List[int]):
        """
        Tubal SVD object initializer, uses the DFT matrix for M
        :param rank: A list of rank of the form:
            If the list is a single int, takes that int as the rank for all faces.
            Else, is sent as a rank for each face
        """
        self._rank = rank if len(rank) > 1 else rank[0]

    def _rank_r_truncated_svd(self, matrix: ndarray):
        """
        Takes a matrix and returns its truncated SVD decomposition
        :param matrix: A ndarray with 2 dims
        :param rank: The rank of the truncation
        :return: U, S, V where matrix = U * S * V.T
        """
        u, s, v_t = np.linalg.svd(matrix, full_matrices=False)
        u_truncated = u[:,:self._rank]
        s_truncated = np.diag(s[:self._rank])
        v_t_truncated = v_t[: self._rank, :]
        return u_truncated, s_truncated, v_t_truncated




    def decompose(self, tensor: ndarray) -> List[ndarray]:
        """
        Decomposes into the Tubal SVD
        :param tensor: A tensor of order 3
        :return:
        """
        assert tensor.ndim == 3, "Current Implementation only for order 3 dims"
        tensor_transformed = np.apply_along_axis(np.fft.rfft, axis=2, arr=tensor)
        U_s = []
        S_s = []
        V_s = []
        for i in range(tensor.shape[-1]):
            u, s, v = self._rank_r_truncated_svd(matrix=tensor_transformed[:,:,i])
            U_s.append(u)
            S_s.append(s)
            V_s.append(v)
        return [tensor]



    @staticmethod
    def compose(decomposed_tensor: List[ndarray]) -> ndarray:
        """
        Assumes the input is a list of [core_tensor, U_1,..,U_N]
        """
        pass

    def __str__(self):
        return f"Algo: TubalSVD, Ranks: {self._rank}"


if __name__ == "__main__":
    data = np.random.randn(3, 4, 5)
    algo = TubalSVD(rank=[3])
    decomp_data = algo.decompose(data)
    # composed_data = algo.compose(decomp_data)
    # print(np.sum(data-composed_data))
