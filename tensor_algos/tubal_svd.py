from typing import List

from numpy import ndarray, fft
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
        u_truncated = u[:, :self._rank]
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
        tensor_transformed = fft.fft(tensor)
        U_s = []
        S_s = []
        V_s = []
        for i in range(tensor.shape[-1]):
            u, s, v = self._rank_r_truncated_svd(matrix=tensor_transformed[:,:,i])
            U_s.append(u)
            S_s.append(s)
            V_s.append(v)
        U_hat = np.stack(U_s, axis=2)
        S_hat = np.stack(S_s, axis=2)
        V_hat = np.stack(V_s, axis=2)
        U = np.real(fft.ifft(U_hat))
        S = np.real(fft.ifft(S_hat))
        V = np.real(fft.ifft(V_hat))
        return U, S, V

    @staticmethod
    def _matrix_m_product(mat_a: ndarray, mat_b: ndarray) -> ndarray:
        assert mat_a.ndim == 3 and mat_b.ndim == 3, "Should have gotten tubal matrices, but didn't get order 3 tensors"
        assert mat_a.shape[2] == mat_b.shape[2], "Both tubal matrices should have the same tube size"
        assert mat_a.shape[1] == mat_b.shape[0], "Should be applicable to matrix multiplication"

        mat_a_hat = fft.fft(mat_a)
        mat_b_hat = fft.fft(mat_b)

        mat_a_hat_transposed = mat_a_hat.transpose((2,0,1))
        mat_b_hat_transposed = mat_b_hat.transpose((2,0,1))

        c_hat_transposed = mat_a_hat_transposed @ mat_b_hat_transposed

        c_hat = c_hat_transposed.transpose((1,2,0))
        c = np.real(fft.ifft(c_hat))
        return c

    @staticmethod
    def compose(decomposed_tensor: List[ndarray]) -> ndarray:
        """
        Assumes the input is a list of tubal matrices U, S, V
        """
        assert len(decomposed_tensor) == 3, "Got wrong input, needs to be a list of size 3"
        U, S, V = decomposed_tensor
        return TubalSVD._matrix_m_product(TubalSVD._matrix_m_product(U, S), V)

    def __str__(self):
        return f"Algo: TubalSVD, Ranks: {self._rank}"

    def get_algo_name(self):
        """
        Get Algo Name
        :return: The name of the algorithm
        """
        return "TUBAL_SVD"


if __name__ == "__main__":
    data = np.random.randn(10, 8, 15)
    algo = TubalSVD(rank=[8])
    decomp_data = algo.decompose(data)
    composed_data = algo.compose(decomp_data)
    print(np.sum(data-composed_data))
