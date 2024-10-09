from typing import List
from numpy import ndarray
import numpy as np


from tensor_algos.tensor_I_algorithms import TensorAlgo


class CpJennrich(TensorAlgo):
    """
    Uses Jennrich's algorithm in order to get the CP decomposition of an order 3 tensor
    """


    def __init__(self, rank, pairs_tol = 1e-5):
        self._rank = rank
        self._pairs_tol = pairs_tol


    def decompose(self, tensor: ndarray) -> List[ndarray]:
        """
        Note that a tensor must be of a certain form
        :param tensor: Order 3 tensor
        :return: The factors needed
        """
        assert len(tensor.shape) == 3, "Tensor must be of order 3"
        m, n, p = tensor.shape
        a, b = np.random.uniform(low=-1, high=1,size=p), np.random.uniform(low=-1, high=1,size=p)
        a, b = a/np.linalg.norm(a),  b/np.linalg.norm(b)
        A = np.einsum("ijk, k -> ij", tensor, a)
        B = np.einsum("ijk, k -> ij", tensor, b)

        AB_inv = A @ np.linalg.pinv(B)
        A_invB = (np.linalg.pinv(A) @ B).T

        eig_vals_AB_inv, eig_vecs_AB_inv = np.linalg.eig(AB_inv)
        eig_vals_A_invB, eig_vecs_A_invB = np.linalg.eig(A_invB)

        us = []
        vs = []
        for i, val1 in enumerate(eig_vals_AB_inv):
            for j, val2 in enumerate(eig_vals_A_invB):
                if np.abs(1 - val1 * val2) <= self._pairs_tol:
                    us.append(i)
                    vs.append(j)
        assert len(us) > 0, "No vectors found"

        factors_num = min(len(us), self._rank)

        # Get the right vectors and cast the complex into a float
        U_mat = eig_vecs_AB_inv[:, us[:factors_num]].astype(np.float64)
        V_mat = eig_vecs_A_invB[:, vs[:factors_num]].astype(np.float64)

        w_systems_vectors = [np.outer(U_mat[:, i], V_mat[:, i]).reshape(-1) for i in range(factors_num)]
        w_system = np.stack(w_systems_vectors).T

        w_s = []
        for i in range(p):
            # Try to solve w_system * W = b system
            curr_w, residuals, _ ,_ = np.linalg.lstsq(w_system, tensor[:,:,i].reshape(-1), rcond=None)
            w_s.append(curr_w)
        W_mat = np.stack(w_s)
        return U_mat, V_mat, W_mat


    def __str__(self):
        return f"Algo: CP_Jennrich's, Rank: {self._rank}"

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

    def get_algo_name(self):
        """
        Get Algo Name
        :return: The name of the algorithm
        """
        return "CP_JENNRICH"


if __name__ == "__main__":
    from tensor_algos.utils import create_random_rank_r_tensor
    from time import time
    algo = CpJennrich(rank=10)
    tensor = create_random_rank_r_tensor(13, (28, 28, 18))
    print("Starting")
    start = time()
    decomp_data = algo.decompose(tensor)
    comped_tensor = algo.compose(decomp_data)
    print((tensor-comped_tensor).sum())