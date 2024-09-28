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

        # Get the right vectors and cast the complex into a float
        U_mat = eig_vecs_AB_inv[:, us].astype(np.float64)
        V_mat = eig_vecs_A_invB[:, vs].astype(np.float64)

        w_systems_vectors = [np.outer(U_mat[:, i], V_mat[:, i]).reshape(-1) for i in range(self._rank)]
        w_system = np.stack(w_systems_vectors).T

        w_s = []
        for i in range(p):
            # Try to solve w_system * W = b system
            curr_w, residuals, _ ,_ = np.linalg.lstsq(w_system, tensor[:,:,i].reshape(-1), rcond=None)
            if residuals > 1e-5:
                print("Problem with w's solving")
            w_s.append(curr_w)
        W_mat = np.stack(w_s)
        return U_mat, V_mat, W_mat





    def __str__(self):
        return f"Algo: CP_Jennrich's, Rank: {self._rank}"

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



from itertools import product
def decompose_tensor_jennrich(rank3tensor):
    # initialize two random variables
    a = np.random.uniform(size=rank3tensor.shape[2])
    b = np.random.uniform(size=rank3tensor.shape[2])

    # T_a and T_b
    Ta = sum(rank3tensor[:, :, i ] *a[i] for i in range(rank3tensor.shape[2]))
    Tb = sum(rank3tensor[:, :, i ] *b[i] for i in range(rank3tensor.shape[2]))

    # eigenvalues of various auxilliary matrices
    a1 = np.matmul(Ta, np.linalg.pinv(Tb))
    a2 = np.transpose(np.matmul(np.linalg.pinv(Ta), Tb))
    eigvals_u, eigvecs_u = np.linalg.eig(a1)
    eigvals_v, eigvecs_v = np.linalg.eig(a2)

    # pair up reciprocal eigenvalues
    # pair up eigenvalues of Ta and Tb
    idx_pairs = []
    tol = 1e-5

    for i, eigval_u in enumerate(eigvals_u):
        for j, eigval_v in enumerate(eigvals_v):
            if abs(eigval_u - 1/ eigval_v) < tol:
                idx_pairs += [(i, j)]
                break

    # solving for third eigenvectors
    nbcomp = len(idx_pairs)
    solved = False
    while not solved:
        try:
            A = np.zeros((nbcomp * rank3tensor.shape[2], nbcomp * rank3tensor.shape[2]))
            B = np.zeros(nbcomp * rank3tensor.shape[2])
            eqidx = 0
            ij_combs = list(tuple(product(range(rank3tensor.shape[0]), range(rank3tensor.shape[1]))))
            for k in range(rank3tensor.shape[2]):
                for ij_comb_idx in np.random.choice(range(len(ij_combs)), size=nbcomp, replace=False):
                    i, j = ij_combs[ij_comb_idx]
                    B[eqidx] = rank3tensor[i, j, k]
                    for ck in range(nbcomp):
                        A[eqidx, ck * rank3tensor.shape[2] + k] = eigvecs_u[i, idx_pairs[ck][0]] * eigvecs_v[j, idx_pairs[ck][1]]
                    eqidx += 1

            sol = np.linalg.solve(A, B)
            solved = True   # exception is not caught at this point
        except np.linalg.LinAlgError:
            solved = False
    eigvecs_w = sol.reshape((nbcomp, rank3tensor.shape[2]), order='F')

    # rearranging eigenvectors
    rearranged_eigvecs_u = np.zeros(shape=eigvecs_u.shape)
    rearranged_eigvecs_v = np.zeros(shape=eigvecs_v.shape)
    for i, (u_idx, v_idx) in enumerate(idx_pairs):
        rearranged_eigvecs_u[:, i] = eigvecs_u[:, u_idx]
        rearranged_eigvecs_v[:, i] = eigvecs_v[:, v_idx]
    rearranged_eigvecs_w = eigvecs_w

    # return values
    return rearranged_eigvecs_u, rearranged_eigvecs_v, rearranged_eigvecs_w


if __name__ == "__main__":
    from tensor_algos.utils import create_random_rank_r_tensor
    from time import time
    algo = CpJennrich(rank=3)
    for i in range(15):
        tensor = create_random_rank_r_tensor(3, (28, 28, 3))
        start = time()
        decomp_data = algo.decompose(tensor)
        comped_tensor = algo.compose(decomp_data)
        print((comped_tensor - tensor).sum())
