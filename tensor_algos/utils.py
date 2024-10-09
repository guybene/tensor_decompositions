import numpy as np


def create_random_rank_r_tensor(rank, shape):
    """
    Creates a tensor of the given rank in the given shape. By making sure all matrices are of full
    rank so we can assert uniqueness
    :param rank: The tensor rank
    :param shape: The Tensor shape
    :return: A tensor of the relevant rank and shape
    """
    # Create matrices of full rank
    matrices = []
    assert rank <= min(shape), f"Cant create a matrix of rank: {rank} with dim: {min(shape)}"
    for dim in shape:
        curr_mat = np.random.randn(dim, rank).astype(np.float32)
        while np.linalg.matrix_rank(curr_mat) != rank:
            curr_mat = np.random.randn(dim, rank).astype(np.float32)
        matrices.append(curr_mat)

    # Builds the required tensor out of <rank> outer products
    tensor = np.zeros(shape, dtype=np.float32)
    einsum_input = ','.join([chr(97 + i) for i in range(len(shape))])
    einsum_output = ''.join([chr(97 + i) for i in range(len(shape))])
    for i in range(rank):
        curr_factors = [factor.T[i] for factor in matrices]
        tensor += np.einsum(f"{einsum_input}->{einsum_output}", *curr_factors)
    return tensor
