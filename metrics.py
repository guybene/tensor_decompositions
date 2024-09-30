from datetime import datetime
import numpy as np
from time import sleep

from tensor_algos.tensor_I_algorithms import TensorAlgo

class AlgoMetrics:
    """
    Strategy class for the metrics
    """

    @staticmethod
    def norm(tensor: np.ndarray):
        """
        Tensor Frobenius norm
        :param tensor: tensor array
        :return: The tensor Frobenius norm
        """
        return np.sqrt(np.sum(np.square(tensor)))

    @staticmethod
    def analyse_single_decomp(algo: TensorAlgo, tensor):
        """
        Returns the metrics:
            decomposition_time, composition_time, frobenius_norm from original, compression_rate
            Note that compression rate will be calculate as compression.size / tensor.size
            Note that the times are calculated in seconds
        :param algo: The algorithm to run
        :param tensor: The tensor to run on
        :return: decomposition_time, composition_time, frobenius_norm from original, compression_rate
        """
        s_time = datetime.now()
        decomp_factors = algo.decompose(tensor)
        decomp_time = s_time - datetime.now()
        composed_tensor = algo.compose(decomp_factors)
        comp_time = s_time - datetime.now()
        error = AlgoMetrics.norm(composed_tensor - tensor)
        compression_rate = 100 * sum([factor.size for factor in decomp_factors]) / tensor.size
        return decomp_time.microseconds, comp_time.microseconds, error, compression_rate


if __name__ == "__main__":
    data = np.random.randn(3,4,6,6,6)
    from tensor_algos.tucker import TuckerHOI
    algo = TuckerHOI(rank=[3])
    print(AlgoMetrics.analyse_single_decomp(algo, data))