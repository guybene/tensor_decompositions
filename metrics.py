from typing import List,  Tuple

from datetime import datetime
import numpy as np


from cifar_model.cifar_eval import CifarModelEvaluator
from tensor_algos.tensor_I_algorithms import TensorAlgo


class AlgoMetrics:
    """
    Utils class for the metrics
    """

    @staticmethod
    def norm(tensor: np.ndarray) -> float:
        """
        Tensor Frobenius norm
        :param tensor: tensor array
        :return: The tensor Frobenius norm
        """
        return np.sqrt(np.sum(np.square(tensor)))

    @staticmethod
    def analyze_single_decomp(algo: TensorAlgo, tensor, noises_variance: List[float]):
        """
        Analyzes a single decomposition algo, i.e. takes an algorithm and a tensor and analyze all the metrics
        with those two.
        Calculates:
            decomposition_time,
            composition_time,
            frobenius_norm from original,
            compression_rate,
            Distance in frobenius norm from original tensor
            Note that compression rate will be calculate as compression.size / tensor.size
            Note that the times are calculated in seconds
        :param algo: The algorithm to run
        :param tensor: The tensor to run on
        :param noises_variance: The list of white noise variances to check for
        :return: decomposition_time, composition_time, frobenius_norm, compression_rate,
         noise degradation in norm from the original, noise degredation in norm from recomposed
        """
        s_time = datetime.now()
        decomp_factors = algo.decompose(tensor)
        decomp_time = s_time - datetime.now()
        composed_tensor = algo.compose(decomp_factors)
        comp_time = s_time - datetime.now()
        error = AlgoMetrics.norm(composed_tensor - tensor)
        compression_rate = 100 * sum([factor.size for factor in decomp_factors]) / tensor.size
        distance_from_original = {}
        distance_from_recomposed = {}
        for noise in noises_variance:
            norm_to_original, norm_to_decomposed= AlgoMetrics.noise_degradation(algo=algo,
                                                                                noise_variance=noise,
                                                                                original_tensor=tensor,
                                                                                recomposed_tensors=composed_tensor)
            distance_from_original[noise] = norm_to_original
            distance_from_recomposed[noise] = norm_to_decomposed
        return decomp_time.microseconds, comp_time.microseconds, error, compression_rate, distance_from_original,\
               distance_from_recomposed


    @staticmethod
    def image_classification_degradation(algo: TensorAlgo, evaluator: CifarModelEvaluator):
        """
        This runs the given algo on a subset of the Cifar dataset
        :param algo: The algo to use in order to decompose the images
        :return: The accuracy of the epoch after usage of the algo
        """
        accuracy = evaluator.eval_model(algo)
        return accuracy

    @staticmethod
    def noise_degradation(algo: TensorAlgo, noise_variance: float, original_tensor: np.ndarray,
                          recomposed_tensors: np.ndarray) -> tuple[float, float]:
        """
        Creates white noise with the given variance, and checks the algorithm noise sensitivity
        :param algo: The algorithm to check
        :param noise_variance: The  variance of the white noise
        :param original_tensor: The original tensor to decompose
        :param recomposed_tensors: The recomposed tensor without the noise
        :return: Frobenious_norm from original tensor, Frobenious_norm from the none noised recomposed tensor
        """
        white_noise = np.random.normal(0, noise_variance, original_tensor.shape)
        noised_tensors = original_tensor + white_noise

        noised_recomposed = algo.compose(algo.decompose(noised_tensors))

        noised_norm = AlgoMetrics.norm(original_tensor - noised_recomposed)
        recomposed_norm = AlgoMetrics.norm(recomposed_tensors - noised_recomposed)

        return noised_norm, recomposed_norm

if __name__ == "__main__":
    data = np.random.randn(3,4,6,6,6)
    from tensor_algos.tucker import TuckerHOI
    algo = TuckerHOI(rank=[3])
    print(AlgoMetrics.analyze_single_decomp(algo, data, [0.1]))