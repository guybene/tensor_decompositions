from abc import ABC, abstractmethod
from numpy import ndarray
from typing import Tuple, List

class TensorAlgo(ABC):

    @abstractmethod
    def get_algo_name(self):
        raise Exception("Not Implemented")

    @abstractmethod
    def decompose(self, tensor: ndarray) -> List[ndarray]:
        raise Exception("Not Implemented")

    @staticmethod
    @abstractmethod
    def compose(decomposed_tensor: List[ndarray]) -> ndarray:
        raise Exception("Not Implemented")

    @abstractmethod
    def __str__(self):
        raise Exception("Not Implemented")



