from enum import Enum, auto

from tensor_algos import (cp_jennrich, cp_als, tensor_train, tucker, tubal_svd)

class decompositions(Enum):
    cp_jennrich = auto()
    cp_als = auto()
    tensor_train = auto()
    tucker = ()
    tubal_svd = ()
