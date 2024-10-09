from typing import Dict
from enum import Enum
import pickle

from tqdm import tqdm

from metrics import AlgoMetrics
from tensor_algos.utils import create_random_rank_r_tensor


class results_names(Enum):
    BATCH_1_1 = "jenrich_als_compare"
    BATCH_1_2 = "tucker_tubal_tt_compare"
    BATCH_1_3 = "t_svd_tt_compare"
    BATCH_1_4 = "full_basic_compare"


def generate_data():
    print("Starting Data Generation")
    SAMPLES_SIZE_PER_SHAPE_PER_RANK = 100
    DIM_SIZE = 25
    RANKS = [5, 10, 20]
    N_DIMS = [3, 4, 5]
    GENERATED_DATA_PATH = "results_file/generated_data.pkl"

    generated_data = {}
    for rank in RANKS:
        for n_dim in N_DIMS:
            print("Creating: ", rank, n_dim)
            current_data_generated = []
            for i in tqdm(range(SAMPLES_SIZE_PER_SHAPE_PER_RANK)):
                current_data_generated.append(create_random_rank_r_tensor(rank, [DIM_SIZE] * n_dim))
            generated_data[(rank, n_dim)] = current_data_generated

    with open(GENERATED_DATA_PATH, "w") as f:
        pickle.dump(generated_data, f)

    print("Finished Data Generation")





def main():
    # Basic Comparison
    generate_data()


if __name__ == "__main__":
    main()