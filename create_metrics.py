from enum import Enum
import pickle
import os

import pandas as pd
from tqdm import tqdm

from metrics_utils import AlgoMetrics
from tensor_algos.cp_jennrich import CpJennrich
from tensor_algos.cp_als import CpAls
from tensor_algos.tensor_train import TensorTrain
from tensor_algos.tubal_svd import TubalSVD
from tensor_algos.tucker import TuckerHOI

from tensor_algos.utils import create_random_rank_r_tensor


class results_names(Enum):
    BATCH_1_1 = "jenrich_als_compare.csv"
    BATCH_1_2 = "tucker_tubal_tt_compare.csv"
    BATCH_1_3 = "t_svd_tt_compare.csv"
    BATCH_1_4 = "full_basic_compare.csv"

GENERATED_DATA_PATH_PREFIX = "generated_data/generated_data"
RANKS = [5, 10, 20]

def generate_data():
    print("Starting Data Generation")
    SAMPLES_SIZE_PER_SHAPE_PER_RANK = 100
    DIM_SIZE = [[5, 10, 20, 50], [20], [20]]
    N_DIMS = [3, 4, 5]

    for rank in RANKS:
        data_per_rank = {}
        for dims, n_dim in zip(DIM_SIZE, N_DIMS):
            current_data_generated = []
            for size in dims:
                print("Creating: ", rank, n_dim, size)
                for _ in tqdm(range(SAMPLES_SIZE_PER_SHAPE_PER_RANK)):
                    if rank <= size:
                        current_data_generated.append(create_random_rank_r_tensor(rank, [size] * n_dim))
            data_per_rank[n_dim] = current_data_generated
        with open(GENERATED_DATA_PATH_PREFIX + f"_rank_{rank}.pkl", "wb") as f:
            pickle.dump(data_per_rank, f)
        print(f"Pickled generated dataset of rank {rank}")


    print("Finished Data Generation")

def get_dataset(rank):
    """
    Returns the relevant dataset with the specified rank of tensors
    """
    folder_name = GENERATED_DATA_PATH_PREFIX.split("/")[0]
    datasets_file_names = os.listdir(folder_name)
    for dataset_name in datasets_file_names:
        if str(rank) in dataset_name:
            with open(os.path.join(folder_name, dataset_name), "rb") as f:
                data = pickle.load(f)
            return data
    raise Exception(f"Dataset with rank: {rank}, not found")

def batch_1_1_metrics():
    """
    Compares between the jennrich algorithm and the cp_als agorithm
    """
    guessing_ranks = [0.7, 0.85, 0.95, 1, 1.1]
    iter_n_choices = [20, 50, 70]
    stage_1_res_dict = {"algo": [], "rank": [], "guess_rank": [],
                        "n_dim": [], "dim_size": [], "n_iter": [],
                        "compose_time": [], "decompose_time": [],
                        "error": [], "compression_rate": []}
    for rank in RANKS:
        dataset = get_dataset(rank)
        for n_dim, data_per_dim in dataset.items():
            if n_dim != 3:
                continue
            for i, tensor in enumerate(data_per_dim):
                print("Rank: ", rank, "N_dim: ", n_dim, "Tensor: ", i)
                for guess in guessing_ranks:
                    curr_rank_guess = int(rank * guess)
                    curr_jennrich = CpJennrich(rank=curr_rank_guess)
                    curr_als = CpAls(rank=curr_rank_guess)
                    for algo in [curr_jennrich, curr_als]:
                        decomp_time, comp_time, error, compression_rate, _, _ = \
                            AlgoMetrics.analyze_single_decomp(algo, tensor, [])
                        stage_1_res_dict["algo"].append(algo.get_algo_name())
                        stage_1_res_dict["rank"].append(rank)
                        stage_1_res_dict["guess_rank"].append(curr_rank_guess)
                        stage_1_res_dict["n_dim"].append(tensor.ndim)
                        stage_1_res_dict["dim_size"].append(tensor.shape[0])
                        stage_1_res_dict["n_iter"].append(100)
                        stage_1_res_dict["decompose_time"].append(decomp_time)
                        stage_1_res_dict["compose_time"].append(comp_time)
                        stage_1_res_dict["error"].append(error)
                        stage_1_res_dict["compression_rate"].append(compression_rate)
                for iter_n in iter_n_choices:
                    algo = CpAls(rank=rank, n_iter_max=iter_n)
                    decomp_time, comp_time, error, compression_rate, _, _ = \
                        AlgoMetrics.analyze_single_decomp(algo, tensor, [])
                    stage_1_res_dict["algo"].append(algo.get_algo_name())
                    stage_1_res_dict["rank"].append(rank)
                    stage_1_res_dict["guess_rank"].append(rank)
                    stage_1_res_dict["n_dim"].append(tensor.ndim)
                    stage_1_res_dict["dim_size"].append(tensor.shape[0])
                    stage_1_res_dict["decompose_time"].append(decomp_time)
                    stage_1_res_dict["compose_time"].append(comp_time)
                    stage_1_res_dict["error"].append(error)
                    stage_1_res_dict["compression_rate"].append(compression_rate)
    pd.DataFrame(stage_1_res_dict).to_csv(f"results/{results_names.BATCH_1_1.value}")

def batch_1_2_metrics():
    pass

def basic_comparison_metrics():
    """
    Basic comparison metrics
    """
    # batch_1_1_metrics()
    batch_1_2_metrics()



def main():
    # generate_data()
    basic_comparison_metrics()


if __name__ == "__main__":
    main()