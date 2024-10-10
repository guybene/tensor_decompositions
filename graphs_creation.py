import seaborn as sns
import pandas as pd

from create_metrics import results_names

def batch_1_1():
    res_df = pd.read_csv("results/" + results_names.BATCH_1_1.value)
    sns.relplot(data=res_df, x="decompose_time", y="error", col="rank", row="algo", hue="dim_size")


if __name__ == "__main__":
    batch_1_1()