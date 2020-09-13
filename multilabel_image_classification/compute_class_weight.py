from os.path import realpath, dirname, join
import json
import pandas as pd
import torch

def compute_class_freqs(labels):
    """
    https://www.kaggle.com/eugennekhai/chest-x-ray-pytorch-lightning-densnet121
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
    """
    N = len(labels)
    pow_weight = torch.sum(labels, axis=0) / N
    return pow_weight

df_path = join(dirname(realpath(__file__)), "dataframe.csv")
df = pd.read_csv(df_path)
column = df["label"].tolist()
labels = list(map(lambda row : json.loads(row), column))
labels = torch.tensor(labels, dtype=torch.float)
pos_weight = compute_class_freqs(labels)
