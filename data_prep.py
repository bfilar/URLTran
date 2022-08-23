import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class URLTranDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        super(URLTranDataset).__init__()
        self.df = pd.read_csv(filepath)
        self.df = self.df.sample(frac=1.0)  # shuffle samples
        self.url_data = self.df.url.values.tolist()
        self.labels = self.df.label.astype(int).values.tolist()
        self.encodings = preprocess(self.url_data, tokenizer)

    def __getitem__(self, idx):
        obs_dict = {k: v[idx] for k, v in self.encodings.items()}
        obs_dict["label"] = self.labels[idx]
        return obs_dict

    def __len__(self):
        return len(self.encodings.input_ids)


def preprocess(url_data, tokenizer):
    inputs = tokenizer(
        url_data, return_tensors="pt", max_length=128, truncation=True, padding=True
    )

    inputs["mlm_labels"] = inputs.input_ids.detach().clone()
    return inputs


def masking_step(inputs):
    rand = torch.rand(inputs.shape)
    # mask array that replicates BERT approach for MLM
    # ensure that [cls], [sep], [mask] remain untouched
    mask_arr = (rand < 0.15) * (inputs != 101) * (inputs != 102) * (inputs != 0)

    selection = [
        torch.flatten(mask_arr[i].nonzero()).tolist() for i in range(inputs.shape[0])
    ]

    for i in range(inputs.shape[0]):
        inputs[i, selection[i]] = 103

    return inputs


def split_data(dataset_path):
    df_final = pd.read_csv(dataset_path)
    X = df_final.url
    y = df_final.label.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    cnames = ["url", "label"]
    train_df = pd.DataFrame(zip(X_train.values, y_train.values), columns=cnames)
    test_df = pd.DataFrame(zip(X_test.values, y_test.values), columns=cnames)
    return train_df, test_df
