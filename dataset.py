import torch
from torch.utils.data import Dataset


class ForexDataset(Dataset):
    def __init__(self, X, y):
        """
        X: numpy array of shape (num_samples, seq_len, num_features)
        y: numpy array of shape (num_samples,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]