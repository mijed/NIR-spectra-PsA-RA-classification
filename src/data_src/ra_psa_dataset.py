import torch
from torch.utils.data import Dataset


class NIRDataset(Dataset):
    def __init__(self, nir_data, labels_data):
        self.data = torch.FloatTensor(nir_data)
        self.labels = torch.LongTensor(labels_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx].unsqueeze(0)
        label = self.labels[idx]
        return data_sample, label

