from torch.utils.data import Dataset

class IndexFilteredDataset(Dataset):
    def __init__(self, base_ds, index_list):
        self.base = base_ds
        self.index_list = index_list

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, i):
        return self.base[self.index_list[i]]