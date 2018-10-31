import torch.utils.data

class SparseDataset(torch.utils.data.Dataset):
  def __init__(self, x):
    self.x = x

  def __getitem__(self, index):
    return torch.tensor(self.x[index].A.ravel(), dtype=torch.float)

  def __len__(self):
    return self.x.shape[0]
