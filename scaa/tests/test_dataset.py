import scaa
import scipy.sparse as ss
import torch
import torch.utils.data

from fixtures import *

@pytest.mark.skipif(not torch.cuda.is_available(), reason='torch reports CUDA not available')
def test_sparse_dataset(simulate_holdout):
  x, eta = simulate_holdout
  x = ss.csr_matrix(x.filled(0))
  training_data = torch.utils.data.DataLoader(scaa.dataset.SparseDataset(x), batch_size=25, shuffle=False)
  batch = next(iter(training_data)) 
  assert batch.shape == (25, x.shape[1])
    
