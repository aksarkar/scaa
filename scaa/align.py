import scipy.sparse as ss
import scaa
import torch

def align(counts, labels, latent_dim, batch_size=100, max_epochs=2, n_iters=10, verbose=False):
  """Align the datasets x1 and x2

  :param x1: Matrix of counts (ndarray-like)
  :param x2: Matrix of counts (ndarray-like)
  :param latent_dim: Latent dimension of the embedding

  """
  training_data = torch.utils.data.DataLoader(
    scaa.dataset.SparseDataset(counts),
    batch_size=batch_size,
    num_workers=3,
    pin_memory=True,
    shuffle=True,
  )
  batch_data = torch.utils.data.DataLoader(
    torch.tensor(labels, dtype=torch.long),
    batch_size=batch_size,
    num_workers=3,
    pin_memory=True,
    shuffle=True,
  )
  eval_data = torch.utils.data.DataLoader(
    scaa.dataset.SparseDataset(counts),
    batch_size=batch_size,
    num_workers=3,
    pin_memory=True,
    shuffle=False,
  )
  # Fit the model
  with torch.cuda.device(0):
    model = scaa.modules.ZIPAAE(input_dim=counts.shape[1], latent_dim=latent_dim, num_classes=2).fit(x=training_data, y=batch_data, max_epochs=max_epochs, n_iters=n_iters, verbose=verbose)
  return
  # Recover the embedding
  with torch.set_grad_enabled(False):
    z = torch.cat([model.vae.forward(batch)[0] for batch in eval_data]).cpu().numpy()
  return z
