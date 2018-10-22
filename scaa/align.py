import scaa
import torch

from .modules import *

def align(x1, x2, latent_dim, max_epochs=5, **kwargs):
  """Align the datasets x1 and x2

  :param x1: torch.utils.data.TensorDataset
  :param x2: torch.utils.data.TensorDataset
  :param latent_dim: 

  """
  raise NotImplementedError
  n1, p1 = x1.shape
  n2, p2 = x2.shape
  assert p1 == p2

  y = np.zeros(n1 + n2)
  y[n1:] = 1
  encoder = Encoder(p1, latent_dim)
  decoder = ZIP(latent_dim, p1)
  adversary = BinaryDisciminator(latent_dim)

  for epoch in range(max_epochs):
    for batches in zip(torch.utils.data.DataLoader(x1, **kwargs), torch.utils.data.DataLoader(x2, **kwargs)):
      # [2 * batch_size, stoch_samples]
      qz = encoder.forward(torch.cat(*batches))
      # [2 * batch_size, p1]
      px = decoder.forward(qz)
      # [2 * batch_size, 1]
      py = adversary.forward(torch.mean(qz))
