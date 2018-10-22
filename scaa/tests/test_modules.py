import numpy as np
import torch
import torch.utils.data
import scaa

from fixtures import *

def test_encoder(dims):
  n, p, d, stoch_samples = dims
  enc = scaa.modules.Encoder(p, d, stoch_samples=stoch_samples)
  x = torch.tensor(np.random.normal(size=(n, p)), dtype=torch.float)
  qz = enc.forward(x)
  assert qz.shape == (stoch_samples, n, d)

def test_decoder(dims):
  n, p, d, stoch_samples = dims
  dec = scaa.modules.ZIP(d, p)
  x = torch.tensor(np.random.normal(size=(n, d)), dtype=torch.float)
  pi, lam = dec.forward(x)
  assert pi.shape == (n, p)
  assert lam.shape == (n, p)

def test_binary_disciminator(dims):
  n, p, d, stoch_samples = dims
  adv = scaa.modules.BinaryDisciminator(d)
  z = torch.tensor(np.random.normal(size=(n, d)), dtype=torch.float)
  py = adv.forward(z)
  assert py.shape == (n, 1)

def test_zipvae(simulate):
  x, eta = simulate
  n, p = x.shape
  latent_dim = 10
  x = torch.utils.data.DataLoader(torch.tensor(x, dtype=torch.float), batch_size=n)
  model = scaa.modules.ZIPVAE(p, latent_dim)
  model.fit(x, lr=1e-2, verbose=True, max_epochs=100)
