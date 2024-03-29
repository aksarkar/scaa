import numpy as np
import torch
import scaa

from fixtures import *

def test_kl(dims):
  n, p, d, stoch_samples = dims
  mean = torch.tensor(np.random.normal(size=(n, d)), dtype=torch.float)
  scale = torch.exp(torch.tensor(np.random.normal(size=(n, d)), dtype=torch.float))
  res = scaa.loss.kl_term(mean, scale)
  assert res.shape == (n, d)
  assert np.isfinite(res).all()
  assert (res > 0).all()

def test_pois_llik(dims):
  n, p, d, stoch_samples = dims
  mean = torch.ones([n, p])
  inv_disp = torch.zeros([n, p])
  x = torch.ones([n, p])
  res = scaa.loss.pois_llik(x, mean)
  assert res.shape == (n, p)

def test_zip_llik(dims):
  n, p, d, stoch_samples = dims
  logodds = torch.ones([n, p])
  mean = torch.ones([n, p])
  inv_disp = torch.zeros([n, p])
  x = torch.ones([n, p])
  res = scaa.loss.zip_llik(x, mean, logodds)
  assert res.shape == (n, p)

def test_nb_llik(dims):
  n, p, d, stoch_samples = dims
  mean = torch.ones([n, p])
  inv_disp = torch.zeros([n, p])
  x = torch.ones([n, p])
  res = scaa.loss.nb_llik(x, mean, inv_disp)
  assert res.shape == (n, p)

def test_zinb_llik(dims):
  n, p, d, stoch_samples = dims
  logodds = torch.ones([n, p])
  mean = torch.ones([n, p])
  inv_disp = torch.zeros([n, p])
  x = torch.ones([n, p])
  res = scaa.loss.zinb_llik(x, mean, inv_disp, logodds)
  assert res.shape == (n, p)
