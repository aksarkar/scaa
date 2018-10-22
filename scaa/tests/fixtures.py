import numpy as np
import pytest

@pytest.fixture
def dims():
  # Data (n, p); latent representation (n, d)
  n = 50
  p = 1000
  d = 20
  stoch_samples = 10
  return n, p, d, stoch_samples

@pytest.fixture
def simulate():
  np.random.seed(0)
  l = np.random.normal(size=(100, 3))
  f = np.random.normal(size=(3, 1000))
  eta = l.dot(f)
  eta *= 5 / eta.max()
  x = np.random.poisson(lam=np.exp(eta))
  return x, eta
