import numpy as np
import pytest
import scaa
import scipy.sparse as ss
import torch

from fixtures import *

def test_simulate_pois_rank1():
  x, eta = scaa.benchmark.simulate_pois(n=30, p=60, rank=1)
  assert x.shape == (30, 60)
  assert eta.shape == (30, 60)
  assert (x >= 0).all()
  assert (~np.isclose(np.linalg.svd(eta, compute_uv=False, full_matrices=False), 0)).sum() == 1

def test_simulate_pois_rank2():
  x, eta = scaa.benchmark.simulate_pois(n=30, p=60, rank=2)
  assert x.shape == (30, 60)
  assert eta.shape == (30, 60)
  assert (x >= 0).all()
  assert (~np.isclose(np.linalg.svd(eta, compute_uv=False, full_matrices=False), 0)).sum() == 2

def test_simulate_pois_masked():
  x, eta = scaa.benchmark.simulate_pois(n=30, p=60, rank=2, holdout=.25)
  assert np.ma.is_masked(x)

def test_training_score_oracle(simulate):
  x, eta = simulate
  res = scaa.benchmark.training_score_oracle(x, eta)
  assert res <= 0

def test_training_score_nmf(simulate):
  x, eta = simulate
  res = scaa.benchmark.training_score_nmf(x, rank=10)
  assert res <= 0

def test_training_score_grad(simulate):
  x, eta = simulate
  res = scaa.benchmark.training_score_grad(x, rank=1)
  assert res <= 0

def test_training_score_plra(simulate):
  x, eta = simulate
  res = scaa.benchmark.training_score_plra(x, rank=1)
  assert res <= 0

def test_training_score_plra1(simulate):
  x, eta = simulate
  res = scaa.benchmark.training_score_plra1(x, rank=1)
  assert res <= 0

def test_evaluate_training():
  res = scaa.benchmark.evaluate_training(num_trials=1)
  assert res.shape == (1, 6)

def test_loss():
  pred = np.random.normal(size=100)
  true = np.random.normal(size=100)
  res = scaa.benchmark.loss(pred, true)
  assert len(res) == 2

def test_imputation_score_mean(simulate_holdout):
  x, eta = simulate_holdout
  res = scaa.benchmark.imputation_score_mean(x)

def test_imputation_score_nmf(simulate_holdout):
  x, eta = simulate_holdout
  res = scaa.benchmark.imputation_score_nmf(x, rank=10)

def test_imputation_score_plra1(simulate_holdout):
  x, eta = simulate_holdout
  res = scaa.benchmark.imputation_score_plra1(x, rank=1)

def test_imputation_score_plra(simulate_holdout):
  x, eta = simulate_holdout
  res = scaa.benchmark.imputation_score_plra(x, rank=1)

def test_evaluate_pois_imputation():
  res = scaa.benchmark.evaluate_pois_imputation(eta_max=3, num_trials=1)

def test_train_test_split(simulate):
  x, eta = simulate
  train, test = scaa.benchmark.train_test_split(x)
  assert train.shape == x.shape
  assert test.shape == x.shape

def test_train_test_split_sparse_csr(simulate_holdout):
  x, eta = simulate_holdout
  x = ss.csr_matrix(x.filled(0))
  train, test = scaa.benchmark.train_test_split(x)
  assert train.shape == x.shape
  assert test.shape == x.shape
  assert ss.isspmatrix_csr(train)
  assert ss.isspmatrix_csr(test)

def test_train_test_split_sparse_csc(simulate_holdout):
  x, eta = simulate_holdout
  x = ss.csc_matrix(x.filled(0))
  train, test = scaa.benchmark.train_test_split(x)
  assert train.shape == x.shape
  assert test.shape == x.shape
  assert ss.isspmatrix_csc(train)
  assert ss.isspmatrix_csc(test)

def test_generalization_score_oracle(simulate_train_test):
  train, test, eta = simulate_train_test
  scaa.benchmark.generalization_score_oracle(train, test, eta=eta)

def test_generalization_score_plra1(simulate_train_test):
  train, test, eta = simulate_train_test
  scaa.benchmark.generalization_score_plra1(train, test, eta=eta)

def test_generalization_score_nmf(simulate_train_test):
  train, test, eta = simulate_train_test
  scaa.benchmark.generalization_score_nmf(train, test, eta=eta)

def test_generalization_score_grad(simulate_train_test):
  train, test, eta = simulate_train_test
  scaa.benchmark.generalization_score_grad(train, test, eta=eta)

@pytest.mark.skip(reason='CUDA version incompatability')
def test_generalization_score_hpf(simulate_train_test):
  train, test, eta = simulate_train_test
  scaa.benchmark.generalization_score_hpf(train, test, eta=eta)

@pytest.mark.skip(reason='torch bug?')
def test_generalization_score_scvi(simulate_train_test):
  train, test, eta = simulate_train_test
  scaa.benchmark.generalization_score_scvi(train, test, eta=eta)

@pytest.mark.skip(reason='Broken package')
def test_generalization_score_dca(simulate_train_test):
  train, test, eta = simulate_train_test
  scaa.benchmark.generalization_score_dca(train, test, eta=eta)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='torch reports CUDA not available')
def test_generalization_score_zipvae(simulate_train_test):
  train, test, eta = simulate_train_test
  scaa.benchmark.generalization_score_zipvae(train, test, eta=eta)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='torch reports CUDA not available')
def test_generalization_score_zipaae(simulate_train_test):
  train, test, eta = simulate_train_test
  y = (np.random.uniform(size=train.shape[0]) < 0.5).astype(int)
  scaa.benchmark.generalization_score_zipaae(train, test, y=y, eta=eta)

def test_read_ipsc():
  res = scaa.benchmark.read_ipsc()
  assert res.shape == (5597, 9957)

def test_evaluate_generalization():
  res = scaa.benchmark.evaluate_generalization(num_trials=1)
