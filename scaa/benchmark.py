import itertools
import numpy as np
import pandas as pd
import scipy.sparse as ss
import scipy.stats as st
import torch

def simulate_pois(n, p, rank, eta_max=None, holdout=None, seed=0):
  np.random.seed(seed)
  l = np.random.normal(size=(n, rank))
  f = np.random.normal(size=(rank, p))
  eta = l.dot(f)
  if eta_max is not None:
    # Scale the maximum value
    eta *= eta_max / eta.max()
  x = np.random.poisson(lam=np.exp(eta))
  if holdout is not None:
    mask = np.random.uniform(size=(n, p)) < holdout
    x = np.ma.masked_array(x, mask=mask)
  return x, eta

def training_score_oracle(x, eta):
  return st.poisson(mu=np.exp(eta)).logpmf(x).sum()

def training_score_nmf(x, rank):
  from wlra.nmf import nmf
  return st.poisson(mu=nmf(x, rank)).logpmf(x).sum()

def training_score_grad(x, rank):
  import wlra.grad
  m = (wlra.grad.PoissonFA(n_samples=x.shape[0], n_features=x.shape[1], n_components=rank)
       .fit(x, atol=1e-3, max_epochs=10000))
  return st.poisson(mu=np.exp(m.L.dot(m.F))).logpmf(x).sum()

def training_score_plra(x, rank):
  import wlra
  return st.poisson(mu=np.exp(wlra.plra(x, rank=rank, max_outer_iters=100, check_converged=True))).logpmf(x).sum()

def training_score_plra1(x, rank):
  import wlra
  return st.poisson(mu=np.exp(wlra.plra(x, rank=rank, max_outer_iters=1))).logpmf(x).sum()

def evaluate_training(rank=3, eta_max=2, num_trials=10):
  result = []
  for trial in range(num_trials):
    x, eta = simulate_pois(n=200, p=300, rank=rank, eta_max=eta_max, seed=trial)
    result.append([
      trial,
      training_score_oracle(x, eta),
      training_score_nmf(x, rank),
      training_score_grad(x, rank),
      training_score_plra(x, rank),
      training_score_plra1(x, rank)
    ])
  result = pd.DataFrame(result)
  result.columns = ['trial', 'Oracle', 'NMF', 'Grad', 'PLRA', 'PLRA1']
  return result

def rmse(pred, true):
  return np.sqrt(np.square(pred - true).mean())

def pois_loss(pred, true):
  return (pred - true * np.log(pred + 1e-8)).mean()

losses = [rmse, pois_loss]

def loss(pred, true):
  return [f(pred, true) for f in losses]

def imputation_score_mean(x):
  """Mean-impute the data"""
  return loss(x.mean(), x.data[x.mask])

def imputation_score_nmf(x, rank):
  try:
    from wlra.nmf import nmf
    res = nmf(x, rank, atol=1e-3)
    return loss(res[x.mask], x.data[x.mask])
  except RuntimeError:
    return [np.nan for f in losses]

def imputation_score_plra1(x, rank):
  try:
    import wlra
    res = np.exp(wlra.plra(x, rank=rank, max_outer_iters=1))
    return loss(res[x.mask], x.data[x.mask])
  except RuntimeError:
    return [np.nan for f in losses]

def imputation_score_plra(x, rank):
  try:
    import wlra
    res = np.exp(wlra.plra(x, rank=rank, max_outer_iters=100, check_converged=True))
    return loss(res[x.mask], x.data[x.mask])
  except RuntimeError:
    return [np.nan for f in losses]

def evaluate_pois_imputation(rank=3, holdout=0.25, eta_max=None, num_trials=10):
  result = []
  for trial in range(num_trials):
    x, eta = simulate_pois(n=200, p=300, rank=rank, eta_max=eta_max,
                           holdout=holdout, seed=trial)
    result.append(list(itertools.chain.from_iterable(
      [[trial],
       imputation_score_mean(x),
       imputation_score_nmf(x, rank),
       imputation_score_plra(x, rank),
       imputation_score_plra1(x, rank),
      ])))
  result = pd.DataFrame(result)
  result.columns = ['trial', 'rmse_mean', 'pois_loss_mean', 'rmse_nmf',
                    'pois_loss_nmf', 'rmse_plra', 'pois_loss_plra',
                    'rmse_plra1', 'pois_loss_plra1']
  return result

def pois_llik(lam, train, test):
  lam *= test.sum(axis=0, keepdims=True) / train.sum(axis=0, keepdims=True)
  return st.poisson(mu=lam).logpmf(test).sum()

def train_test_split(x, p=0.5):
  if ss.issparse(x):
    data = np.random.binomial(n=x.data.astype(np.int), p=p, size=x.data.shape)
    if ss.isspmatrix_csr(x):
      train = ss.csr_matrix((data, x.indices, x.indptr), shape=x.shape)
    elif ss.isspmatrix_csc(x):
      train = ss.csc_matrix((data, x.indices, x.indptr), shape=x.shape)
    else:
      raise NotImplementedError('sparse matrix type not supported')
  else:
    train = np.random.binomial(n=x, p=p, size=x.shape)
  test = x - train
  return train, test

def generalization_score_oracle(train, test, eta):
  return pois_llik(np.exp(eta), train, test)

def generalization_score_plra1(train, test, rank=10, **kwargs):
  try:
    import wlra
    lam = np.exp(wlra.plra(train, rank=rank))
    return pois_llik(lam, train, test)
  except:
    return np.nan

def generalization_score_nmf(train, test, rank=50, **kwargs):
  try:
    from wlra.nmf import nmr
    lam = nmf(train, rank=rank)
    return pois_llik(lam, train, test)
  except:
    return np.nan

def generalization_score_grad(train, test, rank=10, **kwargs):
  try:
    from wlra.grad import PoissonFA
    model = wlra.grad.PoissonFA(n_samples=train.shape[0], n_features=train.shape[1], n_components=rank).fit(train, atol=1e-3, matrain_epochs=10000)
    lam = np.exp(np.exp(m.L.dot(m.F)))
    return pois_llik(lam, train, test)
  except:
    return np.nan

def generalization_score_hpf(train, test, rank=50, **kwargs):
  import scHPF.preprocessing
  import scHPF.train
  import tempfile
  with tempfile.TemporaryDirectory(prefix='/scratch/midway2/aksarkar/ideas/') as d:
    # scHPF assumes genes x cells
    scHPF.preprocessing.split_dataset_hpf(train.T, outdir=d)
    # Set bp, dp as in scHPF.train
    bp = train.sum(axis=1).mean() / train.sum(axis=1).var()
    dp = train.sum(axis=0).mean() / train.sum(axis=0).var()
    opt = scHPF.train.run_trials(
      indir=d, outdir=d, prefix='',
      nfactors=rank, a=0.3, ap=1, bp=bp, c=0.3, cp=1, dp=dp,
      # This is broken when we call the API directly
      logging_options={'log_phi': False})
    L = np.load(f'{opt}/beta_invrate.npy') * np.load(f'{opt}/beta_shape.npy')
    F = np.load(f'{opt}/theta_invrate.npy') * np.load(f'{opt}/theta_shape.npy')
    return pois_llik(L.dot(F.T), train, test)

def generalization_score_scvi(train, test, **kwargs):
  from scvi.dataset import GeneExpressionDataset
  from scvi.inference import UnsupervisedTrainer
  from scvi.models import VAE
  data = GeneExpressionDataset(*GeneExpressionDataset.get_attributes_from_matrix(train))
  vae = VAE(n_input=train.shape[1])
  m = UnsupervisedTrainer(vae, data, verbose=False)
  m.train(n_epochs=100)
  # Training permuted the data for minibatching. Unpermute before "imputing"
  # (estimating lambda)
  lam = np.vstack([m.train_set.sequential().imputation(),
                   m.test_set.sequential().imputation()])
  return pois_llik(lam, train, test)

def generalization_score_dca(train, test, **kwargs):
  import anndata
  import scanpy.api
  data = anndata.AnnData(X=train)
  # "Denoising" is estimating lambda
  scanpy.api.pp.dca(data, mode='denoise')
  lam = data.X
  return pois_llik(lam, train, test)

def get_data_loader(x):
  import torch.utils.data
  if ss.issparse(x):
    x = scaa.dataset.SparseDataset(x)
  else:
    x = torch.tensor(x, dtype=torch.float)
  training_data = torch.utils.data.DataLoader(x, batch_size=25, shuffle=False)

def generalization_score_zipvae(train, test, **kwargs):
  import scaa
  import torch
  training_data = get_data_loader(train)
  with torch.cuda.device(0):
    model = scaa.modules.ZIPVAE(train.shape[1], 10).fit(training_data, lr=1e-2, max_epochs=10, verbose=False)
    lam = model.denoise(training_data)
  return pois_llik(lam, train, test)

def generalization_score_zipaae(train, test, y, **kwargs):
  import scaa
  import torch
  import torch.utils.data
  n, p = train.shape
  training_data = torch.utils.data.DataLoader(torch.tensor(train, dtype=torch.float), batch_size=25, shuffle=False)
  labels = torch.utils.data.DataLoader(torch.tensor(y, dtype=torch.long), batch_size=25, shuffle=False)
  with torch.cuda.device(0):
    model = scaa.modules.ZIPAAE(p, 10, num_classes=(y.max() + 1)).fit(training_data, labels, lr=1e-2, max_epochs=10, verbose=False)
    lam = model.denoise(training_data)
  return pois_llik(lam, train, test)

def read_ipsc():
  keep_samples = pd.read_table('/project2/mstephens/aksarkar/projects/singlecell-qtl/data/quality-single-cells.txt', index_col=0, header=None)
  keep_genes = pd.read_table('/project2/mstephens/aksarkar/projects/singlecell-qtl/data/genes-pass-filter.txt', index_col=0, header=None)
  x = (pd.read_table('/project2/mstephens/aksarkar/projects/singlecell-qtl/data/scqtl-counts.txt.gz', index_col=0)
       .loc[keep_genes.values.ravel(),keep_samples.values.ravel()]
       .values.T)
  return x

def evaluate_generalization(num_trials):
  result = dict()
  for method in ['oracle', 'zipvae']:
    result[method] = []
    for trial in range(num_trials):
      x, eta = simulate_pois(n=500, p=1000, rank=3, eta_max=3, seed=trial)
      train, test = train_test_split(x)
      score = globals()[f'generalization_score_{method}'](train, test, eta=eta)
      result[method].append(score)
  result = pd.DataFrame.from_dict(result)
  result.index.name = 'trial'
  return result
