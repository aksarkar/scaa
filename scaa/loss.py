import torch

def kl_term(qz):
  return torch.distributions.Normal(loc=0, scale=1).log_prob(qz)

def pois_llik(x, mean):
  return x * torch.log(mean) + mean - torch.lgamma(x + 1)

def zip_llik(x, mean, logodds):
  S = torch.nn.functional.softplus
  case_zero = -S(-logodds) + S(pois_llik(x, mean) + S(-logodds))
  case_non_zero = -S(logodds) + pois_llik(x, mean)
  return torch.where(x < 1, case_zero, case_non_zero)

def nb_llik(x, mean, inv_disp):
  """Log likelihood of x distributed as NB

  See Hilbe 2012, eq. 8.10

  mean - mean (> 0)
  inv_disp - inverse dispersion (> 0)

  """
  return (x * torch.log(mean / inv_disp) -
          x * torch.log(1 + mean / inv_disp) -
          inv_disp * torch.log(1 + mean / inv_disp) +
          torch.lgamma(x + inv_disp) -
          torch.lgamma(inv_disp) -
          torch.lgamma(x + 1))

def zinb_llik(x, mean, inv_disp, logodds):
  """Log likelihood of x distributed as ZINB

  See Hilbe 2012, eq. 11.12, 11.13

  mean - mean (> 0)
  inv_disp - inverse dispersion (> 0)
  logodds - logit proportion of excess zeros

  """
  # Important identities:
  # log(x + y) = log(x) + softplus(y - x)
  # log(sigmoid(x)) = -softplus(-x)
  S = torch.nn.functional.softplus
  case_zero = -S(-logodds) + S(nb_llik(x, mean, inv_disp) + S(-logodds))
  case_non_zero = -S(logodds) + nb_llik(x, mean, inv_disp)
  return torch.where(x < 1, case_zero, case_non_zero)
