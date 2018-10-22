import scipy.special as sp
import torch

from .modules import *

class SCAA(torch.nn.Module):
  def __init__(self, input_dim, latent_dim=50, stoch_samples=10):
    # Encode x -> z
    self.enc = Encoder(input_dim, latent_dim, stoch_samples)
    # Decode z -> x
    self.dec = ZIP(latent_dim, input_dim)
    # Generate z -> z (for incremental training)
    self.gen = Generator(latent_dim)
    # Discriminate z -> z (for incremental training)
    self.disc0 = BinaryDiscriminator(latent_dim)
    # Discriminate y (for alignment)
    self.disc1 = MulticlassDiscriminator(latent_dim)

  def forward(self, x, y):
    # VAE component
    qz = self.enc(x)
    logodds, rate = self.dec(qz)

    pois_llik = x * torch.log(rate) - rate + sp.gammaln(x)
    # Important identities:
    # log(x + y) = log(x) + softplus(y - x)
    # log(sigmoid(x)) = -softplus(-x)
    case_zero = -torch.nn.Softplus(-logodds) + torch.nn.Softplus(pois_llik(x, rate) + torch.nn.Softplus(-logodds))
    case_non_zero = -torch.nn.Softplus(logodds) + pois_llik(x, mean)
    zip_llik = torch.where(torch.less(x, 1), case_zero, case_non_zero)

    kl_pz_qz = .5 * (1 + T.log(prec) + prior_prec * (torch.square(mean) + 1 / prec))

    vae_loss = torch.mean(torch.mean(zip_llik) - kl_pz_qz)
    return vae_loss
